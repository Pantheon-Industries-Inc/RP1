"""Offline goal-conditioned reward head ``R(z, z_g) -> [0, 1]``.

PWM (arXiv 2407.02466) trains encoder ``E_phi``, dynamics ``F_phi`` and a reward
predictor ``R_phi`` as "supervised learning problems with fixed targets", then
extracts a policy by first-order optimisation through the model, using
``R_phi`` to score imagined latents. Our world models ship with ``E`` and ``F``
but no reward head, so this supplies the missing component.

Why the label is computed rather than read from the dataset
-----------------------------------------------------------
The evaluation protocol replays dataset rows: start = row ``t``, goal = row
``t + offset``. The dataset's own ``reward`` / ``distance_to_target`` columns
refer to the *episode's* target, which is NOT the replayed goal, so they are the
wrong signal for this task distribution. The correct label is the protocol's own
success criterion, which is a known function of the oracle state:

    r(s_t, g) = 1[ ||state(t) - state(g)||_2 < success_radius ]

`cache_latents.py --state-key pos_agent` stores exactly that state alongside
every latent, so the label is exact and no inverse RL is required -- this is
plain supervised classification on cached latent pairs.

The head consumes the same ``pair_features`` map as the metric heads and shares
the HER sampler, so positives are not vanishingly rare: hindsight goals are
drawn at a controlled offset distribution rather than uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger as logging
from torch import nn
from tqdm import tqdm

from ..head import pair_features
from ..latent_cache import LatentCache
from ..samplers import NStepGoalSampler


@dataclass
class RewardConfig:
    hidden_dim: int = 256
    depth: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 4000
    success_radius: float = 16.0  # TwoRoom: env terminates within 16 px
    n_step: int = 50              # only to reuse NStepGoalSampler's goal draw
    p_cross: float = 0.3
    balanced: bool = True
    max_delta: int | None = None
    n_buckets: int = 10
    seed: int = 0
    pos_weight: float | None = None  # None -> computed from the sampled label rate


class RewardHead(nn.Module):
    """``[z, z_g, z - z_g, |z - z_g|] -> MLP -> logit``.

    ``forward`` returns a logit; ``reward`` returns the calibrated probability in
    [0, 1], which is what the planner consumes.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, depth: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        layers: list[nn.Module] = [nn.Linear(4 * latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        return self.mlp(pair_features(z, z_g)).squeeze(-1)

    def reward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        """Probability of being at the goal. Differentiable in ``z``."""
        return torch.sigmoid(self.forward(z, z_g))


def fit(cache: LatentCache, cfg: RewardConfig, device: str = 'cpu') -> RewardHead:
    """Fit ``R(z, z_g)`` against the protocol's own success criterion."""
    if cache.state is None:
        raise ValueError(
            'reward learner needs oracle states: rebuild the cache with '
            'cache_latents.py --state-key <agent-position column> '
            '(pos_agent for the canonical TwoRoom h5)'
        )
    torch.manual_seed(cfg.seed)
    head = RewardHead(cache.latent_dim, cfg.hidden_dim, cfg.depth).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sampler = NStepGoalSampler(
        cache, n_step=cfg.n_step, p_cross=cfg.p_cross, n_buckets=cfg.n_buckets,
        balanced=cfg.balanced, seed=cfg.seed, max_delta=cfg.max_delta,
    )
    state = cache.state.to(device).float()

    # positive rate is low even with HER, so rebalance the loss rather than the
    # sampler (keeping the goal distribution identical to the critic's)
    pos_weight = cfg.pos_weight
    if pos_weight is None:
        with torch.no_grad():
            b = sampler.sample(min(8192, cfg.batch_size * 8))
            lab = _label(state, b, cfg.success_radius, device)
            rate = lab.mean().clamp_min(1e-6)
            pos_weight = float(((1.0 - rate) / rate).clamp(1.0, 200.0))
        logging.info(f'reward head: positive rate {rate.item():.4f} '
                     f'-> pos_weight {pos_weight:.1f}')

    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    head.train()
    pbar = tqdm(range(cfg.steps), desc=f'reward(r<{cfg.success_radius:g})')
    for step in pbar:
        b = sampler.sample(cfg.batch_size)
        z, z_g = b['z_t'].to(device), b['z_g'].to(device)
        lab = _label(state, b, cfg.success_radius, device)
        loss = lossf(head(z, z_g), lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                acc = ((head.reward(z, z_g) > 0.5).float() == lab).float().mean()
            pbar.set_postfix(loss=loss.item(), acc=acc.item(),
                             pos=lab.mean().item())
    logging.success(f'reward head trained (final loss {loss.item():.4f})')
    head.eval()
    return head


def _label(state: torch.Tensor, batch: dict, radius: float, device: str) -> torch.Tensor:
    """1 iff the sampled state is within ``radius`` of the sampled goal state."""
    t = batch['t_idx'] if 't_idx' in batch else batch['idx_t']
    g = batch['g_idx'] if 'g_idx' in batch else batch['idx_g']
    s_t = state[t.to(device)]
    s_g = state[g.to(device)]
    return (torch.linalg.norm(s_t - s_g, dim=-1) < radius).float()


__all__ = ['RewardConfig', 'RewardHead', 'fit']
