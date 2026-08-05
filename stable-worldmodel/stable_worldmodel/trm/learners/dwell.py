"""Discounted-dwell goal value: how long you STAY on target, not how fast you arrive.

A quasimetric ``d(z -> g)`` is a *hitting time* -- the optimal cost-to-go of a
stopping problem. Once the goal is touched the problem is over, so the objective
is indifferent to the state you arrive in and cannot prefer settling over
sailing through. That is exactly the latched success metric, and it is the wrong
target once success requires the arm to still be on target at the end: measured
on reacher, the two hitting-time arms (LIP 70.7, TD+CEM 69.3) cluster ~8 points
below latent-MSE (79.3), which implicitly demands the terminal latent to match.

This learner replaces the target with an ordinary goal-conditioned value

    V(z_t, g) = E[ sum_k gamma^k r_{t+k} ],   r = 1{ all joints within tol of g }

so value accrues for every step spent inside the tolerance ball. Arriving early
and remaining is worth strictly more than arriving late, and arriving fast then
overshooting is worth less than arriving slower and settling -- which is the
ordering the evaluation actually rewards.

Two consequences worth stating:

* The head is an unconstrained scalar, not a quasimetric. A discounted return is
  not a metric (no triangle inequality), so imposing one would be a lie about
  the object. The cost is that we lose the stitching that makes quasimetrics
  strong at long range, which is why the h50 cells must be measured, not assumed.
* ``cost()`` must return LOWER = better to satisfy the planner interface, while
  V is higher = better. It returns ``v_max - V`` with ``v_max = 1/(1-gamma)``,
  keeping costs non-negative and comparable in scale to a step count.

Reward needs ground-truth state, so the cache must carry one
(``cache_latents.py --state-key qpos``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from loguru import logger as logging
from torch import nn
from tqdm import tqdm

from ..head import PairwiseMetricHead
from ..latent_cache import LatentCache
from ..samplers import NStepGoalSampler


@dataclass
class DwellConfig:
    steps: int = 6000
    batch_size: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.0
    hidden_dim: int = 256
    depth: int = 2
    n_step: int = 5
    gamma: float = 0.98
    tol: float = 0.05           # per-joint tolerance, matches the eval criterion
    p_cross: float = 0.3
    n_buckets: int = 8
    balanced: bool = True
    max_delta: int | None = None
    tau: float = 0.005
    seed: int = 0
    huber_beta: float = 1.0


class DwellValue(nn.Module):
    """Scalar goal-conditioned value with a lower-is-better ``cost`` adapter."""

    def __init__(self, latent_dim, hidden_dim=256, depth=2, gamma=0.98):
        super().__init__()
        # Declared per-side input width. Load-bearing: the inner
        # PairwiseMetricHead's first Linear is 4*latent_dim wide (it consumes
        # [z_i, z_j, z_i-z_j, |z_i-z_j|]), so a caller that infers the width by
        # reflecting on that layer reads 4x too much and mistakes this value for
        # a 4-frame-stacked metric. nn.Module does not forward attribute lookups
        # to submodules, so this has to be set here explicitly.
        self.latent_dim = int(latent_dim)
        # softplus=False: a discounted return is signed and unconstrained; the
        # non-negativity a metric head enforces would be a wrong prior here
        self.net = PairwiseMetricHead(
            latent_dim, hidden_dim=hidden_dim, depth=depth, softplus=False
        )
        self.v_max = 1.0 / (1.0 - gamma)

    def forward(self, z_i, z_j):
        return self.net(z_i, z_j)

    # no @torch.no_grad(): GradientSolver backprops through the terminal cost
    def cost(self, z_pred, z_goal):
        """Lower is better, as the planner interface requires."""
        return self.v_max - self.forward(z_pred, z_goal)


def fit(cache: LatentCache, cfg: DwellConfig, device: str = "cpu"):
    if cache.state is None:
        raise ValueError(
            'dwell learner needs ground-truth state for the reward; rebuild the '
            'cache with cache_latents.py --state-key qpos'
        )
    torch.manual_seed(cfg.seed)
    value = DwellValue(
        cache.latent_dim, cfg.hidden_dim, cfg.depth, cfg.gamma
    ).to(device)
    target = copy.deepcopy(value).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    state = cache.state.to(device)
    opt = torch.optim.AdamW(
        value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sampler = NStepGoalSampler(
        cache, n_step=cfg.n_step, p_cross=cfg.p_cross, n_buckets=cfg.n_buckets,
        balanced=cfg.balanced, seed=cfg.seed, max_delta=cfg.max_delta,
    )
    g = cfg.gamma
    value.train()
    pbar = tqdm(range(cfg.steps), desc=f'dwell(g={g},tol={cfg.tol})')
    for step in pbar:
        b = sampler.sample(cfg.batch_size)
        z_t = b['z_t'].to(device)
        z_tn = b['z_tn'].to(device)
        z_g = b['z_g'].to(device)
        ne = b['n_eff'].to(device)
        ti = b['t_idx'].to(device)
        gi = b['g_idx'].to(device)

        with torch.no_grad():
            # r_t = 1 iff EVERY joint is inside the ball -- the same all-joints
            # test the environment scores, so the value is trained on the
            # criterion it will be judged by
            in_ball = (
                (state[ti] - state[gi]).abs().amax(dim=-1) < cfg.tol
            ).float()
            # n-step return. Rewards between t and t+n are unknown without
            # walking the interval, so credit the reward at t and bootstrap the
            # rest -- exact for n=1 and a mild underestimate of dwell for n>1.
            disc = g ** ne
            tgt = in_ball + disc * target(z_tn, z_g)

        pred = value(z_t, z_g)
        loss = torch.nn.functional.smooth_l1_loss(
            pred, tgt, beta=cfg.huber_beta
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters()):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * sp)
        if step % 200 == 0:
            pbar.set_postfix(
                loss=loss.item(), v=pred.mean().item(),
                frac_in_ball=in_ball.mean().item(),
            )
    logging.success(
        f'dwell value trained (gamma={g} tol={cfg.tol} n={cfg.n_step}), '
        f'final loss={loss.item():.4f}'
    )
    value.eval()
    return value


__all__ = ['DwellConfig', 'DwellValue', 'fit']
