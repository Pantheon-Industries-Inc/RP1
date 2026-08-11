"""Horizon-matched temporal regression -- the paper's TRM objective.

Train the pairwise head to predict the temporal separation between two states
on the same logged trajectory::

    min_phi  E_{(i,j)}  Huber( m_phi(z_i, z_j),  |t_i - t_j| / s )

with balanced full-horizon pair sampling (see
:class:`~rlp.core.value.samplers.BalancedHorizonPairSampler`), AdamW
(lr 1e-3, wd 1e-4), batch 1024, and scale ``s = 224``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rlp.data import LatentCache
from rlp.logging import logger

from ..head import PairwiseMetricHead
from ..samplers import BalancedHorizonPairSampler


@dataclass
class RegressionConfig:
    hidden_dim: int = 256
    depth: int = 2
    symmetric: bool = False
    scale: float = 224.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 5000
    n_buckets: int = 10
    max_delta: int | None = None  # paper ablation: cap separation (e.g. 50)
    shuffle_labels: bool = False  # negative control
    seed: int = 0
    huber_beta: float = 1.0


def fit(cache: LatentCache, cfg: RegressionConfig, device: str = "cpu") -> PairwiseMetricHead:
    """Train and return a :class:`PairwiseMetricHead`."""
    torch.manual_seed(cfg.seed)
    head = PairwiseMetricHead(
        cache.latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        softplus=True,
        symmetric=cfg.symmetric,
        scale=cfg.scale,
    ).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = BalancedHorizonPairSampler(
        cache,
        n_buckets=cfg.n_buckets,
        max_delta=cfg.max_delta,
        seed=cfg.seed,
    )
    rng = torch.Generator().manual_seed(cfg.seed + 1)

    head.train()
    log_interval = max(1, cfg.steps // 20)
    for step in range(cfg.steps):
        batch = sampler.sample(cfg.batch_size)
        z_i = batch["z_i"].to(device)
        z_j = batch["z_j"].to(device)
        y = (batch["label"] / cfg.scale).to(device)
        if cfg.shuffle_labels:  # negative control: break temporal structure
            perm = torch.randperm(y.shape[0], generator=rng)
            y = y[perm]
        pred = head(z_i, z_j)
        loss = F.smooth_l1_loss(pred, y, beta=cfg.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature.
        opt.step()
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == cfg.steps:
            logger.info(f"Regression training step={step + 1}/{cfg.steps} loss={loss.item():.6f}")
    logger.success(f"Regression head trained ({cfg.steps} steps), final loss={loss.item():.4f}")
    head.eval()
    return head


__all__ = ["RegressionConfig", "fit"]
