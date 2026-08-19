"""Offline TD learning of a goal-conditioned reachability (quasi)metric.

We learn a goal-conditioned temporal-distance value ``d(z, z_g)`` by **n-step**
distance TD with **HER** hindsight goals (balanced full-horizon + cross-episode),
optionally on a **quasimetric** head so long cross-room distances *stitch*:

    n-step target (distance):
        reached within n_eff steps  ->  target = δ            (Monte-Carlo, exact)
        else                        ->  target = c(n_eff) + γ^{n_eff} · d_target(z_{t+n}, z_g)
        with c(n_eff) = n_eff (γ=1) or (1-γ^{n_eff})/(1-γ)   (discounted step cost)

    min  expectile_Huber( d(z_t, z_g) − stop_grad(target) )

n→∞ recovers Monte-Carlo (= the paper's regression target); n=1 is pure bootstrap.
γ→1 learns true undiscounted steps-to-go (full-horizon); γ<1 discounts long range.
The planner terminal cost is ``d(z_pred, z_goal)`` (lower == closer).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from loguru import logger as logging
from tqdm import tqdm

from ..head import PairwiseMetricHead, QuasimetricHead
from ..latent_cache import LatentCache
from ..samplers import NStepGoalSampler


@dataclass
class TDConfig:
    head: str = "quasimetric"     # 'mlp' | 'quasimetric'
    symmetric: bool = False       # mlp head only: symmetrize d(x,y)
    hidden_dim: int = 256
    depth: int = 2
    embed_dim: int = 128          # quasimetric embedding dim
    n_step: int = 5               # n-step return (n→large ≈ Monte-Carlo)
    gamma: float = 1.0            # 1.0 = undiscounted steps-to-go
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 6000
    tau: float = 0.005
    expectile: float = 0.7        # >0.5 optimistic (shortest-path)
    p_cross: float = 0.3          # fraction of cross-episode (stitching) goals
    balanced: bool = True         # balanced full-horizon hindsight goals
    max_delta: int | None = None  # cap hindsight-goal offsets (horizon matching)
    boundary: str = "legacy"      # n-step seam at gamma<1: legacy keeps the raw
                                  # in-window label against a discounted bootstrap
                                  # (target is NON-MONOTONE at delta=n: e.g.
                                  # g=0.98/n=50 -> V(50)=50 vs V(51)~32); smooth
                                  # uses boot = n_eff + g^n_eff * d_next (continuous,
                                  # per-window discounting, ceiling n/(1-g^n)); disc
                                  # discounts the exact branch too (standard
                                  # discounted quasimetric, (1-g^d)/(1-g))
    n_buckets: int = 10
    seed: int = 0
    huber_beta: float = 1.0


def _expectile_loss(diff, expectile, beta):
    huber = torch.where(diff.abs() < beta, 0.5 * diff.pow(2) / beta, diff.abs() - 0.5 * beta)
    weight = torch.where(diff > 0, 1.0 - expectile, expectile)  # diff=pred-target
    return (weight * huber).mean()


def _make_head(cfg, latent_dim):
    if cfg.head == "quasimetric":
        return QuasimetricHead(latent_dim, hidden_dim=cfg.hidden_dim, embed_dim=cfg.embed_dim, depth=cfg.depth)
    return PairwiseMetricHead(latent_dim, hidden_dim=cfg.hidden_dim, depth=cfg.depth, softplus=True, symmetric=cfg.symmetric)


def fit(cache: LatentCache, cfg: TDConfig, device: str = "cpu"):
    """Train and return a temporal-distance (quasi)metric head."""
    torch.manual_seed(cfg.seed)
    value = _make_head(cfg, cache.latent_dim).to(device)
    target = copy.deepcopy(value).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = NStepGoalSampler(cache, n_step=cfg.n_step, p_cross=cfg.p_cross,
                               n_buckets=cfg.n_buckets, balanced=cfg.balanced, seed=cfg.seed,
                               max_delta=cfg.max_delta)
    g = cfg.gamma
    value.train()
    pbar = tqdm(range(cfg.steps), desc=f"td(n={cfg.n_step},g={cfg.gamma},{cfg.head})")
    for step in pbar:
        b = sampler.sample(cfg.batch_size)
        z_t, z_tn, z_g = b["z_t"].to(device), b["z_tn"].to(device), b["z_g"].to(device)
        ne, reached, dist = b["n_eff"].to(device), b["reached"].to(device), b["dist"].to(device)
        with torch.no_grad():
            d_next = target(z_tn, z_g)
            if g >= 1.0:
                c, disc = ne, torch.ones_like(ne)
            else:
                disc = g ** ne
                c = ne if cfg.boundary == "smooth" else (1.0 - disc) / (1.0 - g)
            boot = c + disc * d_next
            dist_t = dist
            if g < 1.0 and cfg.boundary == "disc":
                dist_t = (1.0 - g ** dist) / (1.0 - g)
            tgt = reached * dist_t + (1.0 - reached) * boot
        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters()):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * sp)
        if step % 200 == 0:
            pbar.set_postfix(loss=loss.item(), pred=pred.mean().item())
    logging.success(f"TD trained (n={cfg.n_step} γ={cfg.gamma} head={cfg.head}), final loss={loss.item():.4f}")
    value.eval()
    return value


__all__ = ["TDConfig", "fit", "_expectile_loss"]
