"""Offline TD learning of a goal-conditioned reachability (quasi)metric.

A goal-conditioned temporal-distance value ``d(z, z_g)``, learned by n-step
distance TD with hindsight goals (balanced full-horizon and cross-episode),
optionally on a quasimetric head so long cross-room distances stitch:

    n-step target (distance):
        reached within n_eff steps  ->  target = δ            (Monte-Carlo, exact)
        else                        ->  target = c(n_eff) + gamma^n_eff * d_target(z_{t+n}, z_g)
        with c(n_eff) = n_eff (gamma=1) or (1-gamma^n_eff)/(1-gamma)

    min  expectile_Huber( d(z_t, z_g) - stop_grad(target) )

n→∞ recovers Monte-Carlo (= the paper's regression target); n=1 is pure bootstrap.
gamma approaching 1 learns true undiscounted steps-to-go; gamma < 1 discounts long range.
The planner terminal cost is ``d(z_pred, z_goal)`` (lower == closer).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from rp1.core.agent.value.head import IQEHead, PairwiseMetricHead, QuasimetricHead
from rp1.data import LatentCache
from rp1.training.phases.agent.samplers import NStepGoalSampler
from rp1.utils.logging import logger


@dataclass
class TDConfig:
    head: str
    symmetric: bool
    hidden_dim: int
    depth: int
    embed_dim: int
    n_step: int
    gamma: float
    lr: float
    weight_decay: float
    batch_size: int
    steps: int
    save_every: int
    tau: float
    expectile: float
    p_cross: float
    balanced: bool
    max_delta: int | None
    n_buckets: int
    seed: int
    huber_beta: float
    eikonal_weight: float
    num_components: int
    rank_weight: float
    rank_margin: float
    rank_max_delta: int
    softplus: bool
    sym_frac: float
    alpha_init: float
    near_frac: float
    near_max: int


MetricHead = IQEHead | PairwiseMetricHead | QuasimetricHead


def expectile_loss(diff: torch.Tensor, expectile: float, beta: float) -> torch.Tensor:
    """Expectile-weighted Huber loss."""
    huber = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    weight = torch.where(diff > 0, 1.0 - expectile, expectile)  # diff=pred-target
    return (weight * huber).mean()


def _make_head(cfg: TDConfig, latent_dim: int) -> MetricHead:
    if cfg.head == "iqe":
        return IQEHead(
            latent_dim,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_components=cfg.num_components,
            alpha_init=cfg.alpha_init,
        )
    if cfg.head == "quasimetric":
        return QuasimetricHead(
            latent_dim,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            sym_frac=cfg.sym_frac,
        )
    return PairwiseMetricHead(
        latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        softplus=cfg.softplus,
        symmetric=cfg.symmetric,
        scale=1.0,
    )


def fit(
    cache: LatentCache,
    cfg: TDConfig,
    device: str,
    snapshot: Callable[[MetricHead, int], None] | None = None,
) -> MetricHead:
    """Train a temporal-distance head, handing ``snapshot`` a CPU copy every ``save_every`` steps."""
    torch.manual_seed(cfg.seed)
    value = _make_head(cfg, cache.latent_dim).to(device)
    target = copy.deepcopy(value).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = NStepGoalSampler(
        cache,
        n_step=cfg.n_step,
        p_cross=cfg.p_cross,
        n_buckets=cfg.n_buckets,
        balanced=cfg.balanced,
        seed=cfg.seed,
        max_delta=cfg.max_delta,
        near_frac=cfg.near_frac,
        near_max=cfg.near_max,
    )
    if cfg.near_frac > 0:
        logger.info(f"TD near-goal oversampling: frac={cfg.near_frac} max={cfg.near_max} steps")
    g = cfg.gamma
    step_norm = 1.0
    episodes = cache.episodes()
    if cfg.eikonal_weight > 0:
        displacement, count = 0.0, 0
        for rows in list(episodes.values())[:200]:
            episode_z = cache.z[torch.as_tensor(rows)].float()
            displacement += (episode_z[1:] - episode_z[:-1]).norm(dim=-1).sum().item()
            count += len(rows) - 1
        step_norm = max(displacement / max(count, 1), 1e-6)
        logger.info(f"Eikonal mean per-step latent displacement={step_norm:.4f}")

    rank_rng = np.random.default_rng(cfg.seed + 1)
    rank_episodes = [rows for rows in episodes.values() if len(rows) > 3]

    def rank_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        near = np.empty(batch_size, dtype=np.int64)
        far = np.empty(batch_size, dtype=np.int64)
        goal = np.empty(batch_size, dtype=np.int64)
        gaps = np.empty(batch_size, dtype=np.float32)
        for index in range(batch_size):
            rows = rank_episodes[rank_rng.integers(len(rank_episodes))]
            max_distance = min(cfg.rank_max_delta, len(rows) - 1)
            far_distance = int(rank_rng.integers(2, max_distance + 1))
            near_distance = int(rank_rng.integers(1, far_distance))
            goal_index = int(rank_rng.integers(far_distance, len(rows)))
            near[index] = rows[goal_index - near_distance]
            far[index] = rows[goal_index - far_distance]
            goal[index] = rows[goal_index]
            gaps[index] = far_distance - near_distance
        return cache.z[near], cache.z[far], cache.z[goal], torch.from_numpy(gaps)

    value.train()
    log_interval = max(1, cfg.steps // 20)
    for step in range(cfg.steps):
        b = sampler.sample(cfg.batch_size)
        z_t, z_tn, z_g = (
            b["z_t"].to(device),
            b["z_tn"].to(device),
            b["z_g"].to(device),
        )
        ne, reached, dist = (
            b["n_eff"].to(device),
            b["reached"].to(device),
            b["dist"].to(device),
        )
        with torch.no_grad():
            d_next = target(z_tn, z_g)
            if g >= 1.0:
                c, disc = ne, torch.ones_like(ne)
            else:
                disc = g**ne
                c = (1.0 - disc) / (1.0 - g)
            tgt = reached * dist + (1.0 - reached) * (c + disc * d_next)
        pred = value(z_t, z_g)
        loss = expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
        if cfg.rank_weight > 0:
            z_near, z_far, z_rank_goal, gap = (item.to(device) for item in rank_batch(cfg.batch_size))
            rank_loss = torch.relu(
                cfg.rank_margin * gap + value(z_near, z_rank_goal) - value(z_far, z_rank_goal)
            ).mean()
            loss = loss + cfg.rank_weight * rank_loss
        if cfg.eikonal_weight > 0:
            z_input = z_t.detach().requires_grad_(True)
            (gradient,) = torch.autograd.grad(value(z_input, z_g).sum(), z_input, create_graph=True)
            eikonal_loss = ((gradient.norm(dim=-1) * step_norm - 1.0) ** 2).mean()
            loss = loss + cfg.eikonal_weight * eikonal_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature.
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters(), strict=True):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * sp)
        if step == 0 or (step + 1) % log_interval == 0 or step + 1 == cfg.steps:
            logger.info(
                f"TD training step={step + 1}/{cfg.steps} loss={loss.item():.6f} "
                f"prediction_mean={pred.mean().item():.6f}"
            )
        if snapshot is not None and cfg.save_every > 0 and (step + 1) % cfg.save_every == 0 and step + 1 < cfg.steps:
            snapshot(copy.deepcopy(value).cpu().eval(), step + 1)
    logger.success(f"TD trained (n={cfg.n_step} gamma={cfg.gamma} head={cfg.head}), final loss={loss.item():.4f}")
    value.eval()
    return value


__all__ = ["TDConfig", "expectile_loss", "fit"]
