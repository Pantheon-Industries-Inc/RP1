"""Offline TD learning of a goal-conditioned reachability (quasi)metric.

We learn a goal-conditioned temporal-distance value ``d(z, z_g)`` by **n-step**
distance TD with **HER** hindsight goals (balanced full-horizon + cross-episode),
optionally on a **quasimetric** head so long cross-room distances *stitch*:

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
from dataclasses import dataclass

import numpy as np
import torch

from rlp.data import LatentCache
from rlp.logging import logger

from ..head import IQEHead, PairwiseMetricHead, QuasimetricHead
from ..samplers import NStepGoalSampler


@dataclass
class TDConfig:
    head: str = "quasimetric"  # 'mlp' | 'quasimetric' | 'iqe'
    symmetric: bool = False  # mlp head only: symmetrize d(x,y)
    hidden_dim: int = 256
    depth: int = 2
    embed_dim: int = 128  # quasimetric embedding dim
    n_step: int = 5  # n-step return (n→large ≈ Monte-Carlo)
    gamma: float = 1.0  # 1.0 = undiscounted steps-to-go
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 6000
    tau: float = 0.005
    expectile: float = 0.7  # >0.5 optimistic (shortest-path)
    p_cross: float = 0.3  # fraction of cross-episode (stitching) goals
    balanced: bool = True  # balanced full-horizon hindsight goals
    max_delta: int | None = None  # cap hindsight-goal offsets (horizon matching)
    n_buckets: int = 10
    seed: int = 0
    huber_beta: float = 1.0
    eikonal_weight: float = 0.0
    num_components: int = 8
    rank_weight: float = 0.0
    rank_margin: float = 0.5
    rank_max_delta: int = 200
    # counterfactual agent augmentation (PushT): with prob ``aug_p`` the query
    # state z_t is swapped for its agent-displaced re-render and the label
    # grows by ``aug_transit_scale * transit`` (steps to walk the agent back)
    aug_p: float = 0.0
    aug_transit_scale: float = 1.0


MetricHead = IQEHead | PairwiseMetricHead | QuasimetricHead

# (z_aug, transit): row-aligned agent-displaced latents (already windowed like
# the training cache) and the per-row free-transit cost in primitive steps
type AugCache = tuple[torch.Tensor, torch.Tensor]


def _expectile_loss(diff: torch.Tensor, expectile: float, beta: float) -> torch.Tensor:
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
        )
    if cfg.head == "quasimetric":
        return QuasimetricHead(
            latent_dim,
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
        )
    return PairwiseMetricHead(
        latent_dim,
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        softplus=True,
        symmetric=cfg.symmetric,
    )


def fit(cache: LatentCache, cfg: TDConfig, device: str = "cpu", aug: AugCache | None = None) -> MetricHead:
    """Train and return a temporal-distance (quasi)metric head.

    ``aug`` enables the counterfactual agent augmentation: ``(z_aug, transit)``
    row-aligned with ``cache`` (see :mod:`rlp.tools.data.cache_agent_aug`).
    A fraction ``cfg.aug_p`` of every batch's query states is replaced by the
    displaced-agent latent; its target is the original label plus the transit
    cost — the cost of walking the agent back and then following the data.
    """
    torch.manual_seed(cfg.seed)
    use_aug = aug is not None and cfg.aug_p > 0
    if use_aug and aug is not None and (len(aug[0]) != len(cache.z) or len(aug[1]) != len(cache.z)):
        raise ValueError("aug latents/transit must be row-aligned with the training cache")
    aug_rng = np.random.default_rng(cfg.seed + 7)
    if use_aug:
        logger.info(f"TD counterfactual agent augmentation: p={cfg.aug_p} transit_scale={cfg.aug_transit_scale}")
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
    )
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
        z_t_cpu = b["z_t"]
        transit = torch.zeros(cfg.batch_size)
        if use_aug and aug is not None:
            z_aug, transit_all = aug
            mask = torch.from_numpy(aug_rng.random(cfg.batch_size) < cfg.aug_p)
            if bool(mask.any()):
                t_aug = b["t_idx"][mask]
                z_t_cpu = z_t_cpu.clone()
                z_t_cpu[mask] = z_aug[t_aug].float()
                transit[mask] = transit_all[t_aug].float() * cfg.aug_transit_scale
        z_t, z_tn, z_g = (
            z_t_cpu.to(device),
            b["z_tn"].to(device),
            b["z_g"].to(device),
        )
        ne, reached, dist = (
            b["n_eff"].to(device),
            b["reached"].to(device),
            b["dist"].to(device),
        )
        transit = transit.to(device)
        with torch.no_grad():
            d_next = target(z_tn, z_g)
            ne_total = ne + transit  # displaced queries first walk the agent back
            if g >= 1.0:
                c, disc = ne_total, torch.ones_like(ne)
            else:
                disc = g**ne_total
                c = (1.0 - disc) / (1.0 - g)
            boot = c + disc * d_next
            tgt = reached * (dist + transit) + (1.0 - reached) * boot
        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
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
    logger.success(f"TD trained (n={cfg.n_step} gamma={cfg.gamma} head={cfg.head}), final loss={loss.item():.4f}")
    value.eval()
    return value


__all__ = ["AugCache", "TDConfig", "fit", "_expectile_loss"]
