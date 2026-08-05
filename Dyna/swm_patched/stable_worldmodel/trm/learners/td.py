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

import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm

from ..head import IQEHead, PairwiseMetricHead, QuasimetricHead
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
    n_buckets: int = 10
    seed: int = 0
    huber_beta: float = 1.0
    eikonal_weight: float = 0.0   # Eik-HIQL unit-gradient penalty (0 = off)
    td_weight: float = 1.0        # 0 = pure ranking (no distance regression)
    rank_weight: float = 0.0      # pairwise ranking hinge (0 = off)
    rank_margin: float = 0.5      # margin per step of true separation
    rank_max_delta: int = 200     # max goal distance for ranking triplets
    num_components: int = 8       # iqe head only


def _expectile_loss(diff, expectile, beta):
    huber = torch.where(diff.abs() < beta, 0.5 * diff.pow(2) / beta, diff.abs() - 0.5 * beta)
    weight = torch.where(diff > 0, 1.0 - expectile, expectile)  # diff=pred-target
    return (weight * huber).mean()


def _make_head(cfg, latent_dim):
    if cfg.head == "iqe":
        return IQEHead(latent_dim, hidden_dim=cfg.hidden_dim, embed_dim=cfg.embed_dim,
                       depth=cfg.depth, num_components=cfg.num_components)
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

    if cfg.rank_weight > 0:
        # planner-relevant triplets: ONE goal, two states at different true
        # distances from it. (The TD sampler gives one state + one goal, which
        # is the wrong shape for this comparison.)
        _rk_eps = cache.episodes()
        _rk_keys = [k for k in _rk_eps if len(_rk_eps[k]) > 3]
        _rk_rows = {k: np.asarray(_rk_eps[k]) for k in _rk_keys}
        _rk_rng = np.random.default_rng(cfg.seed + 1)

        def _rank_batch(bs):
            near = np.empty(bs, np.int64); far = np.empty(bs, np.int64)
            goal = np.empty(bs, np.int64); gap = np.empty(bs, np.float32)
            for b in range(bs):
                r = _rk_rows[_rk_keys[_rk_rng.integers(len(_rk_keys))]]
                L = len(r)
                hi = min(cfg.rank_max_delta, L - 1)
                d_far = int(_rk_rng.integers(2, hi + 1))
                d_near = int(_rk_rng.integers(1, d_far))
                gi = int(_rk_rng.integers(d_far, L))
                near[b], far[b], goal[b] = r[gi - d_near], r[gi - d_far], r[gi]
                gap[b] = d_far - d_near
            return (cache.z[near], cache.z[far], cache.z[goal],
                    torch.from_numpy(gap))


    step_norm = 1.0
    if cfg.eikonal_weight > 0:
        # mean ||z_{t+1} - z_t|| over consecutive rows within episodes: converts
        # the Eikonal target from "per unit latent" to "per ENV STEP".
        _eps = cache.episodes()
        _d, _n = 0.0, 0
        for _rws in list(_eps.values())[:200]:
            _r = torch.as_tensor(list(_rws))
            _zz = cache.z[_r].float()
            _d += (_zz[1:] - _zz[:-1]).norm(dim=-1).sum().item()
            _n += len(_r) - 1
        step_norm = max(_d / max(_n, 1), 1e-6)
        logging.info(f"eikonal: mean per-step latent displacement = {step_norm:.4f}")

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
                c = (1.0 - disc) / (1.0 - g)
            boot = c + disc * d_next
            tgt = reached * dist + (1.0 - reached) * boot
        pred = value(z_t, z_g)
        loss = cfg.td_weight * _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
        if cfg.rank_weight > 0:
            zn, zf, zg_r, gap = _rank_batch(cfg.batch_size)
            zn, zf, zg_r, gap = zn.to(device), zf.to(device), zg_r.to(device), gap.to(device)
            v_near, v_far = value(zn, zg_r), value(zf, zg_r)
            margin = cfg.rank_margin * gap
            loss = loss + cfg.rank_weight * torch.relu(margin + v_near - v_far).mean()
        if cfg.eikonal_weight > 0:
            # ||grad_z V(z, g)|| * step_norm == 1  <=>  V changes by ~1 per env step
            z_in = z_t.detach().requires_grad_(True)
            v_in = value(z_in, z_g)
            (grad_z,) = torch.autograd.grad(v_in.sum(), z_in, create_graph=True)
            gnorm = grad_z.norm(dim=-1)
            eik = ((gnorm * step_norm - 1.0) ** 2).mean()
            loss = loss + cfg.eikonal_weight * eik
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
