"""QRL — Quasimetric RL value learning (Wang, Torralba, Isola, Zhang, ICML 2023,
"Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning").

Why this is different from everything we tried before: TD *regresses* long
distances (and collapses toward the conditional mean when they're ambiguous).
QRL never labels a far pair at all. It only constrains LOCAL one-step
transitions to cost <= 1, then MAXIMALLY SPREADS all other pairs apart. Because
the head is a quasimetric (triangle inequality), any long distance is then
*derived* by composing local steps -- the longest chain of unit steps that fits.

    max_d   E_{(s,g)~random}[ phi(d(s,g)) ]
    s.t.    E_{(s,s')~transitions}[ relu(d(s,s') - c)^2 ] <= eps^2

phi is a bounded, monotonically-increasing, concave "spreading" transform
(phi(x) = -exp(-x/alpha)) so genuinely unreachable pairs can't dominate by
running off to infinity. The constraint is enforced with a learned Lagrange
multiplier via dual ascent (lambda = softplus(raw), so lambda >= 0).

REQUIRES a quasimetric head (iqe / mrn): with an unconstrained MLP there is no
triangle inequality, hence nothing to derive long distances *from*.

Optional `rank_weight` adds the planner-relevant ranking hinge in tandem
(one goal, two states at different true distances) -- QRL fixes the geometry,
the hinge fixes residual ordering.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as logging
from tqdm import tqdm

from ..head import IQEHead, QuasimetricHead
from ..latent_cache import LatentCache


@dataclass
class QRLConfig:
    head: str = "iqe"             # 'iqe' | 'quasimetric' (MRN). MLP is invalid.
    hidden_dim: int = 256
    depth: int = 2
    embed_dim: int = 128
    num_components: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 6000
    seed: int = 0
    # --- QRL specific
    step_cost: float = 1.0        # cost c of one env step
    eps: float = 0.25             # constraint slack: E[relu(d-c)^2] <= eps^2
    spread_temp: float = 100.0    # alpha in phi(x) = -exp(-x/alpha)
    lam_lr: float = 1e-2          # dual ascent rate for the Lagrange multiplier
    lam_init: float = 1.0
    # --- optional ranking hinge (tandem)
    rank_weight: float = 0.0
    rank_margin: float = 0.5
    rank_max_delta: int = 200


def _make_head(cfg, latent_dim):
    if cfg.head == "iqe":
        return IQEHead(latent_dim, hidden_dim=cfg.hidden_dim, embed_dim=cfg.embed_dim,
                       depth=cfg.depth, num_components=cfg.num_components)
    if cfg.head == "quasimetric":
        return QuasimetricHead(latent_dim, hidden_dim=cfg.hidden_dim,
                               embed_dim=cfg.embed_dim, depth=cfg.depth)
    raise ValueError(f"QRL needs a quasimetric head (iqe|quasimetric), got '{cfg.head}'")


def fit(cache: LatentCache, cfg: QRLConfig, device: str = "cpu"):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    value = _make_head(cfg, cache.latent_dim).to(device)
    z = cache.z

    # local transitions: consecutive rows inside an episode (fs1 cache => 1 env step)
    eps_map = cache.episodes()
    keys = [k for k in eps_map if len(eps_map[k]) > 3]
    rows = {k: np.asarray(eps_map[k]) for k in keys}
    src = np.concatenate([rows[k][:-1] for k in keys])
    dst = np.concatenate([rows[k][1:] for k in keys])
    n_trans, n_all = len(src), len(z)
    logging.info(f"QRL: {n_trans} one-step transitions, {n_all} states, head={cfg.head}")

    raw_lam = torch.tensor(float(np.log(np.expm1(cfg.lam_init))), device=device, requires_grad=True)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_lam = torch.optim.Adam([raw_lam], lr=cfg.lam_lr)

    def rank_batch(bs):
        near = np.empty(bs, np.int64); far = np.empty(bs, np.int64)
        goal = np.empty(bs, np.int64); gap = np.empty(bs, np.float32)
        for b in range(bs):
            r = rows[keys[rng.integers(len(keys))]]
            L = len(r)
            hi = min(cfg.rank_max_delta, L - 1)
            d_far = int(rng.integers(2, hi + 1))
            d_near = int(rng.integers(1, d_far))
            gi = int(rng.integers(d_far, L))
            near[b], far[b], goal[b] = r[gi - d_near], r[gi - d_far], r[gi]
            gap[b] = d_far - d_near
        return near, far, goal, torch.from_numpy(gap)

    value.train()
    pbar = tqdm(range(cfg.steps), desc=f"qrl({cfg.head},rank={cfg.rank_weight})")
    for step in pbar:
        # ---- spreading term: push random pairs apart (bounded)
        i = rng.integers(0, n_all, cfg.batch_size)
        j = rng.integers(0, n_all, cfg.batch_size)
        d_rand = value(z[i].to(device), z[j].to(device))
        spread = torch.exp(-d_rand / cfg.spread_temp).mean()   # minimise == maximise d

        # ---- local constraint: one env step must cost <= step_cost
        t = rng.integers(0, n_trans, cfg.batch_size)
        d_loc = value(z[src[t]].to(device), z[dst[t]].to(device))
        viol = F.relu(d_loc - cfg.step_cost).pow(2).mean()

        lam = F.softplus(raw_lam)
        loss = spread + lam.detach() * viol

        if cfg.rank_weight > 0:
            nr, fr, gl, gap = rank_batch(cfg.batch_size)
            v_near = value(z[nr].to(device), z[gl].to(device))
            v_far = value(z[fr].to(device), z[gl].to(device))
            margin = cfg.rank_margin * gap.to(device)
            loss = loss + cfg.rank_weight * F.relu(margin + v_near - v_far).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ---- dual ascent on lambda: grow while the constraint is violated
        lam_loss = -(F.softplus(raw_lam) * (viol.detach() - cfg.eps ** 2))
        opt_lam.zero_grad(set_to_none=True)
        lam_loss.backward()
        opt_lam.step()

        if step % 200 == 0:
            with torch.no_grad():
                pbar.set_postfix(d_rand=float(d_rand.mean()), d_loc=float(d_loc.mean()),
                                 viol=float(viol), lam=float(F.softplus(raw_lam)))
    with torch.no_grad():
        logging.success(
            f"QRL trained (head={cfg.head} rank={cfg.rank_weight}): "
            f"mean d_random={float(d_rand.mean()):.2f} mean d_1step={float(d_loc.mean()):.3f} "
            f"viol={float(viol):.4f} lambda={float(F.softplus(raw_lam)):.2f}")
    value.eval()
    return value
