"""Train a MACRO-LEVEL quasimetric value over HWM waypoint latents.

Why this exists
---------------
Every high-level value in the 2026-07 HWM campaign was an *env-step* critic:
``train_lip_hl.py`` trains its critic on the dense fs1 cache with targets in
primitive steps, and only the ACTOR lived in macro space. The high-level CEM
path never used a learned value at all — ``solver/hlip.py::_cem_search`` scores
candidates with raw terminal latent MSE. So "hierarchical CEM + value" has
never actually been deployed.

That matters most at h200. RESULTS_hwm.md F7 read the h200 floor (every family
<= 2.0) as a data-support boundary: episodes are 201 primitive steps, so no
trajectory demonstrates a 200-step task and the env-step critic can only reach
200 by chaining n-step-50 bootstraps, which saturate. **At macro granularity
that argument does not hold.** With stride K=25 a 201-step episode is 9
waypoints, and a 200-step goal is a distance-**8** query — a pair that occurs
in-episode in every single episode, learnable with n_step=8 and no bootstrap
chain at all. Whether the h200 floor is really about data or merely about the
units the value was trained in is therefore an open, testable question, and
this trainer is the instrument.

What it produces
----------------
A standard ``kind='td'`` metric checkpoint (MRN ``QuasimetricHead`` by default)
over 192-d waypoint latents, with distances in MACRO steps. It loads with
``rlp.core.value.io.load_metric`` and is consumed by ``HLIPSolver`` via
``hl_value_path`` + ``hl_cost=value`` (see 0001-hl-cem-value-cost.patch).

Usage
-----
    pixi run python scripts/hierarchy/train_hl_value.py \
        --cache /workspace/caches/cube_v2wm_fs1.pt \
        --out /workspace/metrics/hl_value_s25_cube.pt \
        --stride 25 --max-delta 8 --n-step 8 --steps 6000

Protocol notes
--------------
* Distances are macro steps; the deployed planner's horizon is in the same
  units, so ``hl_cutoff`` / energies read directly.
* ``--holdout-ep`` mirrors the HWM trainer so the value never sees the eval
  episodes when a split is in force.
* gamma defaults to 1.0: undiscounted steps-to-go. At K=25 the whole episode is
  8 macro steps, far inside any sane 1/(1-gamma) ceiling, so discounting buys
  nothing and only risks the saturation that cost TwoRoom 39 points at h100.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from rlp.core.value.io import build_metric, save_metric
from rlp.core.value.learners.td import _expectile_loss
from rlp.data import LatentCache


def _cos(base, final, t, T):
    """Cosine decay base -> final over T steps; constant when final is None."""
    if final is None or T <= 0:
        return base
    t = min(t, T)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t / T))


def build_waypoints(cache, stride, holdout_ep=-1):
    """Row indices of the waypoint grid, one array per episode.

    Waypoints sit at primitive steps 0, K, 2K, ... within each episode, so a
    201-step cube episode at K=25 yields 9 waypoints spanning 8 macro steps.
    """
    eps = cache.episodes()
    out = {}
    for ep_id in sorted(eps):
        if holdout_ep > 0 and ep_id >= holdout_ep:
            continue
        rows = eps[ep_id]
        picked = rows[::stride]
        if len(picked) >= 2:
            out[ep_id] = picked
    return out


class MacroPairSampler:
    """In-episode waypoint pairs at macro separation, plus cross-episode goals.

    Returns, per batch: ``z_t`` (start waypoint), ``z_tn`` (the waypoint
    ``n_eff`` macro steps later, for bootstrapping), ``z_g`` (goal waypoint),
    ``n_eff`` (macro steps from t to tn), ``dist`` (macro steps from t to the
    goal when the goal IS the n-step landing point or nearer), and ``reached``
    (whether ``dist`` is exact rather than a bootstrap).

    Balanced buckets over delta keep the long separations — the ones that decide
    h200 — from being swamped by the short ones.
    """

    def __init__(self, cache, waypoints, n_step, max_delta, p_cross, seed=0, n_buckets=8):
        self.z = cache.z
        self.wp = waypoints
        self.keys = np.array(sorted(waypoints.keys()))
        self.n_step = int(n_step)
        self.max_delta = int(max_delta)
        self.p_cross = float(p_cross)
        self.rng = np.random.default_rng(seed)
        self.n_buckets = int(n_buckets)

    def sample(self, batch):
        rows_t, rows_tn, rows_g = [], [], []
        n_eff, dist, reached = [], [], []
        # balanced delta buckets across [1, max_delta]
        edges = np.linspace(1, self.max_delta + 1, self.n_buckets + 1)
        for _ in range(batch):
            ep = int(self.rng.choice(self.keys))
            w = self.wp[ep]
            L = len(w)
            if L < 2:
                continue
            b = int(self.rng.integers(self.n_buckets))
            lo, hi = edges[b], edges[b + 1]
            delta = int(np.clip(round(self.rng.uniform(lo, hi)), 1, min(self.max_delta, L - 1)))
            i = int(self.rng.integers(0, L - delta))
            j = i + delta
            k = min(i + self.n_step, L - 1)  # n-step landing waypoint
            rows_t.append(w[i])
            rows_tn.append(w[k])
            n_eff.append(k - i)
            if self.rng.random() < self.p_cross:
                # cross-episode HER goal: no exact distance, always bootstrap
                ep2 = int(self.rng.choice(self.keys))
                w2 = self.wp[ep2]
                rows_g.append(w2[int(self.rng.integers(0, len(w2)))])
                dist.append(0.0)
                reached.append(0.0)
            else:
                rows_g.append(w[j])
                dist.append(float(delta))
                # exact only when the goal is at or before the n-step landing
                reached.append(1.0 if delta <= (k - i) else 0.0)

        idx = lambda r: self.z[torch.as_tensor(np.asarray(r), dtype=torch.long)]
        return {
            "z_t": idx(rows_t),
            "z_tn": idx(rows_tn),
            "z_g": idx(rows_g),
            "n_eff": torch.tensor(n_eff, dtype=torch.float32),
            "dist": torch.tensor(dist, dtype=torch.float32),
            "reached": torch.tensor(reached, dtype=torch.float32),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="fs1 (dense) LatentCache")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=25, help="env steps per macro")
    p.add_argument("--max-delta", type=int, default=8, help="max goal separation, MACRO steps")
    p.add_argument("--n-step", type=int, default=8, help="TD backup span, MACRO steps")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-final", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--expectile", type=float, default=0.1)
    p.add_argument("--expectile-final", type=float, default=0.03)
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--ema-tau", type=float, default=0.005)
    p.add_argument("--head", choices=["quasimetric", "mlp"], default="quasimetric")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--holdout-ep", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)

    cache = LatentCache.load(a.cache)
    assert (cache.meta or {}).get("stride", 1) == 1, "need the fs1 (dense) cache"
    cache.z = cache.z.to(dev).float()

    wp = build_waypoints(cache, a.stride, a.holdout_ep)
    spans = np.array([len(v) - 1 for v in wp.values()])
    print(
        f"{len(wp)} episodes, waypoints/episode {spans.min() + 1}-{spans.max() + 1}, "
        f"max macro separation {spans.max()} (= {spans.max() * a.stride} env steps)",
        flush=True,
    )
    if spans.max() < a.max_delta:
        print(
            f"WARNING: max_delta={a.max_delta} exceeds the longest in-episode "
            f"separation ({spans.max()}); those buckets will clamp.",
            flush=True,
        )

    arch = {
        "head": a.head,
        "hidden_dim": a.hidden_dim,
        "embed_dim": a.embed_dim,
        "depth": a.depth,
        "softplus": True,
        "symmetric": False,
    }
    critic = build_metric("td", cache.latent_dim, arch).to(dev)
    teacher = build_metric("td", cache.latent_dim, arch).to(dev)
    teacher.load_state_dict(critic.state_dict())
    for q in teacher.parameters():
        q.requires_grad_(False)

    opt = torch.optim.AdamW(critic.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sampler = MacroPairSampler(
        cache, wp, n_step=a.n_step, max_delta=a.max_delta, p_cross=a.p_cross, seed=a.seed
    )

    for step in range(a.steps):
        tau = _cos(a.expectile, a.expectile_final, step, a.steps)
        lr = _cos(a.lr, a.lr_final, step, a.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        b = sampler.sample(a.batch)
        z_t, z_tn, z_g = b["z_t"], b["z_tn"], b["z_g"]
        ne, dist, reached = b["n_eff"].to(dev), b["dist"].to(dev), b["reached"].to(dev)

        with torch.no_grad():
            d_next = teacher(z_tn, z_g)
            if a.gamma >= 1.0:
                cost, disc = ne, torch.ones_like(ne)
            else:
                disc = a.gamma**ne
                cost = (1.0 - disc) / (1.0 - a.gamma)
            tgt = reached * dist + (1.0 - reached) * (cost + disc * d_next)

        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), critic.parameters()):
                tp.mul_(1.0 - a.ema_tau).add_(a.ema_tau * sp)

        if step % 500 == 0 or step == a.steps - 1:
            with torch.no_grad():
                # diagnostics that matter for h200: is V monotone in separation,
                # and is V(g,g) still ~0?
                probe = sampler.sample(512)
                pv = teacher(probe["z_t"], probe["z_g"])
                same = teacher(probe["z_g"], probe["z_g"]).mean().item()
                far = probe["dist"].to(dev)
                m = far > 0
                corr = (
                    float(np.corrcoef(pv[m].cpu().numpy(), far[m].cpu().numpy())[0, 1])
                    if m.sum() > 2
                    else float("nan")
                )
            print(
                f"step {step}: loss {loss.item():.4f} tau {tau:.3f} "
                f"V(g,g) {same:.3f} corr(V,delta) {corr:.3f}",
                flush=True,
            )

    # RLP's save_metric writes a Stable-WM weights+config directory:
    # save_metric(module, run_name=..., cache_dir=...) -> Path. Treat --out as
    # <cache_dir>/<run_name> so the result loads with load_metric(path).
    out = Path(a.out)
    saved = save_metric(teacher.cpu(), run_name=out.name, cache_dir=str(out.parent))
    print(f"saved {saved}", flush=True)


if __name__ == "__main__":
    main()
