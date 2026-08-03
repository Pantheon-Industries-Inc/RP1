"""Offline quality screen for a trained TD metric, on HELD-OUT episodes.

A full TD+CEM eval costs ~33 min, so screening a 12-cell sweep that way is
~7 h. TD training itself is 95 s and reports only a final loss, which is not
comparable across expectiles (different objectives => different loss scales),
so it cannot be used to rank configs.

This measures what a planner actually needs from a cost-to-go metric: does
d(z_t, z_goal) order states by their true remaining steps?

  spearman   rank correlation between d(z_t, z_g) and true steps-to-goal
  pair_acc   P(d(z_i,g) < d(z_j,g)) for random pairs where i is genuinely
             closer to g than j -- a direct "would the planner prefer the
             better state" score. 0.5 = chance.
  monotone   mean fraction of consecutive steps along a trajectory where the
             distance to a fixed later goal decreases

Pairs are drawn ONLY from episodes >= --ep-lo, i.e. episodes no stage of
training ever saw. This is a screen, not a substitute for the real eval: the
top configs still get scored by TD+CEM.

Usage: probe_metric.py --cache FULL_FS1.pt --metric TD.pt [--ep-lo 8000]
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, load_metric  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--metric', required=True)
    p.add_argument('--ep-lo', type=int, default=8000,
                   help='use only episodes >= this (held out from training)')
    p.add_argument('--n-pairs', type=int, default=20000)
    p.add_argument('--max-delta', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    dev = args.device
    cache = LatentCache.load(args.cache)
    metric = load_metric(args.metric).to(dev).eval()

    ep = cache.episode_idx.numpy().reshape(-1)
    st = cache.step_idx.numpy().reshape(-1)
    keep = np.nonzero(ep >= args.ep_lo)[0]
    if len(keep) == 0:
        sys.exit(f'no rows with episode >= {args.ep_lo}')

    z = cache.z[keep].float()
    ep, st = ep[keep], st[keep]
    # row lookup within the kept slice: (episode, step) -> local index
    order = np.lexsort((st, ep))
    z, ep, st = z[order], ep[order], st[order]
    ep_start = {}
    for i, e in enumerate(ep):
        if e not in ep_start:
            ep_start[e] = i
    ep_len = {e: int((ep == e).sum()) for e in np.unique(ep)}

    rng = np.random.default_rng(args.seed)
    eps = np.array(sorted(ep_start))

    # ---- sample (state, goal) pairs within episodes, goal strictly later
    s_idx, g_idx, delta = [], [], []
    while len(s_idx) < args.n_pairs:
        e = eps[rng.integers(len(eps))]
        L = ep_len[e]
        if L < 3:
            continue
        a = rng.integers(0, L - 1)
        d = rng.integers(1, min(args.max_delta, L - 1 - a) + 1)
        s_idx.append(ep_start[e] + a)
        g_idx.append(ep_start[e] + a + d)
        delta.append(d)
    s_idx = np.asarray(s_idx); g_idx = np.asarray(g_idx)
    delta = np.asarray(delta, dtype=np.float64)

    with torch.no_grad():
        d_pred = []
        for i in range(0, len(s_idx), 8192):
            zs = z[s_idx[i:i + 8192]].to(dev)
            zg = z[g_idx[i:i + 8192]].to(dev)
            d_pred.append(metric.cost(zs, zg).float().cpu().numpy().reshape(-1))
    d_pred = np.concatenate(d_pred)

    finite = np.isfinite(d_pred)
    d_pred, delta_f = d_pred[finite], delta[finite]

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / den) if den > 0 else float('nan')

    rho = spearman(d_pred, delta_f)

    # ---- pair accuracy: same goal, two states at different true distances
    n = len(d_pred) // 2
    i1, i2 = np.arange(n), np.arange(n, 2 * n)
    diff = delta_f[i1] - delta_f[i2]
    ok = diff != 0
    closer_is_smaller = (
        (d_pred[i1][ok] < d_pred[i2][ok]) == (diff[ok] < 0)
    )
    pair_acc = float(closer_is_smaller.mean())

    # ---- monotonicity along trajectories toward a fixed goal
    mono = []
    for _ in range(300):
        e = eps[rng.integers(len(eps))]
        L = ep_len[e]
        if L < 12:
            continue
        g = ep_start[e] + L - 1
        rows = np.arange(ep_start[e], ep_start[e] + min(L - 1, 60))
        with torch.no_grad():
            dd = metric.cost(
                z[rows].to(dev), z[g].to(dev).expand(len(rows), -1)
            ).float().cpu().numpy().reshape(-1)
        if np.isfinite(dd).all() and len(dd) > 1:
            mono.append(float((np.diff(dd) < 0).mean()))
    mono = float(np.mean(mono)) if mono else float('nan')

    print(f'PROBE{(" " + args.tag) if args.tag else ""} '
          f'spearman={rho:.4f} pair_acc={pair_acc:.4f} monotone={mono:.4f} '
          f'n={len(d_pred)} eps>={args.ep_lo}')


if __name__ == '__main__':
    main()
