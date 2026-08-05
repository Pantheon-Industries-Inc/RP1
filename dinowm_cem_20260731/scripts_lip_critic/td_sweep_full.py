"""TD teacher sweep on the FULL DINO-WM token grid (75,264-d, no reduction).

One process for the whole sweep: the 484 GB train cache is loaded once (~20 min
at the measured 402 MB/s) and every cell trains against it in RAM, so the load
is paid once instead of per cell. The held-out cache (121 GB, episodes >= 8000)
stays resident too, so each cell is probed immediately after training.

Cells are ordered most-informative-first (canonical recipe, then the winner of
the mean-pooled screen), and each cell's metric + CSV row is written as soon as
it finishes -- so the run can be stopped at any point without losing results.

Axes: --expectile (the IQL tau; low = optimistic toward the min, what a
cost-to-go wants) x --lr. NB TDConfig.tau is the target-network Polyak rate,
NOT the expectile.

Reference points (mean-pooled 384-d substrate, same recipe):
    best  expectile 0.1  lr 1e-3 -> spearman 0.437 pair_acc 0.666 monotone 0.575
    canon expectile 0.03 lr 1e-3 -> spearman 0.413 pair_acc 0.653 monotone 0.638
    v2WM LeWM teacher            -> spearman 0.484 pair_acc 0.691 monotone 0.923
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, learners, save_metric  # noqa: E402
from stable_worldmodel.trm.learners.td import TDConfig  # noqa: E402

# most informative first, so an early stop still yields the useful cells
GRID = [
    (0.03, 1e-3), (0.1, 1e-3), (0.03, 3e-4), (0.1, 3e-4),
    (0.01, 1e-3), (0.5, 1e-3), (0.01, 3e-4), (0.5, 3e-4),
    (0.03, 3e-3), (0.1, 3e-3), (0.01, 3e-3), (0.5, 3e-3),
]


def probe(metric, z, ep, st, dev, n_pairs=20000, max_delta=100, seed=0):
    """Spearman / pair_acc / monotone against true steps-to-goal, held-out rows."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((st, ep))
    ep_s, st_s = ep[order], st[order]
    starts, lens = {}, {}
    for i, e in enumerate(ep_s):
        if e not in starts:
            starts[e] = i
        lens[e] = lens.get(e, 0) + 1
    eps = np.array(sorted(starts))

    s_idx, g_idx, delta = [], [], []
    while len(s_idx) < n_pairs:
        e = eps[rng.integers(len(eps))]
        L = lens[e]
        if L < 3:
            continue
        a = rng.integers(0, L - 1)
        d = rng.integers(1, min(max_delta, L - 1 - a) + 1)
        s_idx.append(order[starts[e] + a]); g_idx.append(order[starts[e] + a + d])
        delta.append(d)
    s_idx, g_idx = np.asarray(s_idx), np.asarray(g_idx)
    delta = np.asarray(delta, dtype=np.float64)

    with torch.no_grad():
        out = []
        for i in range(0, len(s_idx), 2048):
            zs = z[s_idx[i:i + 2048]].to(dev).float()
            zg = z[g_idx[i:i + 2048]].to(dev).float()
            out.append(metric.cost(zs, zg).float().cpu().numpy().reshape(-1))
    d_pred = np.concatenate(out)
    ok = np.isfinite(d_pred)
    d_pred, dl = d_pred[ok], delta[ok]

    def spear(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / den) if den > 0 else float('nan')

    rho = spear(d_pred, dl)
    h = len(d_pred) // 2
    i1, i2 = np.arange(h), np.arange(h, 2 * h)
    diff = dl[i1] - dl[i2]
    m = diff != 0
    pair_acc = float(((d_pred[i1][m] < d_pred[i2][m]) == (diff[m] < 0)).mean())

    mono = []
    for _ in range(200):
        e = eps[rng.integers(len(eps))]
        L = lens[e]
        if L < 12:
            continue
        g = order[starts[e] + L - 1]
        rows = order[starts[e]:starts[e] + min(L - 1, 60)]
        with torch.no_grad():
            zz = z[rows].to(dev).float()
            zg = z[g].to(dev).float().unsqueeze(0).expand(len(rows), -1)
            dd = metric.cost(zz, zg).float().cpu().numpy().reshape(-1)
        if np.isfinite(dd).all() and len(dd) > 1:
            mono.append(float((np.diff(dd) < 0).mean()))
    return rho, pair_acc, (float(np.mean(mono)) if mono else float('nan'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', default='/workspace/caches/dinofull_tr8000.pt')
    p.add_argument('--ho-cache', default='/workspace/caches/dinofull_ho.pt')
    p.add_argument('--out-dir', default='/workspace/metrics')
    p.add_argument('--csv', default='/workspace/results/summary_dinofull_td.csv')
    p.add_argument('--tag', default='dinofull')
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--n-step', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--max-cells', type=int, default=len(GRID))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    dev = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cache = LatentCache.load(args.train_cache)
    print(f'[sweep] train cache {len(cache.z)} x {cache.latent_dim} loaded in '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    t0 = time.time()
    ho = LatentCache.load(args.ho_cache)
    ep_ho = ho.episode_idx.numpy().reshape(-1)
    st_ho = ho.step_idx.numpy().reshape(-1)
    print(f'[sweep] held-out cache {len(ho.z)} x {ho.latent_dim} loaded in '
          f'{(time.time() - t0) / 60:.1f} min (eps '
          f'{ep_ho.min()}-{ep_ho.max()})', flush=True)

    done = set()
    if Path(args.csv).exists():
        for line in Path(args.csv).read_text().splitlines():
            if line.strip():
                done.add(line.split(',')[0])

    for i, (exp, lr) in enumerate(GRID[:args.max_cells]):
        name = f'{args.tag}_e{exp}_lr{lr}'
        if name in done:
            print(f'[sweep] {name} cached, skip', flush=True)
            continue
        cfg = TDConfig(head='quasimetric', n_step=args.n_step, gamma=1.0,
                       expectile=exp, lr=lr, batch_size=args.batch_size,
                       steps=args.steps, seed=0)
        t0 = time.time()
        module = learners.td.fit(cache, cfg, dev)
        train_min = (time.time() - t0) / 60
        print(f'[sweep] {name} trained in {train_min:.1f} min', flush=True)

        # Persist FIRST and never let a downstream error discard a trained cell:
        # each cell costs ~20 min plus a 29 min cache load to redo.
        # NB num_components is a train_metric.py CLI arg, NOT a TDConfig field.
        out = Path(args.out_dir) / f'{name}.pt'
        try:
            save_metric(module.cpu(), 'td', cache.latent_dim,
                        {'head': 'quasimetric', 'hidden_dim': cfg.hidden_dim,
                         'depth': cfg.depth, 'embed_dim': cfg.embed_dim,
                         'softplus': True, 'symmetric': False,
                         'num_components': 8}, str(out))
        except Exception as e:
            torch.save(module.state_dict(), str(out) + '.raw')
            print(f'[sweep] {name} save_metric FAILED ({e}); raw state_dict kept',
                  flush=True)

        rho = pa = mo = float('nan')
        try:
            module = module.to(dev).eval()
            t0 = time.time()
            rho, pa, mo = probe(module, ho.z, ep_ho, st_ho, dev)
            probe_min = (time.time() - t0) / 60
        except Exception as e:
            probe_min = float('nan')
            print(f'[sweep] {name} probe FAILED: {e}', flush=True)
        with open(args.csv, 'a') as f:
            f.write(f'{name},{rho:.4f},{pa:.4f},{mo:.4f},{train_min:.1f}\n')
        print(f'[sweep] {i+1}/{args.max_cells} {name}: spearman={rho:.4f} '
              f'pair_acc={pa:.4f} monotone={mo:.4f} '
              f'(train {train_min:.1f} min, probe {probe_min:.1f} min)', flush=True)
        del module
        torch.cuda.empty_cache()

    print('[sweep] DONE', flush=True)


if __name__ == '__main__':
    main()
