"""Second TD sweep: is the recipe itself wrong for a 75,264-d DINO substrate?

The first sweep tuned expectile x lr and topped out at monotonicity 0.678 vs the
LeWM teacher's 0.923 -- and TD+CEM then scored BELOW plain CEM (78.0 vs 82.0 on
draw 42). So the deficit is in HOW the metric is parameterised/trained, not in
the expectile.

Motivation: every default here was tuned for LeWM's 192-d latents. At 192-d,
`hidden_dim 256` is an EXPANSION; at 75,264-d it is a 294:1 bottleneck in a
single layer, and the quasimetric embedding is 128-d. Two loss terms also target
ordering directly, which is exactly what monotonicity measures:
  * rank_weight   pairwise ranking hinge -- explicit ordering supervision
  * eikonal_weight unit-gradient penalty -- smoother, more regular value field

One factor at a time from the sweep winner (expectile 0.1, lr 3e-4), ordered so
an early stop still covers the most promising levers. PRIMARY target is
monotone; pair_acc/spearman are already at LeWM parity.
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

BASE = dict(head='quasimetric', n_step=50, gamma=1.0, expectile=0.1, lr=3e-4,
            batch_size=1024, steps=6000, seed=0)

# (name, overrides) -- most promising first.
#
# NOTE: `rank_weight` and `eikonal_weight` are NOT TDConfig fields. They are
# train_metric.py CLI args belonging to other learners, so this TD learner has
# NO ranking-hinge and NO eikonal term -- the two losses that would supervise
# ordering and gradient regularity directly, i.e. exactly the monotonicity
# deficit. Adding them means editing learners/td.py; out of scope for a sweep.
#
# What is left is capacity/conditioning and the TD target itself. The strongest
# prior is capacity: 75,264 -> hidden_dim 256 is a 294:1 bottleneck in one
# layer, where for LeWM's 192-d latents the same default was an EXPANSION.
CELLS = [
    ('hidden1024', dict(hidden_dim=1024)),
    ('hidden2048', dict(hidden_dim=2048)),
    ('embed512',   dict(embed_dim=512)),
    ('wide_deep',  dict(hidden_dim=1024, depth=3, embed_dim=256)),
    ('depth3',     dict(depth=3)),
    ('tau0.05',    dict(tau=0.05)),
    ('nstep25',    dict(n_step=25)),
    ('nstep100',   dict(n_step=100)),
    ('headmlp',    dict(head='mlp')),
    ('pcross0.1',  dict(p_cross=0.1)),
    ('wd0',        dict(weight_decay=0.0)),
    ('huber0.1',   dict(huber_beta=0.1)),
]


def _preflight():
    """Validate every override key against the dataclass BEFORE loading 605 GB.

    An invalid key previously surfaced only after a 12-minute cache load and
    killed the run (rank_weight, num_components). Fail in one second instead.
    """
    import dataclasses
    valid = {f.name for f in dataclasses.fields(TDConfig)}
    bad = {}
    for nm, over in CELLS:
        unknown = set(over) - valid
        if unknown:
            bad[nm] = sorted(unknown)
    unknown_base = set(BASE) - valid
    if unknown_base:
        bad['BASE'] = sorted(unknown_base)
    if bad:
        raise SystemExit(
            'invalid TDConfig keys (not dataclass fields): '
            + '; '.join(f'{k}: {v}' for k, v in bad.items())
            + f'\nvalid fields: {sorted(valid)}'
        )
    print(f'[arch] preflight OK: {len(CELLS)} cells, all keys valid', flush=True)


def probe(metric, z, ep, st, dev, n_pairs=20000, max_delta=100, seed=0):
    rng = np.random.default_rng(seed)
    order = np.lexsort((st, ep))
    ep_s = ep[order]
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
    p.add_argument('--csv', default='/workspace/results/summary_dinoarch_td.csv')
    p.add_argument('--tag', default='dinoarch')
    p.add_argument('--max-cells', type=int, default=len(CELLS))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    dev = args.device
    _preflight()          # before the 12-minute cache load, not after
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cache = LatentCache.load(args.train_cache)
    print(f'[arch] train cache {len(cache.z)} x {cache.latent_dim} in '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    t0 = time.time()
    ho = LatentCache.load(args.ho_cache)
    ep_ho = ho.episode_idx.numpy().reshape(-1)
    st_ho = ho.step_idx.numpy().reshape(-1)
    print(f'[arch] held-out cache {len(ho.z)} x {ho.latent_dim} in '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)

    done = set()
    if Path(args.csv).exists():
        for line in Path(args.csv).read_text().splitlines():
            if line.strip():
                done.add(line.split(',')[0])

    for i, (nm, over) in enumerate(CELLS[:args.max_cells]):
        name = f'{args.tag}_{nm}'
        if name in done:
            print(f'[arch] {name} cached, skip', flush=True)
            continue
        kw = dict(BASE); kw.update(over)
        cfg = TDConfig(**kw)
        t0 = time.time()
        try:
            module = learners.td.fit(cache, cfg, dev)
        except Exception as e:
            print(f'[arch] {name} FIT FAILED: {type(e).__name__}: {str(e)[:200]}',
                  flush=True)
            with open(args.csv, 'a') as f:
                f.write(f'{name},FAIL,FAIL,FAIL,0,{over}\n')
            torch.cuda.empty_cache()
            continue
        train_min = (time.time() - t0) / 60
        out = Path(args.out_dir) / f'{name}.pt'
        try:
            save_metric(module.cpu(), 'td', cache.latent_dim,
                        {'head': cfg.head, 'hidden_dim': cfg.hidden_dim,
                         'depth': cfg.depth, 'embed_dim': cfg.embed_dim,
                         'softplus': True, 'symmetric': False,
                         'num_components': 8}, str(out))
        except Exception as e:
            torch.save(module.state_dict(), str(out) + '.raw')
            print(f'[arch] {name} save failed ({e}); raw kept', flush=True)
        rho = pa = mo = float('nan')
        try:
            module = module.to(dev).eval()
            rho, pa, mo = probe(module, ho.z, ep_ho, st_ho, dev)
        except Exception as e:
            print(f'[arch] {name} probe failed: {e}', flush=True)
        with open(args.csv, 'a') as f:
            f.write(f'{name},{rho:.4f},{pa:.4f},{mo:.4f},{train_min:.1f},{over}\n')
        print(f'[arch] {i+1}/{args.max_cells} {name}: spearman={rho:.4f} '
              f'pair_acc={pa:.4f} monotone={mo:.4f} ({train_min:.1f} min) {over}',
              flush=True)
        del module
        torch.cuda.empty_cache()

    print('[arch] DONE', flush=True)


if __name__ == '__main__':
    main()
