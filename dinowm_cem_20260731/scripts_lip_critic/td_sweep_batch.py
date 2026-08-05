"""Batch-size sweep for the DINO TD teacher -- a direct test of underfitting.

Why this axis. The diagnostics say the head is optimization-limited, not
capacity- or generalization-limited: train scores equal held-out scores, and 8
architecture variants (head width 256->2048, embed 128->512, depth 2->3, Polyak,
n-step) all land within +-0.01 on every probe metric. Batch size is an
OPTIMIZATION knob, so it is the right place to look.

It is also a data-volume knob here, which is what makes it a real test: `steps`
is fixed at 6000, so batch size sets how many samples the head ever sees --
1.5M at bs256 up to 24.6M at bs4096. If the head is underfitting, more samples
should help monotonically.

The `bs1024_steps24k` cell disentangles the two readings: it sees the SAME
24.6M samples as bs4096 but in small batches. So
    bs4096 > bs1024_steps24k  -> batch SIZE (gradient variance) is what matters
    bs4096 ~= bs1024_steps24k -> total DATA is what matters
    both ~= baseline          -> neither; the objective itself is the ceiling

Cheap cells first so a signal arrives in minutes rather than hours. Cost scales
with batch: each step moves 3 x batch x 75,264 x 4 B, so bs4096 is ~4x slower
per step than the bs1024 baseline (~24 min).
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, learners, save_metric  # noqa: E402
from stable_worldmodel.trm.learners.td import TDConfig  # noqa: E402

# reuse the probe from the arch sweep rather than duplicating it
_spec = importlib.util.spec_from_file_location('arch', '/workspace/td_sweep_arch.py')
_arch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_arch)
probe = _arch.probe

BASE = dict(head='quasimetric', n_step=50, gamma=1.0, expectile=0.1, lr=3e-4,
            batch_size=1024, steps=6000, seed=0)

# (name, overrides, approx samples seen) -- cheap first
CELLS = [
    ('bs256',            dict(batch_size=256)),
    ('bs512',            dict(batch_size=512)),
    ('bs2048',           dict(batch_size=2048)),
    ('bs4096',           dict(batch_size=4096)),
    ('bs1024_steps24k',  dict(batch_size=1024, steps=24000)),
    ('bs4096_lr1.2e-3',  dict(batch_size=4096, lr=1.2e-3)),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', default='/workspace/caches/dinofull_tr8000.pt')
    p.add_argument('--ho-cache', default='/workspace/caches/dinofull_ho.pt')
    p.add_argument('--out-dir', default='/workspace/metrics')
    p.add_argument('--csv', default='/workspace/results/summary_dinobatch_td.csv')
    p.add_argument('--tag', default='dinobatch')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    import dataclasses
    valid = {f.name for f in dataclasses.fields(TDConfig)}
    for nm, over in CELLS:
        bad = set(over) - valid
        if bad:
            raise SystemExit(f'{nm}: invalid TDConfig keys {sorted(bad)}')
    print(f'[batch] preflight OK: {len(CELLS)} cells', flush=True)

    dev = args.device
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cache = LatentCache.load(args.train_cache)
    print(f'[batch] train cache {len(cache.z)} x {cache.latent_dim} in '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    t0 = time.time()
    ho = LatentCache.load(args.ho_cache)
    ep_ho = ho.episode_idx.numpy().reshape(-1)
    st_ho = ho.step_idx.numpy().reshape(-1)
    print(f'[batch] held-out cache in {(time.time() - t0) / 60:.1f} min', flush=True)

    done = set()
    if Path(args.csv).exists():
        for line in Path(args.csv).read_text().splitlines():
            if line.strip():
                done.add(line.split(',')[0])

    for i, (nm, over) in enumerate(CELLS):
        name = f'{args.tag}_{nm}'
        if name in done:
            print(f'[batch] {name} cached, skip', flush=True)
            continue
        kw = dict(BASE); kw.update(over)
        samples = kw['batch_size'] * kw['steps']
        cfg = TDConfig(**kw)
        t0 = time.time()
        try:
            module = learners.td.fit(cache, cfg, dev)
        except Exception as e:
            print(f'[batch] {name} FIT FAILED: {type(e).__name__}: {str(e)[:200]}',
                  flush=True)
            with open(args.csv, 'a') as f:
                f.write(f'{name},FAIL,FAIL,FAIL,0,{samples},{over}\n')
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
            print(f'[batch] {name} save failed ({e}); raw kept', flush=True)
        rho = pa = mo = float('nan')
        try:
            module = module.to(dev).eval()
            rho, pa, mo = probe(module, ho.z, ep_ho, st_ho, dev)
        except Exception as e:
            print(f'[batch] {name} probe failed: {e}', flush=True)
        with open(args.csv, 'a') as f:
            f.write(f'{name},{rho:.4f},{pa:.4f},{mo:.4f},{train_min:.1f},'
                    f'{samples},{over}\n')
        print(f'[batch] {i+1}/{len(CELLS)} {name}: spearman={rho:.4f} '
              f'pair_acc={pa:.4f} monotone={mo:.4f} '
              f'({train_min:.1f} min, {samples/1e6:.1f}M samples) {over}', flush=True)
        del module
        torch.cuda.empty_cache()

    print('[batch] DONE', flush=True)


if __name__ == '__main__':
    main()
