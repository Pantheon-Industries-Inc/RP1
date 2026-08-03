"""Derive pooled 384-d caches from the full-token 75,264-d caches.

`train_lip_dino.py` and `rollout_terminal_dino` both operate on the MEAN-POOLED
pixel embedding (384-d): the rollout returns `toks[..., :384].mean(dim=1)`, and
PlannerNet is built as `PlannerNet(z.shape[-1])`. So the LIP path needs pooled
caches, whatever the TD sweeps used.

Pooling is exact and cheap: reshape each row to (196, 384) and average over the
patch axis -- the same reduction the deployed rollout applies, so cache and
deploy stay in one space.

Uses torch.load(mmap=True) so the 484 GB file is memory-MAPPED, not read into
RAM: a concurrent sweep already holds ~605 GB, and loading another 484 GB is
what OOM-killed the arch sweep earlier tonight. Rows are pooled in chunks.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache  # noqa: E402


def pool_one(src, dst, patches=196, pix=384, chunk=20000):
    t0 = time.time()
    blob = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
    z = blob['z'] if isinstance(blob, dict) else blob.z
    n, d = z.shape
    assert d == patches * pix + 0 or d % patches == 0, f'unexpected dim {d}'
    per = d // patches
    out = torch.empty(n, pix, dtype=torch.float32)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        blk = z[i:j].reshape(j - i, patches, per)[:, :, :pix]
        out[i:j] = blk.mean(dim=1)
    get = (lambda k: blob[k]) if isinstance(blob, dict) else (lambda k: getattr(blob, k))
    cache = LatentCache(
        z=out,
        episode_idx=get('episode_idx'),
        step_idx=get('step_idx'),
        state=get('state'),
        meta={'derived_from': str(src), 'reduction': 'mean over 196 patches',
              'pix_dim': pix,
              'note': 'matches rollout_terminal_dino, which returns '
                      'toks[..., :384].mean(dim=1)'},
    )
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cache.save(dst)
    gb = Path(dst).stat().st_size / 1e9
    print(f'[pool] {Path(src).name} {n}x{d} -> {Path(dst).name} {n}x{pix} '
          f'({gb:.2f} GB, {(time.time()-t0)/60:.1f} min)', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', nargs='+', required=True,
                   help='src:dst pairs')
    args = p.parse_args()
    for pr in args.pairs:
        src, dst = pr.split(':')
        pool_one(src, dst)
    print('[pool] DONE', flush=True)


if __name__ == '__main__':
    main()
