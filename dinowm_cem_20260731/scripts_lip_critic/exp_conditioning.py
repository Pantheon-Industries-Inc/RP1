"""Why does the same TD critic work on LeWM/PLDM latents but not DINO ones?

Structural fact, from the checkpoint configs: BOTH LeWM (v2WM) and PLDM end their
encoders with an MLP projector whose norm_fn is **BatchNorm1d**:

    projector: MLP(192 -> 2048 -> 192, norm_fn=BatchNorm1d)
    pred_proj: MLP(192 -> 2048 -> 192, norm_fn=BatchNorm1d)

So the critic there consumes a space that is, by construction, per-dimension
zero-mean / unit-variance and isotropic. Our DINO latents are raw mean-pooled
DINOv2 patch activations -- no normalization anywhere in the path.

A quasimetric head computes d = ||u(x) - u(y)|| + max_k relu(v(y)_k - v(x)_k)
with u, v as 2-layer MLPs. On badly-conditioned inputs the Euclidean term is
dominated by a few high-variance directions, and gradient descent has to undo
that scaling before it can learn anything about task progress. That is an
OPTIMIZATION problem, which is exactly the diagnosis every measurement supports:
no train/held-out gap (not overfitting), head capacity irrelevant (7-point null),
and more DATA the only lever that helped (6.1M -> 24.6M samples).

This quantifies the conditioning gap so the fix can be argued from numbers:
  * per-dim std spread (max/min, and the ratio at the 99th/1st percentile)
  * covariance condition number and effective rank (participation ratio)
  * how much total variance the top few directions carry
A BatchNorm'd space would score std-ratio ~1, effective rank ~= dim.
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache  # noqa: E402


def report(name, z):
    z = z.float()
    n, d = z.shape
    mu = z.mean(0)
    sd = z.std(0)
    sd_pos = sd[sd > 0]
    zc = z - mu
    # covariance spectrum on a subsample (d x d is fine for 384; sample rows)
    idx = torch.randperm(n)[:min(n, 60000)]
    x = zc[idx]
    cov = (x.T @ x) / (len(idx) - 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    ev = ev.flip(0)
    tot = ev.sum()
    part_ratio = float((ev.sum() ** 2) / (ev ** 2).sum())     # effective rank
    cond = float(ev[0] / ev[ev > 0][-1]) if (ev > 0).any() else float('inf')
    top1 = float(ev[0] / tot)
    top10 = float(ev[:10].sum() / tot)
    print(f'\n--- {name}: {n} x {d} ---')
    print(f'  per-dim std   min {sd_pos.min():.4g}  median {sd_pos.median():.4g}  '
          f'max {sd_pos.max():.4g}   max/min {sd_pos.max()/sd_pos.min():.1f}x')
    print(f'  per-dim mean  min {mu.min():.4g}  median {mu.median():.4g}  max {mu.max():.4g}')
    print(f'  cov cond number      {cond:.3g}')
    print(f'  effective rank       {part_ratio:.1f} / {d}   ({100*part_ratio/d:.1f}% of dims)')
    print(f'  variance in top-1    {100*top1:.1f}%     top-10 {100*top10:.1f}%')
    return dict(std_ratio=float(sd_pos.max() / sd_pos.min()), cond=cond,
                eff_rank=part_ratio, dim=d, top10=top10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pooled', default='/workspace/caches/dinopool_tr8000_fs1.pt')
    p.add_argument('--full', default='/workspace/caches/dinofull_tr8000.pt')
    p.add_argument('--n-full', type=int, default=40000,
                   help='rows to sample from the 75k-d cache (mmap-read)')
    a = p.parse_args()

    c = LatentCache.load(a.pooled)
    r_pool = report('DINO pooled 384-d (what the LIP critic sees)', c.z)

    # full-token: mmap and subsample rows, then look at a random 384-dim slice
    blob = torch.load(a.full, map_location='cpu', weights_only=False, mmap=True)
    z = blob['z'] if isinstance(blob, dict) else blob.z
    idx = torch.randperm(z.shape[0])[:a.n_full].sort().values
    sub = z[idx].float()
    r_full = report('DINO full-token 75,264-d (subsampled rows)', sub)

    print('\n=== reference: LeWM / PLDM ===')
    print('  Both end in MLP(192->2048->192, norm_fn=BatchNorm1d), so by')
    print('  construction: per-dim std ratio ~1.0, means ~0, effective rank ~192/192.')
    print('\n=== implication ===')
    if r_pool['std_ratio'] > 5 or r_pool['eff_rank'] / r_pool['dim'] < 0.5:
        print('  DINO latents are strongly anisotropic where LeWM/PLDM are isotropic')
        print('  by construction. The quasimetric Euclidean term is dominated by a')
        print('  few directions, so the head must first undo the scaling -- an')
        print('  optimization burden LeWM/PLDM never pay. Fix: standardize the')
        print('  latents per-dim, then FOLD the affine transform into the head\'s')
        print('  first Linear (W/sd, b - W@mu/sd) so the saved metric still accepts')
        print('  RAW latents and every call site -- CEM hook, LIP, probes -- works')
        print('  unchanged.')
    else:
        print('  Conditioning looks comparable; the gap is elsewhere.')


if __name__ == '__main__':
    main()
