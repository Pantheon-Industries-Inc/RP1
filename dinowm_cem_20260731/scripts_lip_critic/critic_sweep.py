"""Critic sweep: standardization x sample count, on the pooled DINO latents.

WHY. LeWM and PLDM both end their encoder in MLP(192->2048->192,
norm_fn=BatchNorm1d), so their TD critic consumes a per-dim zero-mean/unit-var,
isotropic space BY CONSTRUCTION. Our DINO latents are raw mean-pooled DINOv2
activations with no normalization anywhere. The quasimetric head's Euclidean
term is then dominated by a few high-variance directions, and SGD must undo that
scaling before it can encode task progress -- an optimization burden LeWM/PLDM
never pay. Every measurement points at optimization: no train/held-out gap,
capacity irrelevant (7-point null), supervised regression no better than TD, and
sample count the only lever that has moved anything (6.1M -> 24.6M).

DESIGN. 2x3 factorial so standardization and data volume are separable:
    standardize in {off, on} x samples in {24.6M, 49.2M, 98.3M}
The `off` row re-measures the current recipe at higher volume, so a win for
`on` cannot be confused with simply training longer.

FOLDING. Train on standardized latents, then fold the affine transform into
enc[0] (the single entry Linear both z_i and z_j pass through):
    W' = W / sd            b' = b - W @ (mu / sd)
The saved metric therefore accepts RAW latents and every call site -- the CEM
metric hook, LIP, the probes -- works unchanged. Verified numerically per cell
before saving (folded-on-raw vs unfolded-on-standardized must agree).
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

import importlib.util as _iu
_spec = _iu.spec_from_file_location('arch', '/workspace/td_sweep_arch.py')
_arch = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_arch)
probe = _arch.probe


def fold_standardization(module, mu, sd):
    """Rewrite enc[0] so the head accepts RAW latents after training on standardized.

    first layer trained as  W @ (x-mu)/sd + b
                          = (W/sd) @ x + (b - W @ mu/sd)
    """
    enc = module.enc if hasattr(module, 'enc') else None
    if enc is None:
        raise RuntimeError('head has no .enc; cannot fold')
    lin = enc[0]
    if not isinstance(lin, torch.nn.Linear):
        raise RuntimeError(f'enc[0] is {type(lin).__name__}, expected Linear')
    with torch.no_grad():
        W = lin.weight.data                       # (hidden, dim)
        b = lin.bias.data                         # (hidden,)
        sd_ = sd.to(W.device).clamp_min(1e-8)
        mu_ = mu.to(W.device)
        lin.bias.data = b - (W / sd_) @ mu_
        lin.weight.data = W / sd_
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', default='/workspace/caches/dinopool_tr8000_fs1.pt')
    p.add_argument('--ho-cache', default='/workspace/caches/dinopool_ho.pt')
    p.add_argument('--out-dir', default='/workspace/metrics')
    p.add_argument('--csv', default='/workspace/results/summary_critic_sweep.csv')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = a.device
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)

    raw = LatentCache.load(a.train_cache)
    ho = LatentCache.load(a.ho_cache)
    ep_ho = ho.episode_idx.numpy().reshape(-1)
    st_ho = ho.step_idx.numpy().reshape(-1)
    print(f'[cs] train {len(raw.z)}x{raw.latent_dim} | held-out {len(ho.z)}', flush=True)

    mu = raw.z.float().mean(0)
    sd = raw.z.float().std(0).clamp_min(1e-8)
    print(f'[cs] per-dim std: min {sd.min():.4g} median {sd.median():.4g} '
          f'max {sd.max():.4g}  (max/min {sd.max()/sd.min():.1f}x)', flush=True)

    z_std = ((raw.z.float() - mu) / sd)
    std_cache = LatentCache(z=z_std, episode_idx=raw.episode_idx,
                            step_idx=raw.step_idx, state=raw.state,
                            meta={'derived': 'per-dim standardized'})

    done = set()
    if Path(a.csv).exists():
        for line in Path(a.csv).read_text().splitlines():
            if line.strip():
                done.add(line.split(',')[0])

    CELLS = [(s, steps) for s in (True, False) for steps in (24000, 48000, 96000)]
    for standardize, steps in CELLS:
        tag = f"{'std' if standardize else 'raw'}{steps // 1000}k"
        if tag in done:
            print(f'[cs] {tag} cached, skip', flush=True)
            continue
        cache = std_cache if standardize else raw
        cfg = TDConfig(head='quasimetric', n_step=50, gamma=1.0, expectile=0.1,
                       lr=3e-4, batch_size=a.batch_size, steps=steps, seed=0)
        t0 = time.time()
        m = learners.td.fit(cache, cfg, dev)
        train_min = (time.time() - t0) / 60

        if standardize:
            # numerical check: folded-on-raw must equal unfolded-on-standardized
            m = m.to(dev).eval()
            idx = torch.randperm(len(ho.z))[:512]
            zr = ho.z[idx].float().to(dev)
            zg = ho.z[torch.randperm(len(ho.z))[:512]].float().to(dev)
            with torch.no_grad():
                ref = m.cost((zr - mu.to(dev)) / sd.to(dev),
                             (zg - mu.to(dev)) / sd.to(dev)).cpu()
            m = fold_standardization(m, mu, sd)
            with torch.no_grad():
                got = m.cost(zr, zg).cpu()
            err = (ref - got).abs().max().item()
            print(f'[cs] {tag} fold check: max|diff| {err:.3e} '
                  f'({"OK" if err < 1e-3 else "MISMATCH"})', flush=True)
            if err >= 1e-3:
                print(f'[cs] {tag} FOLD FAILED, not saving', flush=True)
                continue

        out = Path(a.out_dir) / f'dinopool_td_{tag}.pt'
        save_metric(m.cpu(), 'td', raw.latent_dim,
                    {'head': 'quasimetric', 'hidden_dim': cfg.hidden_dim,
                     'depth': cfg.depth, 'embed_dim': cfg.embed_dim,
                     'softplus': True, 'symmetric': False, 'num_components': 8},
                    str(out))
        m = m.to(dev).eval()
        rho, pa, mo = probe(m, ho.z, ep_ho, st_ho, dev)   # RAW held-out latents
        with open(a.csv, 'a') as f:
            f.write(f'{tag},{rho:.4f},{pa:.4f},{mo:.4f},{train_min:.1f},'
                    f'{a.batch_size * steps / 1e6:.1f}M\n')
        print(f'[cs] {tag}: spearman={rho:.4f} pair_acc={pa:.4f} monotone={mo:.4f} '
              f'({train_min:.1f} min, {a.batch_size*steps/1e6:.1f}M samples)', flush=True)
        del m
        torch.cuda.empty_cache()

    print('\n=== critic sweep done (refs: LeWM 0.484/0.691/0.923, '
          'pre-fix pooled 0.4743/0.6870/0.6236) ===', flush=True)
    if Path(a.csv).exists():
        print(Path(a.csv).read_text(), flush=True)


if __name__ == '__main__':
    main()
