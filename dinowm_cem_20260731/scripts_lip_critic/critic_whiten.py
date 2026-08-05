"""PCA-whitened critic: does removing the 97% null directions replace 16x data?

MEASURED conditioning of the pooled DINO latents (1.608M x 384):
    per-dim std spread   5.7x          <- mild; standardization only fixes this
    cov condition number 3.35e4
    effective rank       13.2 / 384    <- the actual problem: 3.4% of dims
    top-10 directions    64.8% of variance

So the critic has been reading 384 dims of which ~371 are near-null. Two
consequences: it must average that noise out (hence 98.3M samples helping), and
per-dim standardization AMPLIFIES the null directions by dividing by tiny stds --
it was fighting itself even while netting +0.03.

Whitening to the effective rank instead: project onto the top-k principal
components and scale each to unit variance. Covariance becomes identity
(condition number 1), the null directions are DROPPED rather than amplified, and
the input shrinks 384 -> k so the head has far less to fit.

Test is deliberately at LOW sample counts. If whitening is the real fix, k=24 at
6.1M (the STOCK budget) should rival std96k at 98.3M -- i.e. 16x less compute.

FOLDING: whitening is affine, z -> diag(1/sqrt(ev)) @ P^T @ (z - mu), so it folds
into enc[0] exactly as standardization did:
    W' = W @ (diag(1/sqrt(ev)) @ P^T)     b' = b - W' @ mu
The head then accepts RAW 384-d latents; every call site works unchanged.
Verified numerically per cell.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

from stable_worldmodel.trm import LatentCache, learners, save_metric  # noqa: E402
from stable_worldmodel.trm.learners.td import TDConfig  # noqa: E402

import importlib.util as _iu
_spec = _iu.spec_from_file_location('arch', '/workspace/td_sweep_arch.py')
_arch = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_arch)
probe = _arch.probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-cache', default='/workspace/caches/dinopool_tr8000_fs1.pt')
    p.add_argument('--ho-cache', default='/workspace/caches/dinopool_ho.pt')
    p.add_argument('--out-dir', default='/workspace/metrics')
    p.add_argument('--csv', default='/workspace/results/summary_critic_whiten.csv')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = a.device
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    raw = LatentCache.load(a.train_cache)
    ho = LatentCache.load(a.ho_cache)
    ep_ho = ho.episode_idx.numpy().reshape(-1)
    st_ho = ho.step_idx.numpy().reshape(-1)

    # PCA on TRAIN rows only
    z = raw.z.float()
    mu = z.mean(0)
    zc = z - mu
    idx = torch.randperm(len(zc))[:200000]
    cov = (zc[idx].T @ zc[idx]) / (len(idx) - 1)
    ev, P = torch.linalg.eigh(cov.double())
    ev, P = ev.flip(0).clamp_min(1e-10), P.flip(1)          # descending
    evr = (ev / ev.sum()).cpu()
    print(f'[wh] variance retained: k=8 {evr[:8].sum():.3f} k=16 {evr[:16].sum():.3f} '
          f'k=24 {evr[:24].sum():.3f} k=32 {evr[:32].sum():.3f} k=64 {evr[:64].sum():.3f}',
          flush=True)

    done = set()
    if Path(a.csv).exists():
        for line in Path(a.csv).read_text().splitlines():
            if line.strip():
                done.add(line.split(',')[0])

    # (k, steps): stock budget 6000 first -- the whole point is doing MORE with LESS
    CELLS = [(24, 6000), (24, 24000), (16, 24000), (32, 24000), (64, 24000)]
    for k, steps in CELLS:
        tag = f'wh{k}_{steps // 1000}k'
        if tag in done:
            print(f'[wh] {tag} cached, skip', flush=True)
            continue
        Wt = (P[:, :k] / ev[:k].sqrt()).float()             # (384, k) whitening
        zw = (zc @ Wt)
        cache = LatentCache(z=zw, episode_idx=raw.episode_idx, step_idx=raw.step_idx,
                            state=raw.state, meta={'derived': f'PCA-whitened k={k}'})
        cfg = TDConfig(head='quasimetric', n_step=50, gamma=1.0, expectile=0.1,
                       lr=3e-4, batch_size=a.batch_size, steps=steps, seed=0)
        t0 = time.time()
        m = learners.td.fit(cache, cfg, dev).to(dev).eval()
        train_min = (time.time() - t0) / 60

        # reference on whitened held-out, then fold to accept RAW 384-d
        hz = ho.z.float()
        sel = torch.randperm(len(hz))[:512]
        sel2 = torch.randperm(len(hz))[:512]
        with torch.no_grad():
            ref = m.cost(((hz[sel] - mu) @ Wt).to(dev),
                         ((hz[sel2] - mu) @ Wt).to(dev)).cpu()
        lin = m.enc[0]
        with torch.no_grad():
            W, b = lin.weight.data.clone(), lin.bias.data.clone()   # (hid,k),(hid,)
            Wnew = W @ Wt.T.to(W.device)                            # (hid,384)
            lin.weight = torch.nn.Parameter(Wnew)
            lin.bias = torch.nn.Parameter(b - Wnew @ mu.to(W.device))
            lin.in_features = 384
        m.latent_dim = 384
        with torch.no_grad():
            got = m.cost(hz[sel].to(dev), hz[sel2].to(dev)).cpu()
        err = (ref - got).abs().max().item()
        print(f'[wh] {tag} fold check max|diff| {err:.3e} '
              f'({"OK" if err < 1e-3 else "MISMATCH"})', flush=True)
        if err >= 1e-3:
            print(f'[wh] {tag} FOLD FAILED, skipping', flush=True)
            continue

        out = Path(a.out_dir) / f'dinopool_td_{tag}.pt'
        save_metric(m.cpu(), 'td', 384,
                    {'head': 'quasimetric', 'hidden_dim': cfg.hidden_dim,
                     'depth': cfg.depth, 'embed_dim': cfg.embed_dim,
                     'softplus': True, 'symmetric': False, 'num_components': 8},
                    str(out))
        m = m.to(dev).eval()
        rho, pa, mo = probe(m, ho.z, ep_ho, st_ho, dev)     # RAW held-out latents
        with open(a.csv, 'a') as f:
            f.write(f'{tag},{rho:.4f},{pa:.4f},{mo:.4f},{train_min:.1f},'
                    f'{a.batch_size*steps/1e6:.1f}M,k={k}\n')
        print(f'[wh] {tag}: spearman={rho:.4f} pair_acc={pa:.4f} monotone={mo:.4f} '
              f'({train_min:.1f} min, {a.batch_size*steps/1e6:.1f}M samples)', flush=True)
        del m, cache, zw
        torch.cuda.empty_cache()

    print('\n=== refs: std96k 0.5532/0.7252/0.6534 @98.3M | raw24k 0.4743/0.6870/0.6236 '
          '| LeWM 0.484/0.691/0.923 ===', flush=True)
    if Path(a.csv).exists():
        print(Path(a.csv).read_text(), flush=True)


if __name__ == '__main__':
    main()
