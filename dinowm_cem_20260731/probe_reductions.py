"""Measure how much task-relevant information each latent reduction keeps.

The TD/LIP metric needs a FLAT per-frame latent, but DINO-WM emits a 14x14 grid
of 196 patch tokens x 384 channels. Mean-pooling collapses the spatial axis and
measured badly (trajectory monotonicity 0.64 vs 0.92 for the LeWM teacher), so
this compares candidate reductions on what cube-single actually depends on:
can true cube position be linearly decoded from the reduced latent?

Candidates (all applicable identically at cache time and plan time):
  full        flatten the whole grid, 196*384 = 75264       (no reduction)
  projK       per-token channel PCA 384->K, keep all 196 tokens, flatten
              (no mixing across space -- spatial layout fully preserved)
  grid4x4     keep all 384 channels at 16 strided token positions (selection,
              no averaging)
  meanpool    mean over the 196 tokens -> 384               (current baseline)

Protocol: exact ridge regression in dual form (n < d), features standardised on
train, lambda swept, best test R^2 reported. Train frames come from episodes
0-7999, test frames from 8000-9999, so the probe is scored out of sample.
Targets: cube xyz (privileged_block_0_pos) and end-effector xyz
(proprio_effector_pos) -- the planner needs both.

R^2 is reported per target; 1.0 = perfect linear decode, 0.0 = no better than
predicting the mean.
"""

import argparse
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def encode_tokens(wm, dataset, rows, size, device, batch=256):
    """Return (N, 196, 384) patch tokens for the given dataset rows."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def _decode(p):
        if isinstance(p, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(p))))
        return np.asarray(p)

    out = []
    for i in range(0, len(rows), batch):
        idx = rows[i:i + batch].tolist()
        r = dataset.get_row_data(idx)
        px = r['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_decode(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_decode(q) for q in px])
        x = torch.from_numpy(px).to(device).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        if x.shape[-1] != size:
            x = torch.nn.functional.interpolate(
                x, size=size, mode='bilinear', antialias=True, align_corners=False)
        with torch.no_grad():
            e = wm.encode({'pixels': x.unsqueeze(1)}, emb_keys=[])['pixels_emb'][:, 0]
        out.append(e.half().cpu())
    return torch.cat(out)


def ridge_r2(Xtr, ytr, Xte, yte, lambdas=(1e-2, 1e0, 1e2, 1e4, 1e6)):
    """Exact ridge in dual form (n < d). Returns best test R^2 per target."""
    Xtr, Xte = Xtr.float(), Xte.float()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean(0, keepdim=True)
    ytr_c = ytr - ym
    K = Xtr @ Xtr.T
    Kte = Xte @ Xtr.T
    n = K.shape[0]
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    best = None
    for lam in lambdas:
        alpha = torch.linalg.solve(K + lam * eye, ytr_c)
        pred = Kte @ alpha + ym
        ss_res = ((yte - pred) ** 2).sum(0)
        ss_tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-12)
        r2 = (1 - ss_res / ss_tot)
        if best is None or r2.mean() > best.mean():
            best = r2
    return best.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--dataset',
                   default='/workspace/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--n-train', type=int, default=8000)
    p.add_argument('--n-test', type=int, default=4000)
    p.add_argument('--ep-split', type=int, default=8000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    dev = args.device
    rng = np.random.default_rng(args.seed)

    wm = load_pretrained(args.wm).to(dev).eval()
    wm.requires_grad_(False)
    ds = swm.data.load_dataset(args.dataset)

    ep = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    tr_pool = np.nonzero(ep < args.ep_split)[0]
    te_pool = np.nonzero(ep >= args.ep_split)[0]
    tr = np.sort(rng.choice(tr_pool, args.n_train, replace=False))
    te = np.sort(rng.choice(te_pool, args.n_test, replace=False))
    print(f'[probe] {len(tr)} train frames (eps <{args.ep_split}), '
          f'{len(te)} test frames (eps >={args.ep_split})', flush=True)

    print('[probe] encoding tokens...', flush=True)
    Ztr = encode_tokens(wm, ds, tr, args.img_size, dev)
    Zte = encode_tokens(wm, ds, te, args.img_size, dev)
    N, P, D = Ztr.shape
    print(f'[probe] tokens train {tuple(Ztr.shape)} test {tuple(Zte.shape)}', flush=True)

    targets = {}
    for key in ('privileged_block_0_pos', 'proprio_effector_pos'):
        a = np.asarray(ds.get_row_data(tr.tolist())[key], dtype=np.float32)
        b = np.asarray(ds.get_row_data(te.tolist())[key], dtype=np.float32)
        targets[key] = (torch.from_numpy(a.reshape(len(tr), -1)).to(dev),
                        torch.from_numpy(b.reshape(len(te), -1)).to(dev))

    # ---- channel PCA on individual tokens from TRAIN frames only
    flat = Ztr.reshape(-1, D).float()
    mu = flat.mean(0, keepdim=True)
    # covariance on GPU in chunks (flat can be ~1.5M x 384)
    cov = torch.zeros(D, D, device=dev)
    for i in range(0, flat.shape[0], 200_000):
        c = (flat[i:i + 200_000].to(dev) - mu.to(dev))
        cov += c.T @ c
    cov /= flat.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    evr = (evals / evals.sum()).cpu().numpy()
    print('[probe] channel PCA cumulative explained variance: '
          + ' '.join(f'k={k}:{evr[:k].sum():.3f}' for k in (8, 16, 32, 64, 128)),
          flush=True)

    def project(Z, k):
        Pk = evecs[:, :k]
        return (Z.to(dev).float().reshape(-1, D) @ Pk).reshape(Z.shape[0], -1)

    sel = np.array([1, 5, 8, 12])
    grid_idx = (sel[:, None] * 14 + sel[None, :]).reshape(-1)

    reps = {
        'full (75264)': (lambda Z: Z.to(dev).float().reshape(Z.shape[0], -1)),
        'proj k=8 (1568)': (lambda Z: project(Z, 8)),
        'proj k=16 (3136)': (lambda Z: project(Z, 16)),
        'proj k=32 (6272)': (lambda Z: project(Z, 32)),
        'grid4x4 (6144)': (lambda Z: Z.to(dev).float()[:, grid_idx].reshape(Z.shape[0], -1)),
        'meanpool (384)': (lambda Z: Z.to(dev).float().mean(1)),
    }

    print(f'\n{"representation":22s} {"cube xyz R2":>28s} {"effector xyz R2":>28s}')
    for name, fn in reps.items():
        Xtr, Xte = fn(Ztr), fn(Zte)
        line = f'{name:22s}'
        for key in ('privileged_block_0_pos', 'proprio_effector_pos'):
            ytr, yte = targets[key]
            r2 = ridge_r2(Xtr, ytr, Xte, yte)
            line += '   ' + ' '.join(f'{v:6.3f}' for v in r2)
            line += f' | mean {r2.mean():6.3f}'
        print(line, flush=True)
        del Xtr, Xte
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
