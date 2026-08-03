"""Fit the per-token channel PCA used to reduce DINO-WM latents.

The reduction keeps all 196 patch tokens (spatial layout untouched) and
compresses only each token's 384 channels with ONE shared matrix
P in R^{384 x k}. Fitted on individual token vectors drawn from TRAINING
episodes only (0-7999), so the transform carries no held-out information.

Saved payload is consumed by both cache_latents_dino2.py (cache side) and
eval_wm_dino.py (plan side) -- they must apply the identical matrix or the
metric sees a different space than it was trained on.

Usage: fit_channel_pca.py --k 32 --out /workspace/ckpts/dino_chan_pca_k32.pt
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--dataset',
                   default='/workspace/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--n-frames', type=int, default=12000)
    p.add_argument('--ep-hi', type=int, default=8000, help='fit on episodes < this')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    dev = args.device
    wm = load_pretrained(args.wm).to(dev).eval()
    wm.requires_grad_(False)
    ds = swm.data.load_dataset(args.dataset)

    ep = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    pool = np.nonzero(ep < args.ep_hi)[0]
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(pool, min(args.n_frames, len(pool)), replace=False))
    print(f'[pca] fitting on {len(rows)} frames from episodes <{args.ep_hi}', flush=True)

    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    def _decode(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    # streaming first and second moments over token vectors
    D = None
    n_tok = 0
    s1 = None
    s2 = None
    for i in range(0, len(rows), 256):
        idx = rows[i:i + 256].tolist()
        px = ds.get_row_data(idx)['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_decode(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_decode(q) for q in px])
        x = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        if x.shape[-1] != args.img_size:
            x = torch.nn.functional.interpolate(
                x, size=args.img_size, mode='bilinear', antialias=True,
                align_corners=False)
        with torch.no_grad():
            tok = wm.encode({'pixels': x.unsqueeze(1)}, emb_keys=[])['pixels_emb'][:, 0]
        t = tok.reshape(-1, tok.shape[-1]).double()
        if s1 is None:
            D = t.shape[1]
            s1 = torch.zeros(D, dtype=torch.float64, device=dev)
            s2 = torch.zeros(D, D, dtype=torch.float64, device=dev)
        s1 += t.sum(0)
        s2 += t.T @ t
        n_tok += t.shape[0]

    mu = s1 / n_tok
    cov = s2 / n_tok - torch.outer(mu, mu)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    evr = (evals / evals.sum()).cpu().numpy()
    print('[pca] cumulative EVR: '
          + ' '.join(f'k={k}:{evr[:k].sum():.4f}' for k in (8, 16, 32, 64, 128)),
          flush=True)

    payload = {
        'P': evecs[:, :args.k].float().cpu(),      # (384, k)
        'mean': mu.float().cpu(),                  # (384,) token mean
        'k': args.k,
        'n_tokens': n_tok,
        'evr_cum': float(evr[:args.k].sum()),
        'ep_hi': args.ep_hi,
        'img_size': args.img_size,
        'note': 'per-token channel PCA; apply as (z - mean) @ P then flatten '
                'over the 196 patch axis. Spatial layout preserved.',
    }
    torch.save(payload, args.out)
    print(f'[pca] saved P {tuple(payload["P"].shape)} (EVR {payload["evr_cum"]:.4f}) '
          f'from {n_tok} tokens -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
