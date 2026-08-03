"""Is the TD head UNDERFITTING or failing to GENERALIZE?

The value head is supposed to absorb whatever geometry the encoder provides, so
"DINOv2 features are not trajectory-ordered" does not explain a monotonicity of
0.70 -- it only restates that the head did not learn an ordering. And widening
the head 256 -> 1024 changed monotone by 0.0000, so it is not expressivity.

This discriminates the two remaining stories by scoring the SAME metric on
episodes it TRAINED on (0-7999) versus held-out ones (8000-9999):

  train ~= held-out ~= 0.70  -> UNDERFIT / optimization. The head cannot fit
                               even seen data => conditioning (raw 75k-d inputs
                               with wildly varying per-dim scale) or a weak TD
                               objective. Standardising inputs is the fix.
  train >> held-out          -> GENERALIZATION. Capacity/regularisation story.

It also reports ordering at the PLANNER's horizon (25-step goal offset, which
is what CEM consumes) alongside the long-span number, because the original probe
scored spans up to 60 steps and may simply have measured the wrong thing.

Frames are re-encoded from the dataset (fast: a few thousand per second) rather
than loading the 484 GB cache, so this runs in ~2 min on a spare GPU.
"""

import argparse
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.trm import load_metric  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def encode(wm, ds, rows, size, dev, batch=256):
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    def _dec(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    out = []
    for i in range(0, len(rows), batch):
        idx = rows[i:i + batch].tolist()
        px = ds.get_row_data(idx)['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_dec(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_dec(q) for q in px])
        x = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        if x.shape[-1] != size:
            x = torch.nn.functional.interpolate(
                x, size=size, mode='bilinear', antialias=True, align_corners=False)
        with torch.no_grad():
            tok = wm.encode({'pixels': x.unsqueeze(1)}, emb_keys=[])['pixels_emb'][:, 0]
        out.append(tok.reshape(tok.shape[0], -1).cpu())
    return torch.cat(out)


def score(metric, wm, ds, epi, stp, eps, size, dev, span, n_eps, seed=0):
    """monotone over `span` steps + pair_acc, on the given episode ids."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(eps, size=min(n_eps, len(eps)), replace=False)
    mono, pair_hits, pair_tot = [], 0, 0
    for e in pick:
        rows = np.nonzero(epi == e)[0]
        rows = rows[np.argsort(stp[rows])]
        if len(rows) < span + 2:
            continue
        rows = rows[:span + 1]
        z = encode(wm, ds, rows, size, dev).to(dev).float()
        zg = z[-1:].expand(len(z) - 1, -1)
        with torch.no_grad():
            d = metric.cost(z[:-1], zg).float().cpu().numpy().reshape(-1)
        if not np.isfinite(d).all() or len(d) < 2:
            continue
        mono.append(float((np.diff(d) < 0).mean()))
        # pair ordering: closer-in-time should score lower
        for _ in range(20):
            i, j = rng.integers(0, len(d), 2)
            if i == j:
                continue
            pair_tot += 1
            pair_hits += int((d[i] < d[j]) == (i > j))
    return (float(np.mean(mono)) if mono else float('nan'),
            pair_hits / max(pair_tot, 1), len(mono))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--metric', default='/workspace/metrics/dinofull_e0.1_lr0.0003.pt')
    p.add_argument('--dataset',
                   default='/root/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--ep-split', type=int, default=8000)
    p.add_argument('--n-eps', type=int, default=60)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    dev = args.device
    wm = load_pretrained(args.wm).to(dev).eval()
    wm.requires_grad_(False)
    metric = load_metric(args.metric).to(dev).eval()
    print(f'[fit] metric {args.metric}', flush=True)

    ds = swm.data.load_dataset(args.dataset)
    epi = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    stp = np.asarray(ds.get_col_data('step_idx')).reshape(-1).astype(np.int64)
    tr_eps = np.unique(epi[epi < args.ep_split])
    ho_eps = np.unique(epi[epi >= args.ep_split])

    print(f'\n{"split":10s} {"span":>5s} {"monotone":>9s} {"pair_acc":>9s} {"n_eps":>6s}')
    for span in (25, 60):
        for nm, eps in (('TRAIN', tr_eps), ('HELD-OUT', ho_eps)):
            mo, pa, n = score(metric, wm, ds, epi, stp, eps, args.img_size,
                              dev, span, args.n_eps)
            print(f'{nm:10s} {span:5d} {mo:9.4f} {pa:9.4f} {n:6d}', flush=True)


if __name__ == '__main__':
    main()
