"""Cache mean-pooled DINO-WM (PreJEPA) latents for the TD / LIP stack.

Why this exists: the stock `scripts/trm/cache_latents.py` featurizer returns
`wm.encode(...)['emb'][:, 0]`, which for LeWM is a flat 192-d vector (CLS token
-> projector) but for PreJEPA is a `(n_patches, D)` token grid. The whole
cache -> TD -> LIPv4 -> metric-cost chain assumes a flat per-frame latent, so
PreJEPA needs an explicit reduction.

Reduction used: mean over the 196 patch tokens of the PIXEL embedding only
(384-d), i.e. `info['pixels_emb'].mean(patches)`. Rationale:
  * matches this repo's own DinoWM._encode_pixels, which does
    `h[:, 1:, :].mean(dim=1)` (mean-pool patches, drop CLS);
  * uses `pixels_emb` rather than `emb` so the tiled ACTION embedding is not
    baked into what is supposed to be a state latent.
The identical reduction must be applied at plan time (see pool_prejepa.py).

Image pipeline is matched to eval_wm.img_transform EXACTLY -- ImageNet
normalise at native resolution, THEN resize to `--img-size`. Getting this order
wrong would put the cache in a different image domain than the planner.

Usage:
  cache_latents_dino.py --wm DIR --dataset LANCE --out CACHE.pt \
      --img-size 196 --state-key privileged_block_0_pos
"""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.trm import encode_dataset  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--state-key', default=None)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-rows', type=int, default=None,
                   help='cap rows (smoke test only)')
    args = p.parse_args()

    device = args.device
    wm = load_pretrained(args.wm).to(device).eval()
    wm.requires_grad_(False)
    # honour the checkpoint's config: HF Dinov2Model.forward rejects the kwarg
    print(f'[cache] interpolate_pos_encoding={wm.interpolate_pos_encoding}', flush=True)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    size = int(args.img_size)

    def _decode(px):
        if isinstance(px, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(px))))
        return np.asarray(px)

    reported = {}

    @torch.no_grad()
    def featurize(rows):
        px = rows['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_decode(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_decode(q) for q in px])
        x = torch.from_numpy(px).to(device).permute(0, 3, 1, 2).float().div_(255)
        # eval_wm.img_transform order: normalise first, resize second
        x = (x - mean) / std
        if x.shape[-1] != size:
            x = torch.nn.functional.interpolate(
                x, size=size, mode='bilinear', antialias=True, align_corners=False
            )
        out = wm.encode({'pixels': x.unsqueeze(1)}, emb_keys=[])
        tok = out['pixels_emb'][:, 0]          # (B, P, 384) patch tokens
        z = tok.mean(dim=1)                    # (B, 384) mean-pool over patches
        if not reported:
            reported['shape'] = True
            print(f'[cache] pixels {tuple(x.shape)} -> tokens {tuple(tok.shape)} '
                  f'-> pooled {tuple(z.shape)}', flush=True)
        return z

    dataset = swm.data.load_dataset(args.dataset)

    if args.max_rows is not None:
        full = dataset
        cap = int(args.max_rows)

        class _Sub:
            column_names = full.column_names

            def get_col_data(self, c):
                return full.get_col_data(c)[:cap]

            def get_row_data(self, i):
                return full.get_row_data(i)

        dataset = _Sub()

    cache = encode_dataset(
        dataset, featurize, batch_size=args.batch_size, state_key=args.state_key,
        meta={'wm': args.wm, 'dataset': args.dataset, 'img_size': size,
              'reduction': 'mean-pool patch tokens of pixels_emb'},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)
    print(f'[cache] {len(cache.z)} latents (dim={cache.latent_dim}) -> {args.out}',
          flush=True)


if __name__ == '__main__':
    main()
