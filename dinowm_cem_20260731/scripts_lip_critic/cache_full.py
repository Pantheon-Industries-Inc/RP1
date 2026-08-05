"""Cache the FULL DINO-WM token grid -- no reduction of any kind.

Each frame's latent is the complete 14x14 patch grid flattened:
196 tokens x 384 channels = 75,264 dims. No pooling, no projection, no
selection. This is the zero-choice baseline against which the earlier
mean-pooled result (cube-position R^2 0.915, TD monotonicity 0.58-0.64 vs
LeWM's 0.92) is judged.

Sizes: train (episodes < ep-split) 1.608M x 75,264 x 4 B = 484 GB;
held-out 402k rows = 121 GB. Both are preallocated in RAM (1,979 GB available)
and written to /workspace, measured at 500 MB/s here => ~20 min.

The dataset is read from LOCAL disk (fast random reads); only the big caches go
to the network volume, which has 94 TB.

Image pipeline matches eval_wm.img_transform EXACTLY -- ImageNet-normalise at
native resolution, THEN resize to 196 -- otherwise cache and planner occupy
different image domains.
"""

import argparse
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/workspace/code/stable-worldmodel')

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.trm import LatentCache  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--dataset',
                   default='/root/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--out-dir', default='/workspace/caches')
    p.add_argument('--tag', default='dinofull')
    p.add_argument('--img-size', type=int, default=196)
    p.add_argument('--ep-split', type=int, default=8000)
    p.add_argument('--state-key', default='privileged_block_0_pos')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    dev = args.device
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    wm = load_pretrained(args.wm).to(dev).eval()
    wm.requires_grad_(False)
    print(f'[cache] wm loaded, interpolate_pos_encoding={wm.interpolate_pos_encoding}',
          flush=True)

    ds = swm.data.load_dataset(args.dataset)
    epi = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    stp = np.asarray(ds.get_col_data('step_idx')).reshape(-1).astype(np.int64)
    n = len(epi)
    is_tr = epi < args.ep_split
    tr_rows, ho_rows = np.nonzero(is_tr)[0], np.nonzero(~is_tr)[0]

    # probe one batch to learn the exact flattened dim rather than assume it
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    size = int(args.img_size)

    def _decode(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    def encode_batch(idx):
        rows = ds.get_row_data(idx)
        px = rows['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_decode(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_decode(q) for q in px])
        x = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std                      # normalise BEFORE resize
        if x.shape[-1] != size:
            x = torch.nn.functional.interpolate(
                x, size=size, mode='bilinear', antialias=True, align_corners=False)
        with torch.no_grad():
            tok = wm.encode({'pixels': x.unsqueeze(1)}, emb_keys=[])['pixels_emb'][:, 0]
        return tok, rows

    tok0, _ = encode_batch(list(range(min(8, n))))
    D = tok0.shape[1] * tok0.shape[2]
    print(f'[cache] tokens {tuple(tok0.shape)} -> flat dim {D} '
          f'(NO reduction)', flush=True)
    print(f'[cache] {n} rows -> train {len(tr_rows)} / held-out {len(ho_rows)}; '
          f'{len(tr_rows) * D * 4 / 1e9:.0f} GB + {len(ho_rows) * D * 4 / 1e9:.0f} GB',
          flush=True)

    t0 = time.time()
    z_tr = torch.empty(len(tr_rows), D, dtype=torch.float32)
    z_ho = torch.empty(len(ho_rows), D, dtype=torch.float32)
    print(f'[cache] preallocated {(z_tr.numel() + z_ho.numel()) * 4 / 1e9:.0f} GB '
          f'in {time.time() - t0:.0f}s', flush=True)

    cur = {'tr': 0, 'ho': 0}
    st_chunks = {'tr': [], 'ho': []}
    t0 = time.time()
    for start in tqdm(range(0, n, args.batch_size), desc='encoding'):
        idx = list(range(start, min(start + args.batch_size, n)))
        tok, rows = encode_batch(idx)
        z = tok.reshape(tok.shape[0], -1)
        b_epi = epi[idx]
        m_tr = torch.from_numpy(b_epi < args.ep_split)
        st = np.asarray(rows[args.state_key], dtype=np.float32).reshape(len(idx), -1)
        for split, mask, buf in (('tr', m_tr, z_tr), ('ho', ~m_tr, z_ho)):
            c = int(mask.sum())
            if c == 0:
                continue
            s, e = cur[split], cur[split] + c
            buf[s:e] = z[mask.to(dev)].cpu()
            st_chunks[split].append(st[mask.numpy()])
            cur[split] = e
    print(f'[cache] encoded {n} rows in {(time.time() - t0) / 60:.1f} min; '
          f'filled {cur}', flush=True)
    assert cur['tr'] == len(tr_rows) and cur['ho'] == len(ho_rows), 'row accounting'

    for split, rws, buf in (('tr', tr_rows, z_tr), ('ho', ho_rows, z_ho)):
        name = f'{args.tag}_' + (f'tr{args.ep_split}' if split == 'tr' else 'ho')
        c = LatentCache(
            z=buf,
            episode_idx=torch.from_numpy(epi[rws]),
            step_idx=torch.from_numpy(stp[rws]),
            state=torch.from_numpy(np.concatenate(st_chunks[split], axis=0)),
            meta={'wm': args.wm, 'dataset': args.dataset, 'img_size': size,
                  'reduction': 'none (full 196x384 token grid flattened)',
                  'ep_split': args.ep_split},
        )
        path = out / f'{name}.pt'
        t0 = time.time()
        c.save(str(path))
        gb = path.stat().st_size / 1e9
        print(f'[cache] {name}: {len(c.z)} x {c.latent_dim} -> {path} '
              f'({gb:.0f} GB in {(time.time() - t0) / 60:.1f} min, '
              f'{gb * 1000 / max(time.time() - t0, 1):.0f} MB/s)', flush=True)
    print('[cache] DONE', flush=True)


if __name__ == '__main__':
    main()
