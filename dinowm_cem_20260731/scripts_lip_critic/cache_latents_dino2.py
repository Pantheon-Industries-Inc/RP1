"""Cache DINO-WM latents under TWO spatially-faithful reductions, one pass.

Neither reduction mixes across space, which is the objection to mean-pooling
(measured: mean-pool keeps cube-position R^2 0.915 vs 0.976 for the full grid,
but its TD monotonicity collapses to 0.58-0.64 vs LeWM's 0.92).

  proj{k}   per-token channel PCA 384->k, all 196 tokens kept, flattened.
            Spatial layout fully preserved; only channels compressed.
            PCA components are ORDERED, so the first k' < k columns of this
            cache reshaped as (196, k)[:, :, :k'] ARE the proj{k'} cache --
            k=16/32 substrates are derived by slicing, no re-encode.
  grid4x4   all 384 channels at 16 strided token positions (SELECTION, no
            averaging anywhere) -> 6144.

Writes train (episodes < --ep-split) and held-out (>=) caches separately, so no
484 GB filter pass is needed. Caches go to LOCAL disk: the MFS /workspace volume
measured ~63 MB/s, which would dominate everything.

Image pipeline matches eval_wm.img_transform EXACTLY -- ImageNet-normalise at
native resolution, THEN resize -- or cache and planner sit in different domains.
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
from stable_worldmodel.trm import LatentCache  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wm', default='/workspace/ckpts/dinowm_noprop_cube')
    p.add_argument('--dataset',
                   default='/root/datasets/ogb_cube_single/ogb_cube_single.lance')
    p.add_argument('--proj', default='/workspace/ckpts/dino_chan_pca_k128.pt')
    p.add_argument('--k', type=int, default=64)
    p.add_argument('--out-dir', default='/root/caches')
    p.add_argument('--tag', default='dino')
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

    pay = torch.load(args.proj, map_location='cpu', weights_only=False)
    P = pay['P'][:, :args.k].to(dev)            # (384, k)
    tok_mean = pay['mean'].to(dev)              # (384,)
    print(f'[cache] proj P {tuple(P.shape)} from {args.proj} '
          f'(fitted on eps <{pay["ep_hi"]})', flush=True)

    sel = np.array([1, 5, 8, 12])
    grid_idx = torch.from_numpy(
        (sel[:, None] * 14 + sel[None, :]).reshape(-1)).to(dev)

    ds = swm.data.load_dataset(args.dataset)
    epi = np.asarray(ds.get_col_data('episode_idx')).reshape(-1).astype(np.int64)
    stp = np.asarray(ds.get_col_data('step_idx')).reshape(-1).astype(np.int64)
    n = len(epi)
    is_tr = epi < args.ep_split
    tr_rows, ho_rows = np.nonzero(is_tr)[0], np.nonzero(~is_tr)[0]
    print(f'[cache] {n} rows -> train {len(tr_rows)} (eps <{args.ep_split}) '
          f'/ held-out {len(ho_rows)}', flush=True)

    Dp, Dg = 196 * args.k, 16 * 384
    bufs = {
        ('proj', 'tr'): torch.empty(len(tr_rows), Dp, dtype=torch.float32),
        ('proj', 'ho'): torch.empty(len(ho_rows), Dp, dtype=torch.float32),
        ('grid', 'tr'): torch.empty(len(tr_rows), Dg, dtype=torch.float32),
        ('grid', 'ho'): torch.empty(len(ho_rows), Dg, dtype=torch.float32),
    }
    gb = sum(b.numel() * 4 for b in bufs.values()) / 1e9
    print(f'[cache] preallocated {gb:.1f} GB (proj dim {Dp}, grid dim {Dg})', flush=True)

    # running write cursors per split
    cur = {'tr': 0, 'ho': 0}
    state_chunks = {'tr': [], 'ho': []}

    def _decode(q):
        if isinstance(q, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(q))))
        return np.asarray(q)

    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    size = int(args.img_size)
    reported = []

    from tqdm import tqdm
    for start in tqdm(range(0, n, args.batch_size), desc='encoding'):
        idx = list(range(start, min(start + args.batch_size, n)))
        rows = ds.get_row_data(idx)
        px = rows['pixels']
        if isinstance(px, np.ndarray) and px.dtype == object:
            px = np.stack([_decode(q) for q in px])
        else:
            px = np.asarray(px)
            if px.ndim != 4:
                px = np.stack([_decode(q) for q in px])
        x = torch.from_numpy(px).to(dev).permute(0, 3, 1, 2).float().div_(255)
        x = (x - mean) / std
        if x.shape[-1] != size:
            x = torch.nn.functional.interpolate(
                x, size=size, mode='bilinear', antialias=True, align_corners=False)
        with torch.no_grad():
            tok = wm.encode({'pixels': x.unsqueeze(1)}, emb_keys=[])['pixels_emb'][:, 0]
        zp = ((tok - tok_mean) @ P).reshape(tok.shape[0], -1)     # (B, 196*k)
        zg = tok[:, grid_idx].reshape(tok.shape[0], -1)           # (B, 16*384)
        if not reported:
            reported.append(1)
            print(f'\n[cache] tokens {tuple(tok.shape)} -> proj {tuple(zp.shape)} '
                  f'/ grid {tuple(zg.shape)}', flush=True)

        b_epi = epi[idx]
        m_tr = torch.from_numpy(b_epi < args.ep_split)
        st = np.asarray(rows[args.state_key], dtype=np.float32).reshape(len(idx), -1)
        for split, mask in (('tr', m_tr), ('ho', ~m_tr)):
            c = int(mask.sum())
            if c == 0:
                continue
            s, e = cur[split], cur[split] + c
            bufs[('proj', split)][s:e] = zp[mask.to(dev)].cpu()
            bufs[('grid', split)][s:e] = zg[mask.to(dev)].cpu()
            state_chunks[split].append(st[mask.numpy()])
            cur[split] = e

    assert cur['tr'] == len(tr_rows) and cur['ho'] == len(ho_rows), \
        f'row accounting mismatch: {cur} vs {len(tr_rows)}/{len(ho_rows)}'
    print(f'[cache] filled: {cur}', flush=True)

    for rep, dim in (('proj', Dp), ('grid', Dg)):
        for split, rws in (('tr', tr_rows), ('ho', ho_rows)):
            name = (f'{args.tag}_{rep}{args.k}_' if rep == 'proj'
                    else f'{args.tag}_{rep}_') + (
                        f'tr{args.ep_split}' if split == 'tr' else 'ho')
            c = LatentCache(
                z=bufs[(rep, split)],
                episode_idx=torch.from_numpy(epi[rws]),
                step_idx=torch.from_numpy(stp[rws]),
                state=torch.from_numpy(np.concatenate(state_chunks[split], axis=0)),
                meta={'wm': args.wm, 'dataset': args.dataset,
                      'img_size': size, 'reduction': rep, 'k': args.k,
                      'proj': args.proj, 'ep_split': args.ep_split,
                      'note': 'no spatial mixing; proj = per-token channel PCA, '
                              'grid = strided token selection'},
            )
            path = out / f'{name}.pt'
            c.save(str(path))
            print(f'[cache] {name}: {len(c.z)} x {c.latent_dim} -> {path} '
                  f'({path.stat().st_size / 1e9:.1f} GB)', flush=True)
    print('[cache] DONE', flush=True)


if __name__ == '__main__':
    main()
