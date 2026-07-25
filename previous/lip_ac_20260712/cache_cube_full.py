"""Latent cache from the lewm-cube full h5 (stored 224px frames), GPU-batched.

Adapted from the project's cache_cube_h5.py (numerically eval-matched transform:
float/255 -> ImageNet normalize -> bilinear-antialias resize) with three changes:
--stride (fs1/fs5 caches), --ep-start/--ep-end sharding (one shard per GPU,
merge with merge_caches.py), and per-episode sequential h5 reads (fast gzip
chunk decode; subsampling happens in numpy, never via h5 fancy indexing).
"""
import argparse

import h5py

try:
    import hdf5plugin  # noqa: F401  (registers the compression filter the cube h5 uses)
except ImportError:
    pass
import numpy as np
import torch
import torch.nn.functional as F

import stable_worldmodel as swm
from stable_worldmodel.trm import LatentCache

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--ep-start", type=int, default=0)
    p.add_argument("--ep-end", type=int, default=10 ** 9)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--state-key", default="privileged_block_0_pos")
    a = p.parse_args()

    dev = "cuda"
    m = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    m.requires_grad_(False)
    m.interpolate_pos_encoding = True
    mean, std = MEAN.to(dev), STD.to(dev)

    def encode(frames_np):
        x = torch.from_numpy(frames_np).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - mean) / std
        if x.shape[-1] != a.img_size:
            x = F.interpolate(x, size=a.img_size, mode="bilinear",
                              antialias=True, align_corners=False)
        with torch.no_grad():
            return m.encode({"pixels": x.unsqueeze(1)})["emb"][:, 0].float().cpu()

    with h5py.File(a.h5, "r") as h:
        ln = h["ep_len"][:].astype(np.int64)
        off = (h["ep_offset"][:].astype(np.int64) if "ep_offset" in h
               else np.concatenate([[0], np.cumsum(ln)[:-1]]))
        e_end = min(a.ep_end, len(ln))
        px_ds = h["pixels"]
        st_ds = h[a.state_key] if a.state_key in h else None

        zs, states, epi, sti = [], [], [], []
        buf, bs = [], a.batch_size
        n_rows = 0
        for e in range(a.ep_start, e_end):
            ts = np.arange(0, ln[e], a.stride)
            frames = px_ds[off[e]:off[e] + ln[e]]      # sequential chunk read
            buf.append(frames[ts])
            if st_ds is not None:
                states.append(np.asarray(st_ds[off[e]:off[e] + ln[e]])[ts])
            epi.append(np.full(len(ts), e, dtype=np.int64))
            sti.append(ts // a.stride)
            n_rows += len(ts)
            if sum(len(b) for b in buf) >= bs or e == e_end - 1:
                block = np.concatenate(buf)
                buf = []
                base = len(zs)
                while True:                             # OOM-safe encode of the block
                    try:
                        for i0 in range(0, len(block), bs):
                            zs.append(encode(block[i0:i0 + bs]))
                        break
                    except torch.cuda.OutOfMemoryError:
                        del zs[base:]                   # drop partial block, retry smaller
                        torch.cuda.empty_cache()
                        bs = max(16, bs // 2)
                        print(f"  OOM -> batch {bs}", flush=True)
            if (e - a.ep_start) % 200 == 0:
                print(f"  ep {e}/{e_end} rows {n_rows}", flush=True)

    LatentCache(
        z=torch.cat(zs),
        episode_idx=torch.from_numpy(np.concatenate(epi)),
        step_idx=torch.from_numpy(np.concatenate(sti)),
        state=(torch.from_numpy(np.concatenate(states).astype(np.float32)) if states else None),
        meta={"wm": a.wm, "h5": a.h5, "stride": a.stride,
              "ep_range": [a.ep_start, e_end], "note": "gpu-batched eval-matched transform"},
    ).save(a.out)
    print(f"cached {n_rows} latents -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
