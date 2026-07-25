"""Which action convention does each WM expect? Roll the WM forward under real
expert actions in two conventions and measure open-loop latent prediction error
vs. a copy-last baseline (ratio <1 = predicts, >1 = worse than doing nothing).
The convention the WM was TRAINED with should give the lowest ratio.

  zscore : (a - mean)/std per-dim (our eval harness convention)
  raw    : a in [-1,1] (no normalization)

If v2WM prefers zscore (as our validated harness assumes) but IDM-LeWM prefers a
different convention, that mismatch explains floor-vs-66% CEM.
"""
import h5py

try:
    import hdf5plugin  # noqa
except ImportError:
    pass
import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.trm import LatentCache
from stable_worldmodel.solver.lip import rollout_traj

DEV, FS = "cuda", 5
H5 = "/workspace/datasets/lewm_cube_full/cube_single_expert.h5"
WMS = [("v2WM", "/workspace/ckpts/ogbench_cube_single_v2WM", "/workspace/caches/cube_v2wm_fs1.pt"),
       ("IDM-LeWM", "/workspace/ckpts/idm_lewm", "/workspace/caches/cube_idm_fs1.pt")]

with h5py.File(H5, "r") as h:
    a_stat = h["action"][:1000000]
    e_off = h["ep_offset"][:]
amu = np.nanmean(a_stat, 0)
astd = np.nanstd(a_stat, 0) + 1e-6


def blocks(act, conv):
    an = (act - amu) / astd if conv == "zscore" else act
    n = len(an) // FS
    return an[:n * FS].reshape(n, FS * 5).astype(np.float32)


for name, wm_path, cache_path in WMS:
    m = swm.wm.utils.load_pretrained(wm_path).to(DEV).eval()
    m.requires_grad_(False)
    c = LatentCache.load(cache_path)
    eps = c.episodes()
    with h5py.File(H5, "r") as h:
        for conv in ["zscore", "raw"]:
            pse = cse = 0.0
            for e in list(eps)[:40]:
                rows = eps[e]
                z_fs5 = c.z[rows[::FS]]
                act = h["action"][int(e_off[e]):int(e_off[e]) + len(rows)]
                nanr = np.isnan(act).any(1)
                if nanr.any():
                    act = act[:int(np.argmax(nanr))]
                blk = blocks(act, conv)
                T = min(len(z_fs5), len(blk))
                if T < 8:
                    continue
                zh = z_fs5[:3].unsqueeze(0).to(DEV)
                ah = torch.zeros(1, 2, FS * 5, device=DEV)
                plan = torch.from_numpy(blk[2:T]).unsqueeze(0).to(DEV)
                with torch.no_grad():
                    imag = rollout_traj(m, zh, ah, plan)[0].cpu()
                real = z_fs5[2:T]
                last = z_fs5[1:T - 1]
                pse += ((imag - real) ** 2).sum().item()
                cse += ((last - real) ** 2).sum().item()
            print(f"{name:9s} {conv:7s}: pred/copy-last MSE ratio = {pse / cse:.3f}")
    print()
