"""Build a SLIM reacher h5 (no pixels, canonical column names) for conversion
and for LIP training's --h5 argument.

RECONSTRUCTED 2026-07-30.

Two reasons this file exists:
  1. `convert_reacher_bases.py` and `train_lip_ac.py` read the h5 for ACTIONS,
     episode bookkeeping and action stats -- never pixels. The canonical h5 is
     93G because pixels dominate; everything else is ~250MB, so the slim copy
     makes those steps fast and lets them run while the big file is busy.
  2. The canonical h5 names the episode column `ep_idx`, while the repo's
     loaders expect `episode_idx` (the same mismatch the original campaign hit
     when it had to write a custom build_reacher_h5.py). This renames it.

Row order and every value are preserved, so indices match the caches exactly.
"""

import os

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

SRC = "/workspace/datasets_canon/lewm-reacher/reacher.h5"
DST = "/workspace/reacher_slim.h5"

RENAME = {"ep_idx": "episode_idx"}

src = h5py.File(SRC, "r")
dst = h5py.File(DST, "w")
print(f"source columns: {sorted(src.keys())}")
for k in src.keys():
    if k == "pixels":
        continue
    data = src[k][:]
    name = RENAME.get(k, k)
    dst.create_dataset(name, data=data, compression="gzip", compression_opts=1)
    print(f"  {k} -> {name}: {data.shape} {data.dtype}")

# some loaders want both spellings present; alias episode_idx back to ep_idx
if "episode_idx" in dst and "ep_idx" not in dst:
    dst["ep_idx"] = dst["episode_idx"]
    print("  aliased ep_idx -> episode_idx")

dst.close()
src.close()
print(f"slim h5 written: {os.path.getsize(DST) / 1e6:.0f} MB (pixels omitted)")
