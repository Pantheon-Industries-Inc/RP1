"""Build a tiny action-only h5 (action + ep_offset + ep_len) from a lance
dataset, for train_lip_ac.py (which reads ONLY h['action'], h['ep_offset']
and h['ep_len'] — no pixels). Episode order matches the lance = matches the
latent cache. (Reconstruction of pod A's build_action_h5.py, 2026-07-23.)
"""
import sys

import h5py
import numpy as np

import stable_worldmodel as swm

DS = sys.argv[1] if len(sys.argv) > 1 else "/workspace/datasets/ogb_cube_single"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/datasets/expert_actions.h5"

ds = swm.data.load_dataset(DS)
ecol = "ep_idx" if "ep_idx" in ds.column_names else "episode_idx"
epi = np.asarray(ds.get_col_data(ecol)).reshape(-1).astype(np.int64)
act = np.asarray(ds.get_col_data("action")).reshape(len(epi), -1).astype(np.float32)

bounds = np.flatnonzero(np.diff(epi)) + 1
ep_offset = np.concatenate([[0], bounds]).astype(np.int64)
ep_len = np.diff(np.concatenate([ep_offset, [len(epi)]])).astype(np.int64)

with h5py.File(OUT, "w") as f:
    f.create_dataset("action", data=act)
    f.create_dataset("ep_offset", data=ep_offset)
    f.create_dataset("ep_len", data=ep_len)

print(f"wrote {OUT}: action{act.shape} episodes={len(ep_offset)} "
      f"ep_len(min/med/max)={ep_len.min()}/{int(np.median(ep_len))}/{ep_len.max()}")
