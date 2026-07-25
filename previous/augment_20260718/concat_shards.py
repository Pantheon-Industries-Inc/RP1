"""Concatenate per-worker grasp-miss shards into one miss_all.h5 with contiguous
episode numbering. Keeps only the fields the encode + actor pipeline needs
(pixels for encoding; action/ep_offset/ep_len for the actor action lookup;
privileged_block_0_pos as the cache state_key)."""
import glob
import sys

import h5py
import numpy as np

shard_glob, out = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(shard_glob))
assert paths, f"no shards match {shard_glob}"

eps = []   # list of dicts per episode
for p in paths:
    with h5py.File(p, "r") as f:
        ep_len = f["ep_len"][:]
        ep_off = f["ep_offset"][:]
        for i in range(len(ep_len)):
            o, L = int(ep_off[i]), int(ep_len[i])
            eps.append({
                "pixels": f["pixels"][o:o + L],
                "action": f["action"][o:o + L],
                "block": f["privileged_block_0_pos"][o:o + L],
                "L": L,
            })
n_ep = len(eps)
steps = sum(e["L"] for e in eps)
print(f"{len(paths)} shards -> {n_ep} miss episodes, {steps} frames")

with h5py.File(out, "w") as f:
    f.create_dataset("pixels", (steps, 224, 224, 3), dtype="uint8",
                     compression="gzip", compression_opts=2, chunks=(1, 224, 224, 3))
    f.create_dataset("action", (steps, 5), dtype="f4")
    f.create_dataset("privileged_block_0_pos", (steps, 3), dtype="f8")
    f.create_dataset("ep_len", (n_ep,), dtype="i4")
    f.create_dataset("ep_offset", (n_ep,), dtype="i8")
    f.create_dataset("step_idx", (steps,), dtype="i4")
    f.create_dataset("ep_idx", (steps,), dtype="i4")
    row = 0
    for i, e in enumerate(eps):
        L = e["L"]
        f["pixels"][row:row + L] = e["pixels"]
        f["action"][row:row + L] = e["action"]
        f["privileged_block_0_pos"][row:row + L] = e["block"]
        f["step_idx"][row:row + L] = np.arange(L)
        f["ep_idx"][row:row + L] = i
        f["ep_len"][i] = L
        f["ep_offset"][i] = row
        row += L
print(f"wrote {out}: {n_ep} eps, {row} frames")
