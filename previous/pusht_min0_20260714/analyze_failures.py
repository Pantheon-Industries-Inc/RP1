"""Failure-geometry analysis for the s42 h25 draw.

Reconstructs the 50 drawn start rows from an eval log (the harness prints
them), pulls start/goal states from the h5, and characterizes the failure
groups found in the cross-method matrix.
"""
import re
import sys

import h5py
import numpy as np

LOG = "/workspace/logs/eval_repro_tdcem_h25_s42.log"
H5 = "/workspace/swm_home/datasets/pusht_expert_train.h5"
OFF = 25

log = open(LOG).read()
m = re.search(r"valid starting points found.*?\n\[([\s\S]*?)\]", log)
rows = np.array([int(x) for x in m.group(1).split()])
print("drawn rows:", len(rows))

h = h5py.File(H5, "r")
s0 = np.stack([h["state"][r] for r in rows])
s1 = np.stack([h["state"][r + OFF] for r in rows])

# state: [agent_x, agent_y, block_x, block_y, block_th, vel_x, vel_y]
dblock = np.linalg.norm(s1[:, 2:4] - s0[:, 2:4], axis=1)
dth = np.degrees(np.abs((s1[:, 4] - s0[:, 4] + np.pi) % (2 * np.pi) - np.pi))
dagent = np.linalg.norm(s1[:, 0:2] - s0[:, 0:2], axis=1)
gap0 = np.linalg.norm(s0[:, 0:2] - s0[:, 2:4], axis=1)
vel0 = np.linalg.norm(s0[:, 5:7], axis=1)

# distance of block start/goal from arena center (walls at 0/512?)
c = 256.0
edge0 = np.max(np.abs(s0[:, 2:4] - c), axis=1)
edge1 = np.max(np.abs(s1[:, 2:4] - c), axis=1)

groups = {
    "LIP-core 14,15,48": [14, 15, 48],
    "tandem-only 31,32,39": [31, 32, 39],
    "CEM-only 17,23,42,47": [17, 23, 42, 47],
}
fail_all = sorted({e for g in groups.values() for e in g})
solved = sorted(set(range(50)) - set(fail_all))


def stats(ix):
    return (f"blockD {dblock[ix].mean():6.1f}  rotD {dth[ix].mean():6.1f}  "
            f"agentD {dagent[ix].mean():6.1f}  gap0 {gap0[ix].mean():6.1f}  "
            f"vel0 {vel0[ix].mean():5.2f}  edge0 {edge0[ix].mean():6.1f}  "
            f"edge1 {edge1[ix].mean():6.1f}")


for k, ix in groups.items():
    print(f"{k:22s} {stats(ix)}")
print(f"{'solved (' + str(len(solved)) + ')':22s} {stats(solved)}")
print()
for e in fail_all:
    print(f"ep{e:2d} row {rows[e]:7d}: blockD {dblock[e]:6.1f} rotD {dth[e]:6.1f} "
          f"agentD {dagent[e]:6.1f} gap0 {gap0[e]:6.1f} vel0 {vel0[e]:5.2f} "
          f"edge0 {edge0[e]:6.1f} edge1 {edge1[e]:6.1f}")
