"""Rotation-sensitivity probe: does the value price block angle error?

Success needs position error < 20 px AND angle error < 20°. If d(z, g)
responds to position but not angle, planners park at the right XY with the
wrong theta and the value declares victory — explaining a planner-independent
failure core concentrated at high-rotation tasks.

For each metric: Spearman(d, angle_diff) over pairs matched in block+agent
position (angle series), vs Spearman(d, pos_diff) over pairs matched in angle
(position series). Also mean d at angle buckets (20°/60°/120°) for scale.
"""
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from stable_worldmodel.trm import LatentCache, load_metric

dev = "cuda" if torch.cuda.is_available() else "cpu"
c = LatentCache.load("/workspace/caches/pusht_official_fs1.pt")
z = c.z.float().to(dev)
s = c.state.float().numpy()
N = len(z)
rng = np.random.default_rng(0)

pos, th, apos = s[:, 2:4], s[:, 4], s[:, 0:2]
anchors = rng.choice(N, size=500, replace=False)
cand = rng.choice(N, size=150000, replace=False)

pairs_ang, pairs_pos = [], []   # (anchor_row, cand_row, angle_deg / pos_px)
for a in anchors:
    pd = np.linalg.norm(pos[cand] - pos[a], axis=1)
    ad = np.degrees(np.abs((th[cand] - th[a] + np.pi) % (2 * np.pi) - np.pi))
    agd = np.linalg.norm(apos[cand] - apos[a], axis=1)
    m1 = (pd < 25) & (agd < 60)
    for r, x in zip(cand[m1][:30], ad[m1][:30]):
        pairs_ang.append((a, r, x))
    m2 = (ad < 8) & (agd < 60) & (pd < 150)
    for r, x in zip(cand[m2][:30], pd[m2][:30]):
        pairs_pos.append((a, r, x))

print(f"pairs: angle-series {len(pairs_ang)}, position-series {len(pairs_pos)}")

def dvals(metric, pairs):
    a = torch.as_tensor([p[0] for p in pairs], device=dev)
    b = torch.as_tensor([p[1] for p in pairs], device=dev)
    with torch.no_grad():
        return metric(z[a], z[b]).cpu().numpy()

for path in sys.argv[1:]:
    m = load_metric(path, device=dev)
    da = dvals(m, pairs_ang)
    dp = dvals(m, pairs_pos)
    xa = np.array([p[2] for p in pairs_ang])
    xp = np.array([p[2] for p in pairs_pos])
    ra = spearmanr(da, xa).statistic
    rp = spearmanr(dp, xp).statistic
    name = path.split("/")[-1]
    b1 = da[xa < 20].mean() if (xa < 20).any() else float("nan")
    b2 = da[(xa > 40) & (xa < 80)].mean() if ((xa > 40) & (xa < 80)).any() else float("nan")
    b3 = da[xa > 100].mean() if (xa > 100).any() else float("nan")
    print(f"{name}")
    print(f"  Spearman(d, angle | pos matched):  {ra:5.2f}   "
          f"Spearman(d, pos | angle matched): {rp:5.2f}")
    print(f"  mean d at angle <20/40-80/>100 deg: {b1:5.1f} / {b2:5.1f} / {b3:5.1f}",
          flush=True)
