"""Is the learned quasimetric's asymmetry real and structured?

For same-episode pairs (a = z_t, b = z_{t+k}): compute d(a,b) (forward, along
the arrow of data time) vs d(b,a) (reverse). PushT ground truth: reversing is
expensive iff the BLOCK moved (irreversible pushes); agent-only motion is
cheap both ways. So structured asymmetry predicts:
  gap = d(b,a) - d(a,b) grows with block-pose change, not with agent motion.
"""

import sys

import numpy as np
import torch
from scipy.stats import spearmanr

from stable_worldmodel.trm import LatentCache, load_metric

DEV = "cuda"
metric = load_metric(sys.argv[1] if len(sys.argv) > 1 else "/workspace/metrics/td2_e0.01_n1.pt", device=DEV)
metric.eval()

c = LatentCache.load("/workspace/caches/pusht_official_fs1.pt")
z = c.z.float()
s = c.state.float().numpy()
eps = c.episodes()
rng = np.random.default_rng(0)
keys = [k for k in eps if len(eps[k]) > 60]

A, B, dblock, dagent = [], [], [], []
for _ in range(6000):
    e = keys[rng.integers(len(keys))]
    rows = eps[e]
    t = int(rng.integers(0, len(rows) - 55))
    k = int(rng.integers(5, 51))
    ra, rb = rows[t], rows[t + k]
    A.append(z[ra])
    B.append(z[rb])
    dblock.append(float(np.linalg.norm(s[rb, 2:4] - s[ra, 2:4])))
    dagent.append(float(np.linalg.norm(s[rb, 0:2] - s[ra, 0:2])))

A = torch.stack(A).to(DEV)
B = torch.stack(B).to(DEV)
with torch.no_grad():
    fwd = metric(A, B).cpu().numpy()
    rev = metric(B, A).cpu().numpy()
gap = rev - fwd
dblock = np.array(dblock)
dagent = np.array(dagent)

print(f"forward d(a,b): mean {fwd.mean():.2f}")
print(f"reverse d(b,a): mean {rev.mean():.2f}")
print(f"asymmetry gap (rev-fwd): mean {gap.mean():.2f}, |gap|/fwd median {np.median(np.abs(gap)/np.maximum(fwd,1e-6)):.2%}")
print(f"frac pairs with rev > fwd: {(gap > 0).mean():.2%}")
print(f"Spearman(gap, block-pose change) = {spearmanr(gap, dblock).statistic:.3f}")
print(f"Spearman(gap, agent-only motion) = {spearmanr(gap, dagent).statistic:.3f}")
# control: among pairs with tiny block motion, gap should be ~0
m = dblock < np.percentile(dblock, 20)
print(f"low-block-motion pairs: gap mean {gap[m].mean():.2f} (vs all {gap.mean():.2f})")
m2 = dblock > np.percentile(dblock, 80)
print(f"high-block-motion pairs: gap mean {gap[m2].mean():.2f}")
