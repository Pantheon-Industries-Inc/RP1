"""Far-field probe: is the value usable beyond the 25-step plan horizon?

For within-episode triples (t, t+25, t+50):
  P1 range order:    d(z_t, z_{t+50})  >  d(z_t, z_{t+25})          (farther is farther)
  P2 progress@50:    d(z_{t+25}, z_{t+50})  <  d(z_t, z_{t+50})     (the h50 replan-1 signal)
  P3 progress@25:    d(z_{t+12}, z_{t+25})  <  d(z_t, z_{t+25})     (the h25 analog, control)
Plus mean predicted distance per true-gap bucket (calibration curve).
Chance = 50%. If P3 is high but P2 ~ chance, the far field is mush -> h50 root cause.
"""
import sys

import numpy as np
import torch

from stable_worldmodel.trm import LatentCache, load_metric

dev = "cuda" if torch.cuda.is_available() else "cpu"
c = LatentCache.load("/workspace/caches/pusht_official_fs1.pt")
z = c.z.float().to(dev)
ep = c.episode_idx.numpy()
N = len(z)
rng = np.random.default_rng(0)

ok = np.zeros(N, bool)
ok[:-50] = ep[:-50] == ep[50:]          # t .. t+50 same episode
rows = rng.choice(np.flatnonzero(ok), size=20000, replace=False)
t = torch.as_tensor(rows, device=dev)

def d_of(metric, a, b, D):
    x, g = z[a], z[b]
    if D == 2:
        # delta metric convention (not used for these single-frame probes)
        x = torch.cat([x, torch.zeros_like(x)], -1); g = torch.cat([g, torch.zeros_like(g)], -1)
    return metric(x, g)

for path in sys.argv[1:]:
    m = load_metric(path, device=dev)
    with torch.no_grad():
        d_25 = m(z[t], z[t + 25])
        d_50 = m(z[t], z[t + 50])
        d_25_50 = m(z[t + 25], z[t + 50])
        d_12_25 = m(z[t + 12], z[t + 25])
        p1 = (d_50 > d_25).float().mean().item()
        p2 = (d_25_50 < d_50).float().mean().item()
        p3 = (d_12_25 < d_25).float().mean().item()
        name = path.split("/")[-1]
        print(f"{name}")
        print(f"  P1 range-order(25 vs 50): {100*p1:5.1f}%   "
              f"P2 progress@50: {100*p2:5.1f}%   P3 progress@25: {100*p3:5.1f}%")
        cal = []
        for lo, hi in [(1, 10), (10, 25), (25, 40), (40, 60), (60, 90)]:
            gaps = rng.integers(lo, hi, size=len(rows))
            ok2 = np.zeros(N, bool)
            ok2[:-90] = ep[:-90] == ep[90:]
            r2 = rng.choice(np.flatnonzero(ok2), size=8000, replace=False)
            a = torch.as_tensor(r2, device=dev)
            b = torch.as_tensor(r2 + gaps[:8000], device=dev)
            cal.append(f"{lo}-{hi}: {m(z[a], z[b]).mean().item():5.1f}")
        print("  mean d by true gap: " + "  ".join(cal), flush=True)
