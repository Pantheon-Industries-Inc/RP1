"""Does the value preserve TEMPORAL ORDERING, and is the per-step progress
signal detectable at long range?

TEST A — pairwise ranking accuracy (ordering fidelity):
  for anchor z_t and two futures z_{t+a}, z_{t+b} with a < b, is
  V(z_t, z_{t+a}) < V(z_t, z_{t+b})?  Chance = 50%. Bucketed by distance.

TEST B — progress SNR (what the planner actually needs):
  signal = V(z_t, z_g) - V(z_{t+5}, z_g)   (value drop from 5 steps of progress)
  noise  = std of V(z_·, z_g) over states at the SAME true distance from g
  SNR = signal / noise. Planner can only discriminate good vs bad candidate
  actions if SNR >> 0. Computed as a function of goal distance K.

Usage: ordering_probe.py FS1_CACHE VALUE_PT
"""
import sys
from collections import defaultdict

import numpy as np
import torch

from stable_worldmodel.trm import LatentCache, load_metric

cache = LatentCache.load(sys.argv[1])
net = load_metric(sys.argv[2], device="cuda")
net.eval()
z = cache.z.float()
epi = np.asarray(cache.episode_idx).reshape(-1).astype(np.int64)
step = np.asarray(cache.step_idx).reshape(-1).astype(np.int64)

rows = defaultdict(list)
for i in range(len(epi)):
    rows[int(epi[i])].append(i)
for e in rows:
    rows[e] = sorted(rows[e], key=lambda r: step[r])
eps = list(rows.keys())
rng = np.random.default_rng(0)


def V(ai, gi):
    out = []
    ai, gi = torch.tensor(ai), torch.tensor(gi)
    for s in range(0, len(ai), 4096):
        with torch.no_grad():
            out.append(net(z[ai[s:s+4096]].cuda(), z[gi[s:s+4096]].cuda()).squeeze(-1).float().cpu())
    return torch.cat(out)


print("TEST A — pairwise ranking accuracy (chance=50%)")
print(f"{'a vs b (steps to goal)':>26} | {'acc':>6}")
for a, b in [(5, 10), (25, 50), (50, 75), (75, 100), (100, 150), (150, 200), (50, 200)]:
    ai, bi, ti = [], [], []
    tries = 0
    while len(ti) < 3000 and tries < 60000:
        tries += 1
        e = eps[rng.integers(len(eps))]
        r = rows[e]
        if len(r) <= b:
            continue
        t = rng.integers(0, len(r) - b)
        ti.append(r[t]); ai.append(r[t + a]); bi.append(r[t + b])
    if not ti:
        print(f"{f'{a} vs {b}':>26} |  n/a")
        continue
    va, vb = V(ti, ai), V(ti, bi)
    acc = float((va < vb).float().mean()) * 100
    print(f"{f'{a} vs {b}':>26} | {acc:5.1f}%")

print("\nTEST B — progress SNR (signal = V drop from 5 steps of real progress)")
print(f"{'K (goal dist)':>14} | {'signal':>8} {'noise':>8} {'SNR':>7}")
for K in [25, 50, 100, 150, 200]:
    t_i, t5_i, g_i = [], [], []
    tries = 0
    while len(t_i) < 3000 and tries < 60000:
        tries += 1
        e = eps[rng.integers(len(eps))]
        r = rows[e]
        if len(r) <= K + 5:
            continue
        t = rng.integers(0, len(r) - K - 5)
        t_i.append(r[t]); t5_i.append(r[t + 5]); g_i.append(r[t + K])
    if not t_i:
        print(f"{K:>14} | n/a")
        continue
    v0, v5 = V(t_i, g_i), V(t5_i, g_i)
    signal = float((v0 - v5).mean())          # expected ~5 if calibrated in steps
    noise = float(v0.std())                    # spread across states at same true dist
    print(f"{K:>14} | {signal:8.3f} {noise:8.3f} {signal/max(noise,1e-6):7.3f}")
