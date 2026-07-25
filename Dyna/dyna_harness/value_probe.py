"""Diagnose WHERE LIP fails at long horizon: is the value V(z, z_goal) usable
as a planning signal for far goals?

Deploy-relevant probe: fix a goal K steps ahead in an expert trajectory, then
measure V(z_{t+j}, z_{t+K}) as the agent approaches (j = 0..K). A well-calibrated
value should rise monotonically toward the goal, giving the planner a gradient
from the start. If V is FLAT until j is nearly K, the value is blind to the far
goal -> no gradient -> receding-horizon planner wanders (explains h200 collapse).

Compares K in {25,50,100,200}: h25 works, so its curve is the "good" reference.

Usage: value_probe.py FS1_CACHE VALUE_PT
"""
import sys
import numpy as np
import torch
from collections import defaultdict

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
    rows[e] = [r for r in sorted(rows[e], key=lambda r: step[r])]
eps = list(rows.keys())
rng = np.random.default_rng(0)


def Vb(ai, gi):
    out = []
    ai = torch.tensor(ai); gi = torch.tensor(gi)
    for s in range(0, len(ai), 4096):
        with torch.no_grad():
            v = net(z[ai[s:s+4096]].cuda(), z[gi[s:s+4096]].cuda()).squeeze(-1)
        out.append(v.float().cpu())
    return torch.cat(out)


print("Approach curve: V(z_{t+j}, z_{t+K}) vs progress j/K (mean over ~2500 anchors)")
print(f"{'K':>4} | " + "  ".join(f"j/K={f:.2f}" for f in [0, .25, .5, .75, 1.0])
      + " | monotonic%")
for K in [25, 50, 100, 200]:
    anchors = []
    while len(anchors) < 2500 and len(anchors) < 10 * len(eps):
        e = eps[rng.integers(len(eps))]
        r = rows[e]
        if len(r) <= K:
            continue
        t = rng.integers(0, len(r) - K)
        anchors.append((r, t))
    fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
    cols = []
    curves = []  # per-anchor V across fracs, for monotonicity
    per_frac_idx = {f: ([], []) for f in fracs}
    for r, t in anchors:
        for f in fracs:
            j = int(round(f * K))
            per_frac_idx[f][0].append(r[t + j])
            per_frac_idx[f][1].append(r[t + K])
    meanvals = {}
    allv = {}
    for f in fracs:
        v = Vb(per_frac_idx[f][0], per_frac_idx[f][1])
        meanvals[f] = float(v.mean())
        allv[f] = v
    stack = torch.stack([allv[f] for f in fracs], 1)  # (n, 5)
    mono = (stack[:, 1:] >= stack[:, :-1] - 1e-4).all(1).float().mean().item() * 100
    print(f"{K:>4} | " + "  ".join(f"{meanvals[f]:7.3f}" for f in fracs)
          + f" |  {mono:5.1f}%")

# also raw distance calibration: V(z_t, z_{t+k}) vs k (does value track distance)
print("\nDistance calibration: mean V(z_t, z_{t+k}) vs k")
print(f"{'k':>4}  {'meanV':>8}")
for k in [5, 10, 25, 50, 75, 100, 150, 200]:
    ai, gi = [], []
    tries = 0
    while len(ai) < 3000 and tries < 40000:
        tries += 1
        e = eps[rng.integers(len(eps))]
        r = rows[e]
        if len(r) <= k:
            continue
        t = rng.integers(0, len(r) - k)
        ai.append(r[t]); gi.append(r[t + k])
    v = Vb(ai, gi)
    print(f"{k:>4}  {float(v.mean()):8.3f}")
