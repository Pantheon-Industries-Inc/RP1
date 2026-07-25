"""Two corrections to the earlier diagnostics:

TEST B' — PLANNER-RELEVANT ranking (fixed GOAL, varying candidate state).
  Earlier probes fixed the anchor and varied the goal; the planner does the
  opposite. For a goal g = z_{t+K} and two states en route z_{t+j1}, z_{t+j2}
  (j1 < j2, so j2 is CLOSER to g), a usable value must satisfy
  V(z_{t+j2}, g) < V(z_{t+j1}, g). Measured on the deployed value.

TEST C — RANKING CEILING. The decodability probe was trained with REGRESSION
  (smooth-L1 on k), which collapses to the mean under ambiguity. Here the same
  network is trained with a pairwise MARGIN RANKING loss on the planner-relevant
  comparison. If ranking-trained accuracy >> regression-trained accuracy, then a
  ranking loss on the value is worth building; if it saturates near chance, the
  ordering genuinely isn't recoverable and subgoals are the only route.

Usage: ranking_ceiling_probe.py FS1_CACHE VALUE_PT
"""
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from stable_worldmodel.trm import LatentCache, load_metric

cache = LatentCache.load(sys.argv[1])
net_v = load_metric(sys.argv[2], device="cuda"); net_v.eval()
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
    ai, gi = torch.tensor(np.asarray(ai)), torch.tensor(np.asarray(gi))
    for s in range(0, len(ai), 4096):
        with torch.no_grad():
            out.append(net_v(z[ai[s:s+4096]].cuda(), z[gi[s:s+4096]].cuda()).squeeze(-1).float().cpu())
    return torch.cat(out)


def sample_planner_pairs(n, dist_near, dist_far, ep_pool):
    """states at true distance dist_near / dist_far from the SAME goal."""
    near, far, goal = [], [], []
    tries = 0
    while len(near) < n and tries < n * 30:
        tries += 1
        e = ep_pool[rng.integers(len(ep_pool))]
        r = rows[e]
        if len(r) <= dist_far + 1:
            continue
        gi = int(rng.integers(dist_far, len(r)))          # goal index
        near.append(r[gi - dist_near]); far.append(r[gi - dist_far]); goal.append(r[gi])
    return np.array(near), np.array(far), np.array(goal)


print("TEST B' — PLANNER-RELEVANT ranking of the DEPLOYED value")
print("  (fixed goal; is the state truly closer to g scored lower?)  chance = 50%")
for dn, df in [(5, 10), (25, 50), (50, 75), (75, 100), (100, 150), (150, 200), (25, 100)]:
    nr, fr, gl = sample_planner_pairs(3000, dn, df, eps)
    if len(nr) < 100:
        print(f"  {dn:>3} vs {df:>3} : n/a"); continue
    acc = (V(nr, gl) < V(fr, gl)).float().mean().item() * 100
    print(f"  closer={dn:>3} vs farther={df:>3} : {acc:5.1f}%")

print("\nTEST C — RANKING CEILING (same net, ranking loss vs regression loss)")
split = int(0.9 * len(eps)); tr_eps, te_eps = eps[:split], eps[split:]
D = z.shape[1]


def feats(a, g):
    za, zg = z[a], z[g]
    return torch.cat([za, zg, za - zg], 1)


for dn, df in [(25, 50), (50, 75), (75, 100)]:
    ntr, ftr, gtr = sample_planner_pairs(60000, dn, df, tr_eps)
    nte, fte, gte = sample_planner_pairs(8000, dn, df, te_eps)
    if len(ntr) < 1000 or len(nte) < 200:
        print(f"  {dn} vs {df}: insufficient samples"); continue
    Xn, Xf = feats(ntr, gtr).cuda(), feats(ftr, gtr).cuda()
    Xn_t, Xf_t = feats(nte, gte).cuda(), feats(fte, gte).cuda()
    mu, sd = Xn.mean(0, keepdim=True), Xn.std(0, keepdim=True) + 1e-6
    Xn, Xf, Xn_t, Xf_t = (Xn - mu) / sd, (Xf - mu) / sd, (Xn_t - mu) / sd, (Xf_t - mu) / sd
    net = nn.Sequential(nn.Linear(3 * D, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for it in range(4000):
        idx = torch.randint(0, len(Xn), (1024,), device="cuda")
        sn, sf = net(Xn[idx]).squeeze(-1), net(Xf[idx]).squeeze(-1)
        loss = nn.functional.relu(1.0 + sn - sf).mean()      # margin ranking: sn < sf
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        acc = (net(Xn_t).squeeze(-1) < net(Xf_t).squeeze(-1)).float().mean().item() * 100
    print(f"  closer={dn:>3} vs farther={df:>3} : ranking-trained ceiling {acc:5.1f}%")
