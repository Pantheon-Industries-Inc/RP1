"""Is temporal separation DECODABLE from the latent pair at all?

Trains a supervised MLP  f(z_t, z_g) -> k  on ground-truth k (no TD, no
bootstrapping, no quasimetric constraint — the easiest possible version of the
problem). If this probe can't recover k in the 50-100 band, the information is
genuinely absent from the representation and no value learner can succeed.
Control: the same probe on ground-truth block positions.

Reports MAE and Spearman-style pairwise ranking accuracy per band, plus a
"chance" reference (predicting the mean k).

Usage: decodability_probe.py FS1_CACHE
"""
import sys
from collections import defaultdict

import numpy as np
import torch
from torch import nn

cache_path = sys.argv[1]
from stable_worldmodel.trm import LatentCache

cache = LatentCache.load(cache_path)
z = cache.z.float()
state = cache.state.float() if cache.state is not None else None
epi = np.asarray(cache.episode_idx).reshape(-1).astype(np.int64)
step = np.asarray(cache.step_idx).reshape(-1).astype(np.int64)

rows = defaultdict(list)
for i in range(len(epi)):
    rows[int(epi[i])].append(i)
for e in rows:
    rows[e] = sorted(rows[e], key=lambda r: step[r])
eps = list(rows.keys())
rng = np.random.default_rng(0)
KMAX = 200

def make_pairs(n, ep_pool):
    a, b, ks = [], [], []
    tries = 0
    while len(a) < n and tries < n * 25:
        tries += 1
        e = ep_pool[rng.integers(len(ep_pool))]
        r = rows[e]
        k = int(rng.integers(1, min(KMAX, len(r) - 1) + 1))
        if len(r) <= k:
            continue
        t = int(rng.integers(0, len(r) - k))
        a.append(r[t]); b.append(r[t + k]); ks.append(k)
    return np.array(a), np.array(b), np.array(ks, dtype=np.float32)

split = int(0.9 * len(eps))
tr_eps, te_eps = eps[:split], eps[split:]
ai, bi, ktr = make_pairs(120000, tr_eps)
aj, bj, kte = make_pairs(20000, te_eps)

def run(feat_name, src):
    D = src.shape[1]
    Xtr = torch.cat([src[ai], src[ai]-src[np.maximum(ai-1,0)], src[np.maximum(ai-1,0)]-src[np.maximum(ai-2,0)], src[bi], src[ai] - src[bi]], 1)
    Xte = torch.cat([src[aj], src[aj]-src[np.maximum(aj-1,0)], src[np.maximum(aj-1,0)]-src[np.maximum(aj-2,0)], src[bj], src[aj] - src[bj]], 1)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr, yte = torch.tensor(ktr), torch.tensor(kte)
    net = nn.Sequential(nn.Linear(5 * D, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1)).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr = Xtr.cuda(), ytr.cuda()
    for it in range(3000):
        idx = torch.randint(0, len(Xtr), (1024,), device="cuda")
        loss = nn.functional.smooth_l1_loss(net(Xtr[idx]).squeeze(-1), ytr[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte.cuda()).squeeze(-1).cpu()
    print(f"\n--- {feat_name} ---")
    print(f"  overall MAE {(pred - yte).abs().mean():.1f} steps "
          f"(chance = predict-mean: {(yte - yte.mean()).abs().mean():.1f})")
    for lo, hi in [(1, 25), (25, 50), (50, 75), (75, 100), (100, 150), (150, 200)]:
        m = (yte >= lo) & (yte < hi)
        if m.sum() < 50:
            continue
        print(f"  k in [{lo:>3},{hi:>3}): MAE {(pred[m] - yte[m]).abs().mean():6.1f}  "
              f"pred mean {pred[m].mean():6.1f} vs true {yte[m].mean():6.1f}")
    # pairwise ranking accuracy of the DECODER in the problem band
    for lo, hi in [(50, 75), (75, 100), (25, 50)]:
        m1 = (yte >= lo) & (yte < lo + 10)
        m2 = (yte >= hi) & (yte < hi + 10)
        n = min(m1.sum(), m2.sum(), 2000)
        if n < 50:
            continue
        p1, p2 = pred[m1][:n], pred[m2][:n]
        print(f"  rank acc {lo}±5 vs {hi}±5: {(p1 < p2).float().mean() * 100:5.1f}%")

run("LATENT + 3-FRAME HISTORY (velocity/phase)", z)
if state is not None:
    run("GROUND-TRUTH block pos (3-d, control)", state)
