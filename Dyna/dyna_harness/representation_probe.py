"""Is the mid-range ordering defect created by the VALUE, or inherited from the
WM's LATENT GEOMETRY (and/or the task itself)?

Three curves vs true temporal separation k, on the same anchors:
  A. raw latent  ||z_t - z_{t+k}||_2          (no learning at all)
  B. raw latent  cosine distance
  C. ground-truth state ||block_pos_t - block_pos_{t+k}||_2  (from cache.state)

If A/B dip at k=75-100 like the learned value does, the defect is in the
representation and NO value-side fix can repair it. If C also dips, the block
genuinely revisits goal-like positions mid-episode (task structure / perceptual
aliasing) and the only cure is subgoal decomposition or a better encoder.

Usage: representation_probe.py FS1_CACHE
"""
import sys
from collections import defaultdict

import numpy as np
import torch

from stable_worldmodel.trm import LatentCache

cache = LatentCache.load(sys.argv[1])
z = cache.z.float()
epi = np.asarray(cache.episode_idx).reshape(-1).astype(np.int64)
step = np.asarray(cache.step_idx).reshape(-1).astype(np.int64)
state = cache.state.float() if cache.state is not None else None
print(f"latent dim {z.shape[1]}; state {'present ' + str(tuple(state.shape)) if state is not None else 'ABSENT'}")

rows = defaultdict(list)
for i in range(len(epi)):
    rows[int(epi[i])].append(i)
for e in rows:
    rows[e] = sorted(rows[e], key=lambda r: step[r])
eps = list(rows.keys())
rng = np.random.default_rng(0)

print(f"\n{'k':>5} {'L2 latent':>11} {'cos latent':>11} {'L2 block-pos':>13}")
for k in [5, 10, 25, 50, 75, 100, 150, 200]:
    ai, bi = [], []
    tries = 0
    while len(ai) < 4000 and tries < 80000:
        tries += 1
        e = eps[rng.integers(len(eps))]
        r = rows[e]
        if len(r) <= k:
            continue
        t = rng.integers(0, len(r) - k)
        ai.append(r[t]); bi.append(r[t + k])
    if not ai:
        print(f"{k:>5}   n/a")
        continue
    za, zb = z[ai], z[bi]
    l2 = (za - zb).norm(dim=-1).mean().item()
    cos = (1 - torch.nn.functional.cosine_similarity(za, zb, dim=-1)).mean().item()
    if state is not None:
        st = (state[ai] - state[bi]).norm(dim=-1).mean().item()
        print(f"{k:>5} {l2:>11.4f} {cos:>11.4f} {st:>13.4f}")
    else:
        print(f"{k:>5} {l2:>11.4f} {cos:>11.4f} {'-':>13}")
