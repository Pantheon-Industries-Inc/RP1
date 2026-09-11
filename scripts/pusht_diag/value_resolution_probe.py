"""Does V resolve position INSIDE one primitive step, and what controls that?

E22 put four of five PushT planner failures at 20-30 px against a 20 px
threshold with V in [1.4, 4.4], which suggested V is a staircase in integer
steps-to-go and so has no gradient across the final step. This walks a query
from frame t to frame t+1 and measures how far V actually moves, for the
shipped expectile and a neutral one, with and without sub-step training
queries (TDConfig.subgrid).

Env: CACHE (latent cache path), STEPS (TD steps per fit, default 1000).
"""

import os, numpy as np, torch
from rlp.data import LatentCache
from rlp.core.value.learners.td import TDConfig, fit

cache = LatentCache.load(os.environ.get("CACHE", os.path.expanduser("~/.cache/rlp/caches/tworoom_mixed_fs1.pt")), mmap=True)
rng = np.random.default_rng(0); eps = cache.episodes()
ids = [e for e in eps if len(eps[e]) > 40]
P = []
for e in rng.choice(ids, 200, replace=False):
    rows = eps[int(e)]; t = int(rng.integers(0, len(rows) - 6))
    P.append((rows[t], rows[t + 1], rows[t + int(rng.integers(1, 4))]))
t0 = torch.stack([cache.z[p[0]] for p in P]).float()
t1 = torch.stack([cache.z[p[1]] for p in P]).float()
gg = torch.stack([cache.z[p[2]] for p in P]).float()

step_sep = (t1 - t0).norm(dim=-1)
goal_sep = (gg - t0).norm(dim=-1)
print(f"PREMISE: ||z_t - z_t+1|| = {step_sep.mean():.3f}; ||z_t - z_goal|| (1-3 steps) = {goal_sep.mean():.3f}; "
      f"ratio = {(step_sep / goal_sep.clamp_min(1e-6)).mean():.3f}\n")

base = dict(head="quasimetric", n_step=1, gamma=0.98, batch_size=256,
            steps=int(os.environ.get("STEPS", "1000")), seed=0, near_frac=0.3, near_max=3)
alphas = torch.linspace(0, 1, 9)
for tau in (0.03, 0.5):
    for sg in (0.0, 0.5):
        v = fit(cache, TDConfig(expectile=tau, subgrid=sg, **base), "cpu").eval()
        with torch.no_grad():
            curve = torch.stack([v(t0 + a * (t1 - t0), gg) for a in alphas])
        drop = (curve[0] - curve[-1]).mean()
        mono = ((curve[1:] - curve[:-1]) <= 1e-6).all(0).float().mean()
        print(f"tau={tau:<5} subgrid={sg}: V(t)={curve[0].mean():6.3f} "
              f"drop across step {drop:+.3f} (want ~+1.0)  monotone {100*mono:3.0f}%")
