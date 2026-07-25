"""Velocity-aliasing probe: is agent velocity decodable from z alone vs [z, Δz]?

Δz = z_t − z_{t−5} (block-scale finite difference, matching what the planner's
imagined trajectory can provide at eval time).
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from stable_worldmodel.trm import LatentCache

c = LatentCache.load("/workspace/caches/pusht_official_fs1.pt")
z = c.z.float().numpy()
s = c.state.float().numpy()
ep = c.episode_idx.numpy()

# block-scale delta within episode
dz = np.zeros_like(z)
same = ep[5:] == ep[:-5]
dz[5:][same] = z[5:][same] - z[:-5][same]

rng = np.random.default_rng(0)
idx = rng.permutation(len(z))
tr, te = idx[:400_000], idx[400_000:500_000]

vel = s[:, 5:7]
for name, X in [("z only", z), ("dz only", dz), ("[z, dz]", np.concatenate([z, dz], 1))]:
    r = Ridge(alpha=1.0).fit(X[tr], vel[tr])
    print(f"  {name:10s} vel R2 = {r2_score(vel[te], r.predict(X[te])):.3f}", flush=True)

# also block pose for reference
pose = s[:, 2:5]
r = Ridge(alpha=1.0).fit(z[tr], pose[tr])
print(f"  (ref) z only block-pose R2 = {r2_score(pose[te], r.predict(z[te])):.3f}")
