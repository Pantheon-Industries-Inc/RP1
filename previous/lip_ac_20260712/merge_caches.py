"""Merge LatentCache shards (episode-range shards from cache_cube_full.py) and
derive the fs5 cache from the merged fs1 (identical transform, so subsampling
every 5th frame == encoding with stride 5).

Usage: merge_caches.py OUT_FS1 OUT_FS5 SHARD [SHARD ...]
"""
import sys

import torch

from stable_worldmodel.trm import LatentCache

out1, out5, shards = sys.argv[1], sys.argv[2], sys.argv[3:]
cs = [LatentCache.load(s) for s in shards]
cs.sort(key=lambda c: int(c.episode_idx.min()))
ep = torch.cat([c.episode_idx for c in cs])
assert (torch.diff(ep) >= 0).all(), "shards overlap or are out of order"

merged = LatentCache(
    z=torch.cat([c.z for c in cs]),
    episode_idx=ep,
    step_idx=torch.cat([c.step_idx for c in cs]),
    state=(torch.cat([c.state for c in cs]) if cs[0].state is not None else None),
    meta={**(cs[0].meta or {}), "ep_range": [int(ep.min()), int(ep.max()) + 1]},
)
merged.save(out1)

mask = merged.step_idx % 5 == 0
fs5 = LatentCache(
    z=merged.z[mask],
    episode_idx=merged.episode_idx[mask],
    step_idx=merged.step_idx[mask] // 5,
    state=(merged.state[mask] if merged.state is not None else None),
    meta={**(merged.meta or {}), "stride": 5, "derived_from": out1},
)
fs5.save(out5)
print(f"fs1 {len(merged.z)} rows -> {out1}; fs5 {len(fs5.z)} rows -> {out5}")
