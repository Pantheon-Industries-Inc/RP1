"""Horizon-match a latent cache to the planner's frameskip.

The planner runs at ``action_block = frameskip`` (LeWM: 5), so each planned step
spans `frameskip` env steps. To train the terminal metric on the *same* horizon,
subsample the per-frame (fs1) cache to every `frameskip`-th frame within each
episode and re-index step_idx in planner-step units. Latents are unchanged; only
the temporal indexing (hence the metric's Δ distribution) is matched.

Example::

    python scripts/trm/subsample_cache.py --in caches/lewm_tworoom.pt \
        --out caches/lewm_fs5.pt --frameskip 5
"""

import argparse

import numpy as np
import torch

from stable_worldmodel.trm import LatentCache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frameskip", type=int, default=5)
    args = p.parse_args()

    c = LatentCache.load(args.inp)
    fs = args.frameskip
    zs, eps, sts, states = [], [], [], []
    for e, rows in c.episodes().items():          # rows sorted by step_idx
        sub = rows[::fs]                            # every fs-th frame
        if len(sub) < 2:
            continue
        zs.append(c.z[sub])
        eps.append(np.full(len(sub), e, np.int64))
        sts.append(np.arange(len(sub), dtype=np.int64))   # re-index in planner-steps
        if c.state is not None:
            states.append(c.state[sub])
    out = LatentCache(
        z=torch.cat(zs),
        episode_idx=torch.from_numpy(np.concatenate(eps)),
        step_idx=torch.from_numpy(np.concatenate(sts)),
        state=(torch.cat(states) if states else None),
        meta={**(c.meta or {}), "frameskip": fs, "horizon_matched": True},
    )
    lens = [len(r) for r in out.episodes().values()]
    print(f"fs{fs} cache: {len(out.z)} latents, {len(lens)} episodes, "
          f"len[min/med/max]={min(lens)}/{int(np.median(lens))}/{max(lens)}")
    out.save(args.out)


if __name__ == "__main__":
    main()
