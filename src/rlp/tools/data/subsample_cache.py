"""Horizon-match a latent cache to the planner's frameskip.

The planner runs at ``action_block = frameskip`` (LeWM: 5), so each planned step
spans `frameskip` env steps. To train the terminal metric on the *same* horizon,
subsample the per-frame (fs1) cache to every `frameskip`-th frame within each
episode and re-index step_idx in planner-step units. Latents are unchanged; only
the temporal indexing (hence the metric's Δ distribution) is matched.

Example::

    pixi run tool tool=subsample_cache \
        inp=caches/lewm_tworoom.pt out=caches/lewm_fs5.pt frameskip=5
"""

import numpy as np
import torch
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.data import LatentCache
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    args = cfg

    c = LatentCache.load(args.inp)
    fs = args.frameskip
    zs, eps, sts, states = [], [], [], []
    for e, rows in c.episodes().items():  # rows sorted by step_idx
        sub = rows[::fs]  # every fs-th frame
        if len(sub) < 2:
            continue
        zs.append(c.z[sub])
        eps.append(np.full(len(sub), e, np.int64))
        sts.append(np.arange(len(sub), dtype=np.int64))  # re-index in planner-steps
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
    logger.info(
        f"fs{fs} cache: {len(out.z)} latents, {len(lens)} episodes, "
        f"len[min/med/max]={min(lens)}/{int(np.median(lens))}/{max(lens)}"
    )
    out.save(args.out)


def main() -> object:
    return run_hydra(dispatch, config_name="tools/subsample_cache")


if __name__ == "__main__":
    main()
