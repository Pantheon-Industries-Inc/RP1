"""Row-aligned logged-state cache for success-tolerance relabeling.

Extracts the environment state column of a logged h5 dataset for the train
split (episode prefix) into a :class:`LatentCache` whose ``z`` IS the state,
with the same ``episode_idx`` / ``step_idx`` bookkeeping as the latent caches
built from the same rows, so the TD samplers can index both together.

Example::

    pixi run tool tool=cache_state dataset=$H5 out=caches/counterstrike_state.pt max_episodes=16000
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.data import LatentCache
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    with h5py.File(str(cfg.dataset), "r") as h:
        episode_idx = np.asarray(h["episode_idx"][:]).reshape(-1).astype(np.int64)
        step_idx = np.asarray(h["step_idx"][:]).reshape(-1).astype(np.int64)
        state = np.asarray(h[str(cfg.state_key)][:], dtype=np.float32)
    n = len(episode_idx)
    if cfg.max_episodes is not None:
        in_split = int(np.count_nonzero(episode_idx < int(cfg.max_episodes)))
        if not np.array_equal(np.flatnonzero(episode_idx < int(cfg.max_episodes)), np.arange(in_split)):
            raise ValueError("episodes are not stored contiguously; cannot apply max_episodes as a prefix")
        n = in_split
    state = state[:n].reshape(n, -1)
    cache = LatentCache(
        z=torch.from_numpy(state),
        episode_idx=torch.from_numpy(episode_idx[:n]),
        step_idx=torch.from_numpy(step_idx[:n]),
        state=None,
        meta={"dataset": str(cfg.dataset), "state_key": str(cfg.state_key), "kind": "state"},
    )
    Path(str(cfg.out)).parent.mkdir(parents=True, exist_ok=True)
    cache.save(str(cfg.out))
    logger.success(f"Cached {n} state rows (dim={state.shape[1]}) at {cfg.out}")


def main() -> object:
    return run_hydra(dispatch, config_name="tools/cache_state")


if __name__ == "__main__":
    main()
