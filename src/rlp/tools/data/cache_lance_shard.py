"""Row-range sharding wrapper around the repo's encode_dataset — one shard per
GPU, merge with merge_caches.py. Reuses the exact featurizer (build_featurizer)
so shards are numerically identical to a single-GPU cache_latents.py run.

Rows are (episode, step) flattened; shard on episode boundaries:
row_start = ep_start * steps_per_ep (mixture: 201).

Usage (4-way over the 10k-ep mixture):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i pixi run tool tool=cache_lance_shard wm=<dir> \
      dataset=mix.lance row_start=<start> row_end=<end> \
      state_key=privileged/block_0_pos out=<shard.pt>
  done; wait
  python3 merge_caches.py fs1.pt fs5.pt shard0.pt shard1.pt shard2.pt shard3.pt
"""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.core.world_model.runtime import build_featurizer, load_wm, pick_device
from rlp.data import encode_dataset
from rlp.data.protocols import Array, RowBatch
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    a = cfg

    device = pick_device(a.device)
    wm = load_wm(a.wm, device=device)
    featurizer = build_featurizer(wm, device=device, train_res=None)
    full = swm.data.load_dataset(a.dataset)
    s, e = a.row_start, a.row_end

    class _Slice:
        column_names = full.column_names

        def get_col_data(self, c: str) -> Array:
            return np.asarray(full.get_col_data(c)[s:e])

        def get_row_data(self, i: list[int]) -> RowBatch:
            # encode_dataset always passes a LIST of local row indices
            # (rlp/data/latent_cache.py:125), so the old `s + i` was int + list ->
            # TypeError on the very first batch. Offset element-wise instead.
            rows = full.get_row_data([s + j for j in i])
            return {str(key): np.asarray(value) for key, value in rows.items()}

    cache = encode_dataset(
        _Slice(),
        featurizer,
        batch_size=a.batch_size,
        state_key=a.state_key or None,
        meta={"wm": a.wm, "dataset": a.dataset, "rows": [s, e]},
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(a.out)
    logger.success(f"Cached shard rows=[{s}:{e}] latents={len(cache.z)} path={a.out}")


def main() -> object:
    return run_hydra(dispatch, config_name="tools/data/cache_lance_shard")


if __name__ == "__main__":
    main()
