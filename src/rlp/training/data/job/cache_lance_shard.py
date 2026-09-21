"""Row-range sharding wrapper around the repo's encode_dataset — one shard per
GPU, merge with merge_caches.py. Reuses the exact featurizer (build_featurizer)
so shards are numerically identical to a single-GPU cache_latents.py run.

Rows are (episode, step) flattened; shard on episode boundaries:
row_start = ep_start * steps_per_ep (mixture: 201).

Usage (4-way over the 10k-ep mixture):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i pixi run prepare job=cache_lance_shard preparation.wm=<dir> \
      preparation.dataset=mix.lance row_start=<start> row_end=<end> \
      preparation.state_key=privileged/block_0_pos preparation.out=<shard.pt>
  done; wait
  python3 merge_caches.py fs1.pt fs5.pt shard0.pt shard1.pt shard2.pt shard3.pt
"""

from pathlib import Path
from typing import cast

import stable_worldmodel as swm
from omegaconf import DictConfig

from rlp.core.world_model.featurize import build_featurizer
from rlp.data import encode_dataset
from rlp.data.base import Dataset, RowRange
from rlp.training.harness.checkpointing import load_wm
from rlp.utils.config import dispatch, phase_config, run_hydra
from rlp.utils.device import pick_device
from rlp.utils.logging import logger


def _run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "preparation")

    device = pick_device(a.device)
    wm = load_wm(a.wm, device=device)
    featurizer = build_featurizer(wm, device=device, img_size=a.image_size, train_res=None)
    full = swm.data.load_dataset(a.dataset)
    s, e = a.row_start, a.row_end

    cache = encode_dataset(
        RowRange(cast(Dataset, full), s, e),
        featurizer,
        batch_size=a.batch_size,
        state_key=a.state_key or None,
        meta={"wm": a.wm, "dataset": a.dataset, "rows": [s, e]},
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(a.out)
    logger.success(f"Cached shard rows=[{s}:{e}] latents={len(cache.z)} path={a.out}")


def main() -> object:
    return run_hydra(dispatch, config_name="training/data/job/cache_lance_shard")


if __name__ == "__main__":
    main()
