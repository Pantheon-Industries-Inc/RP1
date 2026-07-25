"""Row-range sharding wrapper around the repo's encode_dataset — one shard per
GPU, merge with merge_caches.py. Reuses the exact featurizer (build_featurizer)
so shards are numerically identical to a single-GPU cache_latents.py run.

Rows are (episode, step) flattened; shard on episode boundaries:
row_start = ep_start * steps_per_ep (mixture: 201).

Usage (4-way over the 10k-ep mixture):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python3 cache_lance_shard.py --wm <dir> \
      --dataset mix.lance --row-start $((i*2500*201)) --row-end $(((i+1)*2500*201)) \
      --state-key privileged/block_0_pos --out shard$i.pt &
  done; wait
  python3 merge_caches.py fs1.pt fs5.pt shard0.pt shard1.pt shard2.pt shard3.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/code/stable-worldmodel/scripts/trm")

from _common import build_featurizer, load_wm, pick_device  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.trm import encode_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--state-key", default=None)
    p.add_argument("--row-start", type=int, required=True)
    p.add_argument("--row-end", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = pick_device(a.device)
    wm = load_wm(a.wm, device=device)
    featurizer = build_featurizer(wm, device=device, train_res=None)
    full = swm.data.load_dataset(a.dataset)
    s, e = a.row_start, a.row_end

    class _Slice:
        column_names = full.column_names

        def get_col_data(self, c):
            return full.get_col_data(c)[s:e]

        def get_row_data(self, i):
            # encode_dataset always passes a LIST of local row indices
            # (latent_cache.py:124-125), so the old `s + i` was int + list ->
            # TypeError on the very first batch. Offset element-wise instead.
            if isinstance(i, (list, tuple)):
                return full.get_row_data([s + j for j in i])
            return full.get_row_data(s + i)

    cache = encode_dataset(
        _Slice(), featurizer, batch_size=a.batch_size,
        state_key=a.state_key or None,
        meta={"wm": a.wm, "dataset": a.dataset, "rows": [s, e]},
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(a.out)
    print(f"shard rows[{s}:{e}] -> {a.out} ({len(cache.z)} latents)")


if __name__ == "__main__":
    main()
