"""Encode a logged dataset into a frozen-latent cache for metric training.

The expensive encoder runs once here; the three metric learners then train on
the cheap cached latents.

Example (state WM)::

    python scripts/trm/cache_latents.py --wm statewm_tworoom \
        --dataset tworoom_expert.lance --out caches/tworoom_state.pt

Example (pixel LeWM)::

    python scripts/trm/cache_latents.py --wm quentinll/lewm-cube \
        --dataset ogbench/cube_single_multiview_expert.lance \
        --out caches/cube_lewm.pt --state-key proprio
"""

import argparse
from pathlib import Path

from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.trm import encode_dataset

from _common import build_featurizer, is_statewm, load_wm, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True, help="world-model checkpoint name/path")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, help="output cache .pt path")
    p.add_argument("--state-key", default=None, help="ground-truth state column for the oracle")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-rows", type=int, default=None, help="cap rows (debug)")
    p.add_argument("--device", default="auto")
    p.add_argument("--train-res", type=int, default=None,
                   help="checkpoint's native training image resolution; frames are "
                        "bottlenecked through it (e.g. 64 for OGBench play retrains). "
                        "Must match eval.train_res at plan time.")
    args = p.parse_args()

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    featurizer = build_featurizer(wm, device=device, train_res=args.train_res)

    dataset = swm.data.load_dataset(args.dataset)
    state_key = args.state_key or (wm.obs_key if is_statewm(wm) else None)

    if args.max_rows is not None:
        full = dataset

        class _Sub:
            column_names = full.column_names

            def get_col_data(self, c):
                return full.get_col_data(c)[: args.max_rows]

            def get_row_data(self, i):
                return full.get_row_data(i)

        dataset = _Sub()

    cache = encode_dataset(
        dataset, featurizer, batch_size=args.batch_size, state_key=state_key,
        meta={"wm": args.wm, "dataset": args.dataset, "device": device,
              "train_res": args.train_res},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)
    logging.success(f"cached {len(cache.z)} latents (dim={cache.latent_dim}) -> {args.out}")


if __name__ == "__main__":
    main()
