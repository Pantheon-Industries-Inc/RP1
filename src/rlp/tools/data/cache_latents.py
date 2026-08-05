"""Encode a logged dataset into a frozen-latent cache for metric training.

The expensive encoder runs once here; the three metric learners then train on
the cheap cached latents.

Example (state WM)::

    pixi run tool tool=cache_latents wm=statewm_tworoom \
        dataset=tworoom_expert.lance out=caches/tworoom_state.pt

Example (pixel LeWM)::

    pixi run tool tool=cache_latents wm=quentinll/lewm-cube \
        dataset=ogbench/cube_single_multiview_expert.lance \
        out=caches/cube_lewm.pt state_key=proprio
"""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.core.world_model.runtime import build_featurizer, is_statewm, load_wm, pick_device
from rlp.data import encode_dataset
from rlp.data.protocols import Array, RowBatch
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    args = cfg

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    featurizer = build_featurizer(wm, device=device, train_res=args.train_res)

    dataset = swm.data.load_dataset(args.dataset)
    state_key: str | None = str(args.state_key) if args.state_key else None
    if state_key is None and is_statewm(wm):
        candidate = getattr(wm, "obs_key", None)
        if not isinstance(candidate, str):
            raise TypeError("StateWM obs_key must be a string")
        state_key = candidate

    if args.max_rows is not None:
        full = dataset

        class _Sub:
            column_names = full.column_names

            def get_col_data(self, c: str) -> Array:
                return np.asarray(full.get_col_data(c)[: args.max_rows])

            def get_row_data(self, i: list[int]) -> RowBatch:
                return {str(key): np.asarray(value) for key, value in full.get_row_data(i).items()}

        dataset = _Sub()

    cache = encode_dataset(
        dataset,
        featurizer,
        batch_size=args.batch_size,
        state_key=state_key,
        meta={
            "wm": args.wm,
            "dataset": args.dataset,
            "device": device,
            "train_res": args.train_res,
        },
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)
    logger.success(f"Cached {len(cache.z)} latents (dim={cache.latent_dim}) at {args.out}")


def main() -> object:
    return run_hydra(dispatch, config_name="tools/cache_latents")


if __name__ == "__main__":
    main()
