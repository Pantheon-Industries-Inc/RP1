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

Example (DINO flat patch tokens, compressed 75264 -> 1024 by random projection)::

    python scripts/trm/cache_latents.py --wm dinowmnp_reacher \
        --dataset dmc/reacher_random.lance --out caches/reacher_rp1024.pt \
        --compress rp1024

``--compress`` shrinks the cached latent so the FULL dataset fits (2M x 1024 =
8 GB instead of 600 GB at 75264-d). The identical compressor is re-applied at
plan time by ``CompressedMetric`` (see ``stable_worldmodel/trm/io.py`` for why
``rp<D>`` -- which preserves the full-vector L2 geometry CEM already exploits --
is preferred over the position-blind ``mean``).
"""

import argparse
from pathlib import Path

import torch
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.trm import encode_dataset
from stable_worldmodel.trm.io import compressed_dim, make_projection, parse_compress, projection_sha256

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
    p.add_argument("--compress", default=None,
                   help="compress flat patch-token latents before caching: "
                        "'mean' (196x384 -> 384), 'rp<D>' (random projection -> D, "
                        "L2-geometry preserving), 'spatial<k>' (14x14 grid -> kxk). "
                        "Must be matched by --compress at train_metric time.")
    p.add_argument("--compress-seed", type=int, default=0,
                   help="RNG seed for the rp<D> projection matrix (recorded in cache meta)")
    p.add_argument("--pool-patches", type=int, default=0,
                   help="[legacy] equivalent to --compress mean with P patches")
    p.add_argument("--train-res", type=int, default=None,
                   help="checkpoint's native training image resolution; frames are "
                        "bottlenecked through it (e.g. 64 for OGBench play retrains). "
                        "Must match eval.train_res at plan time.")
    args = p.parse_args()

    spec = args.compress or ("mean" if args.pool_patches else None)

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    featurizer = build_featurizer(wm, device=device, train_res=args.train_res)

    cmeta: dict = {"compress": None}
    if spec:
        mode, param = parse_compress(spec)
        _base_featurize = featurizer
        state: dict = {}

        def featurizer(rows, _f=_base_featurize):  # noqa: F811
            z = _f(rows)  # (B, full_dim) flat patch tokens
            full = z.shape[-1]
            if not state:
                patches = int(args.pool_patches) or None
                token_dim = None
                if mode in ("mean", "spatial"):
                    patches = patches or 196
                    assert full % patches == 0, f"{patches} patches does not divide {full}"
                    token_dim = full // patches
                out = compressed_dim(spec, token_dim or 384)
                state.update(full=full, patches=patches, token_dim=token_dim, out=out)
                if mode == "rp":
                    P = make_projection(full, out, args.compress_seed).to(z.device)
                    state["proj"] = P
                    state["sha"] = projection_sha256(P)
                    logging.info(f"rp matrix {full}->{out} seed={args.compress_seed} sha256={state['sha'][:16]}")
                cmeta.update(compress=spec, compress_full_dim=full,
                             compress_out_dim=out, compress_patches=patches,
                             compress_token_dim=token_dim,
                             compress_seed=int(args.compress_seed),
                             compress_sha256=state.get("sha"))
                logging.info(f"compress '{spec}': {full} -> {out}")
            if mode == "rp":
                return z.float() @ state["proj"]
            d = state["token_dim"]
            t = z.reshape(z.shape[0], state["patches"], d)
            if mode == "mean":
                return t.mean(1)
            g = int(round(state["patches"] ** 0.5))
            k = param
            b = g // k
            return t.reshape(z.shape[0], k, b, k, b, d).mean(dim=(2, 4)).reshape(z.shape[0], k * k * d)

    dataset = swm.data.load_dataset(args.dataset)
    state_key = args.state_key or (wm.obs_key if is_statewm(wm) else None)

    if args.max_rows is not None:
        full_ds = dataset

        class _Sub:
            column_names = full_ds.column_names

            def get_col_data(self, c):
                return full_ds.get_col_data(c)[: args.max_rows]

            def get_row_data(self, i):
                return full_ds.get_row_data(i)

        dataset = _Sub()

    cache = encode_dataset(
        dataset, featurizer, batch_size=args.batch_size, state_key=state_key,
        meta={"wm": args.wm, "dataset": args.dataset, "device": device,
              "train_res": args.train_res,
              "pool_patches": int(args.pool_patches) or None, **cmeta},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cache.save(args.out)
    logging.success(f"cached {len(cache.z)} latents (dim={cache.latent_dim}) -> {args.out}")


if __name__ == "__main__":
    main()
