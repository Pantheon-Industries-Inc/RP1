"""Train a TRM metric head on a frozen-latent cache.

Three learners share the same cache and produce the same ``cost(z_pred, z_goal)``
interface:

* ``regression``  -- horizon-matched temporal regression (the paper's method).
* ``td``          -- offline goal-conditioned temporal-distance TD.
* ``contrastive`` -- contrastive value learning (InfoNCE).

``--labels shuffled`` (regression only) trains the paper's negative control.

Example::

    python scripts/trm/train_metric.py --cache caches/tworoom_state.pt \
        --learner regression --out metrics/tworoom_regression.pt --scale 100

If the cache was built with ``cache_latents.py --compress <spec>`` (flat
patch-token bases such as DINO), pass the SAME ``--compress <spec>`` here: the
head trains on the compressed rows and is wrapped in ``CompressedMetric`` so the
planner can keep handing it full flat tokens.
"""

import argparse

import torch
from loguru import logger as logging

from stable_worldmodel.trm import LatentCache, learners, save_metric
from stable_worldmodel.trm.io import CompressedMetric, compress_arch, projection_sha256
from stable_worldmodel.trm.learners.contrastive import ContrastiveConfig
from stable_worldmodel.trm.learners.regression import RegressionConfig
from stable_worldmodel.trm.learners.td import TDConfig


def pick_device(name: str = "auto") -> str:
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--learner", required=True, choices=["regression", "td", "contrastive"])
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    # regression
    p.add_argument("--scale", type=float, default=None, help="temporal scale s (default: 90th-pct episode length)")
    p.add_argument("--max-delta", type=int, default=None, help="cap separation (paper ablation)")
    p.add_argument("--labels", choices=["true", "shuffled"], default="true")
    # td
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--head", choices=["mlp", "quasimetric"], default="quasimetric")
    p.add_argument("--symmetric", action="store_true", help="mlp head: symmetrize d(x,y)")
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--no-balanced", action="store_true")
    # contrastive
    p.add_argument("--rep-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    # latent compression (flat patch-token bases)
    p.add_argument("--compress", default=None,
                   help="compressor the cache was built with: mean | rp<D> | spatial<k>. "
                        "Recorded in arch so load_metric rebuilds the CompressedMetric "
                        "wrapper (the planner passes full flat tokens).")
    p.add_argument("--pool-patches", type=int, default=0,
                   help="[legacy] equivalent to --compress mean with P patches")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(args.device)
    cache = LatentCache.load(args.cache)
    logging.info(f"loaded cache: {len(cache.z)} latents dim={cache.latent_dim} on {device}")

    learner = "shuffled" if (args.learner == "regression" and args.labels == "shuffled") else args.learner

    if args.learner == "regression":
        scale = args.scale
        if scale is None:  # default scale ~ horizon so targets aren't dwarfed
            import numpy as np
            lens = [len(r) for r in cache.episodes().values()]
            scale = float(np.percentile(lens, 90))
        cfg = RegressionConfig(
            hidden_dim=args.hidden_dim, depth=args.depth, scale=scale,
            batch_size=args.batch_size, steps=args.steps, max_delta=args.max_delta,
            shuffle_labels=(args.labels == "shuffled"), seed=args.seed,
        )
        module = learners.regression.fit(cache, cfg, device)
        arch = {"hidden_dim": args.hidden_dim, "depth": args.depth, "softplus": True,
                "symmetric": False, "scale": scale}
    elif args.learner == "td":
        cfg = TDConfig(
            head=args.head, hidden_dim=args.hidden_dim, depth=args.depth, embed_dim=args.embed_dim,
            n_step=args.n_step, gamma=args.gamma, expectile=args.expectile,
            p_cross=args.p_cross, balanced=(not args.no_balanced), max_delta=args.max_delta,
            batch_size=args.batch_size, steps=args.steps, seed=args.seed,
            symmetric=args.symmetric,
        )
        module = learners.td.fit(cache, cfg, device)
        arch = {"head": args.head, "hidden_dim": args.hidden_dim, "depth": args.depth,
                "embed_dim": args.embed_dim, "softplus": True, "symmetric": args.symmetric}
    else:  # contrastive
        cfg = ContrastiveConfig(
            hidden_dim=args.hidden_dim, rep_dim=args.rep_dim, depth=args.depth,
            temperature=args.temperature, batch_size=args.batch_size, steps=args.steps, seed=args.seed,
        )
        module = learners.contrastive.fit(cache, cfg, device)
        arch = {"hidden_dim": args.hidden_dim, "rep_dim": args.rep_dim, "depth": args.depth}

    spec = args.compress or ("mean" if args.pool_patches else None)
    if spec:
        # the head trained on compressed rows; wrap it so the planner can hand it
        # full flat patch tokens at deploy time (dispatch is by last dim)
        meta = cache.meta or {}
        cache_spec = meta.get("compress") or ("mean" if meta.get("pool_patches") else None)
        assert cache_spec == spec, (
            f"cache was built with compress={cache_spec!r} but --compress={spec!r}"
        )
        full = meta.get("compress_full_dim")
        patches = meta.get("compress_patches") or meta.get("pool_patches")
        token_dim = meta.get("compress_token_dim")
        seed = int(meta.get("compress_seed", 0))
        if full is None:  # legacy mean-pool cache without the new meta keys
            patches = int(patches or 196)
            token_dim = int(token_dim or cache.latent_dim)
            full = patches * token_dim
        module = CompressedMetric(module, spec, full, cache.latent_dim,
                                  patches=patches, token_dim=token_dim, seed=seed)
        if meta.get("compress_sha256"):
            # hard closure: the projection rebuilt here must be bit-identical to
            # the one the cache was encoded with, else train and deploy disagree
            got = projection_sha256(module.proj)
            assert got == meta["compress_sha256"], (
                f"rp matrix mismatch: cache {meta['compress_sha256'][:16]} vs rebuilt {got[:16]}"
            )
            logging.info(f"rp matrix sha256 matches cache ({got[:16]})")
        arch.update(compress_arch(spec, full, cache.latent_dim, patches, token_dim, seed))
        logging.info(f"wrapped head in CompressedMetric({spec}: {full} -> {cache.latent_dim})")

    save_metric(module.cpu(), learner, cache.latent_dim, arch, args.out)
    logging.success(f"saved {learner} metric -> {args.out}")


if __name__ == "__main__":
    main()
