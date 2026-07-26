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
"""

import argparse

import torch
from loguru import logger as logging

from stable_worldmodel.trm import LatentCache, learners, save_metric
from stable_worldmodel.trm.learners.contrastive import ContrastiveConfig
from stable_worldmodel.trm.learners.regression import RegressionConfig
from stable_worldmodel.trm.learners.qrl import QRLConfig
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
    p.add_argument("--learner", required=True, choices=["regression", "td", "contrastive", "qrl"])
    p.add_argument("--qrl-eps", type=float, default=0.25, help="QRL constraint slack")
    p.add_argument("--qrl-spread-temp", type=float, default=100.0, help="QRL spreading temperature")
    p.add_argument("--qrl-step-cost", type=float, default=1.0, help="QRL cost of one env step")
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
    p.add_argument("--head", choices=["mlp", "quasimetric", "iqe"], default="quasimetric")
    p.add_argument("--eikonal-weight", type=float, default=0.0,
                   help="Eik-HIQL unit-gradient penalty weight (0 = off)")
    p.add_argument("--num-components", type=int, default=8, help="iqe head only")
    p.add_argument("--td-weight", type=float, default=1.0,
                   help="weight on TD distance regression (0 = pure ranking)")
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="pairwise ranking hinge weight (0 = off)")
    p.add_argument("--rank-margin", type=float, default=0.5,
                   help="ranking margin per step of true separation")
    p.add_argument("--symmetric", action="store_true", help="mlp head: symmetrize d(x,y)")
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--no-balanced", action="store_true")
    # contrastive
    p.add_argument("--rep-dim", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
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
            eikonal_weight=args.eikonal_weight, num_components=args.num_components,
            rank_weight=args.rank_weight, rank_margin=args.rank_margin,
            td_weight=args.td_weight,
            n_step=args.n_step, gamma=args.gamma, expectile=args.expectile,
            p_cross=args.p_cross, balanced=(not args.no_balanced), max_delta=args.max_delta,
            batch_size=args.batch_size, steps=args.steps, seed=args.seed,
            symmetric=args.symmetric,
        )
        module = learners.td.fit(cache, cfg, device)
        arch = {"head": args.head, "hidden_dim": args.hidden_dim, "depth": args.depth,
                "embed_dim": args.embed_dim, "softplus": True, "symmetric": args.symmetric,
                "num_components": args.num_components}
    elif args.learner == "qrl":
        cfg = QRLConfig(
            head=args.head, hidden_dim=args.hidden_dim, depth=args.depth,
            embed_dim=args.embed_dim, num_components=args.num_components,
            batch_size=args.batch_size, steps=args.steps, seed=args.seed,
            eps=args.qrl_eps, spread_temp=args.qrl_spread_temp, step_cost=args.qrl_step_cost,
            rank_weight=args.rank_weight, rank_margin=args.rank_margin,
        )
        module = learners.qrl.fit(cache, cfg, device)
        arch = {"head": args.head, "hidden_dim": args.hidden_dim, "depth": args.depth,
                "embed_dim": args.embed_dim, "softplus": True, "symmetric": False,
                "num_components": args.num_components}
    else:  # contrastive
        cfg = ContrastiveConfig(
            hidden_dim=args.hidden_dim, rep_dim=args.rep_dim, depth=args.depth,
            temperature=args.temperature, batch_size=args.batch_size, steps=args.steps, seed=args.seed,
        )
        module = learners.contrastive.fit(cache, cfg, device)
        arch = {"hidden_dim": args.hidden_dim, "rep_dim": args.rep_dim, "depth": args.depth}

    save_metric(module.cpu(), learner, cache.latent_dim, arch, args.out)
    logging.success(f"saved {learner} metric -> {args.out}")


if __name__ == "__main__":
    main()
