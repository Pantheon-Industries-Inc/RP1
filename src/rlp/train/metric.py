"""Train a TRM metric head on a frozen-latent cache.

Three learners share the same cache and produce the same ``cost(z_pred, z_goal)``
interface:

* ``regression``  -- horizon-matched temporal regression (the paper's method).
* ``td``          -- offline goal-conditioned temporal-distance TD.
* ``contrastive`` -- contrastive value learning (InfoNCE).

``labels=shuffled`` (regression only) trains the paper's negative control.

Example::

    pixi run train model=metric cache=caches/tworoom_state.pt \
        learner=regression output.checkpoint=tworoom_regression scale=100
"""

from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.value import build_metric, learners, save_metric
from rlp.core.value.learners.contrastive import ContrastiveConfig
from rlp.core.value.learners.regression import RegressionConfig
from rlp.core.value.learners.td import TDConfig
from rlp.core.world_model.runtime import pick_device
from rlp.data import LatentCache
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    # cfg arrives struct+readonly from dispatch; flatten onto an open copy so
    # the value-head keys and derived aliases can be merged in.
    flat = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    args = cast(DictConfig, OmegaConf.merge(flat, cfg.core.value))
    args.embed_dim = args.embedding_dim
    args.rep_dim = args.representation_dim

    device = pick_device(args.device)
    base_cache = LatentCache.load(args.cache, mmap=bool(args.cache_mmap))
    cache = base_cache.windowed(int(args.window_frames), int(args.window_lag))
    goal_frames = base_cache.z if bool(args.get("goal_tile")) else None
    logger.info(f"Loaded cache: {len(cache.z)} latents dim={cache.latent_dim} on {device}")

    # counterfactual agent augmentation (td only): a row-aligned cache of the
    # same frames re-rendered with the agent displaced, plus the transit cost
    aug = None
    if args.get("aug_cache") and float(args.get("aug_p", 0.0) or 0.0) > 0:
        aug_cache = LatentCache.load(str(args.aug_cache), mmap=bool(args.cache_mmap)).windowed(
            int(args.window_frames), int(args.window_lag)
        )
        if (
            len(aug_cache.z) != len(cache.z)
            or not torch.equal(aug_cache.episode_idx, cache.episode_idx)
            or not torch.equal(aug_cache.step_idx, cache.step_idx)
        ):
            raise ValueError("aug_cache rows do not align with the training cache")
        if aug_cache.state is None:
            raise ValueError("aug_cache carries no state column (transit steps expected in its last column)")
        aug = (aug_cache.z, aug_cache.state[:, -1].float())
        logger.info(f"Loaded agent-aug cache: {len(aug_cache.z)} rows, mean transit {aug[1].mean().item():.2f} steps")

    # success-tolerance relabeling (td only): logged env state row-aligned with the cache
    state = None
    tolerance = None
    if args.get("tol_relabel"):
        if not args.get("state_cache"):
            raise ValueError("tol_relabel=true requires state_cache (tools/cache_state output)")
        state_cache = LatentCache.load(str(args.state_cache), mmap=False)
        if (
            len(state_cache.z) != len(cache.z)
            or not torch.equal(state_cache.episode_idx, cache.episode_idx)
            or not torch.equal(state_cache.step_idx, cache.step_idx)
        ):
            raise ValueError("state_cache rows do not align with the training cache")
        state = state_cache.z.numpy()
        tolerance = cast(dict[str, Any] | None, OmegaConf.to_container(args.tol, resolve=True))
        logger.info(f"Loaded state cache for tolerance relabeling: {state.shape}, tol={tolerance}")

    learner = "shuffled" if (args.learner == "regression" and args.labels == "shuffled") else args.learner

    module: nn.Module
    if args.learner == "l2":
        module = build_metric("l2", cache.latent_dim, {})
    elif args.learner == "regression":
        scale = args.scale
        if scale is None:  # default scale ~ horizon so targets aren't dwarfed
            import numpy as np

            lens = [len(r) for r in cache.episodes().values()]
            scale = float(np.percentile(lens, 90))
        regression_cfg = RegressionConfig(
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            scale=scale,
            batch_size=args.batch_size,
            steps=args.steps,
            max_delta=args.max_delta,
            shuffle_labels=(args.labels == "shuffled"),
            seed=args.seed,
        )
        module = learners.regression.fit(cache, regression_cfg, device)
    elif args.learner == "td":
        td_cfg = TDConfig(
            head=args.head,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            embed_dim=args.embed_dim,
            n_step=args.n_step,
            gamma=args.gamma,
            expectile=args.expectile,
            p_cross=args.p_cross,
            balanced=(not args.no_balanced),
            max_delta=args.max_delta,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
            symmetric=args.symmetric,
            eikonal_weight=args.eikonal_weight,
            num_components=args.num_components,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            aug_p=float(args.get("aug_p", 0.0) or 0.0),
            aug_transit_scale=float(args.get("aug_transit_scale", 1.0)),
            near_frac=float(args.get("near_frac", 0.0) or 0.0),
            near_max=int(args.get("near_max", 3) or 3),
            expectile_near=(None if args.get("expectile_near") is None else float(args.expectile_near)),
            near_steps=float(args.get("near_steps", 3.0) or 3.0),
            near_weight=float(args.get("near_weight", 0.0) or 0.0),
            goal_tile=bool(args.get("goal_tile")),
        )
        module = learners.td.fit(
            cache, td_cfg, device, aug=aug, state=state, tolerance=tolerance, goal_frames=goal_frames
        )
    else:  # contrastive
        contrastive_cfg = ContrastiveConfig(
            hidden_dim=args.hidden_dim,
            rep_dim=args.rep_dim,
            depth=args.depth,
            temperature=args.temperature,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
        )
        module = learners.contrastive.fit(cache, contrastive_cfg, device)
    checkpoint = save_metric(module.cpu(), run_name=args.output.checkpoint, cache_dir=args.run.directory)
    logger.success(f"Saved {learner} metric to {checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/metric")


if __name__ == "__main__":
    main()
