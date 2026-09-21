"""Train a TRM metric head on a frozen-latent cache.

Three learners share the same cache and produce the same ``cost(z_pred, z_goal)``
interface:

* ``regression``  -- horizon-matched temporal regression (the paper's method).
* ``td``          -- offline goal-conditioned temporal-distance TD.
* ``contrastive`` -- contrastive value learning (InfoNCE).

``labels=shuffled`` (regression only) trains the paper's negative control.

Example::

    pixi run training model=metric training.cache=caches/tworoom_state.pt \
        training.learner=regression output.checkpoint=tworoom_regression training.scale=100
"""

from omegaconf import DictConfig
from torch import nn

from rp1.core.agent.value import build_metric
from rp1.data import LatentCache
from rp1.training.harness.checkpointing import save_metric
from rp1.training.phases.agent import learners
from rp1.training.phases.agent.learners.contrastive import ContrastiveConfig
from rp1.training.phases.agent.learners.regression import RegressionConfig
from rp1.training.phases.agent.learners.td import TDConfig
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "training", cfg.core.agent.value)
    args.embed_dim = args.embedding_dim
    args.rep_dim = args.representation_dim

    device = pick_device(args.device)
    base_cache = LatentCache.load(args.cache, mmap=bool(args.cache_mmap))
    max_episodes = args.get("max_episodes")
    if max_episodes:
        base_cache = base_cache.first_episodes(int(max_episodes))
        logger.info(f"Data-volume cap: training on episodes [0, {int(max_episodes)})")
    cache = base_cache.windowed(int(args.window_frames), int(args.window_lag))
    logger.info(f"Loaded cache: {len(cache.z)} latents dim={cache.latent_dim} on {device}")

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
            softplus=args.softplus,
            symmetric=args.symmetric,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            steps=args.steps,
            n_buckets=args.n_buckets,
            max_delta=args.max_delta,
            shuffle_labels=(args.labels == "shuffled"),
            seed=args.seed,
            huber_beta=args.huber_beta,
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
            n_buckets=args.n_buckets,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            tau=args.target_update_rate,
            huber_beta=args.huber_beta,
            symmetric=args.symmetric,
            eikonal_weight=args.eikonal_weight,
            num_components=args.num_components,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            rank_max_delta=args.rank_max_delta,
            softplus=args.softplus,
            sym_frac=args.sym_frac,
            alpha_init=args.alpha_init,
            near_frac=args.near_frac,
            near_max=args.near_max,
        )

        def _save_snapshot(head: nn.Module, step: int) -> None:
            # teacher early-stopping grid: <checkpoint>_step<N> next to the final <checkpoint>
            path = save_metric(head, run_name=f"{args.output.checkpoint}_step{step}", cache_dir=args.run.directory)
            logger.info(f"Saved TD teacher snapshot step={step} to {path}")

        module = learners.td.fit(cache, td_cfg, device, save_fn=_save_snapshot if td_cfg.save_every > 0 else None)
    else:  # contrastive
        contrastive_cfg = ContrastiveConfig(
            hidden_dim=args.hidden_dim,
            rep_dim=args.rep_dim,
            depth=args.depth,
            gamma=args.contrastive_gamma,
            temperature=args.temperature,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            steps=args.steps,
            seed=args.seed,
        )
        module = learners.contrastive.fit(cache, contrastive_cfg, device)
    checkpoint = save_metric(module.cpu(), run_name=args.output.checkpoint, cache_dir=args.run.directory)
    logger.success(f"Saved {learner} metric to {checkpoint}")
