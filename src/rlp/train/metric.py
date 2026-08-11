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

from omegaconf import DictConfig, OmegaConf
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.value import learners, save_metric
from rlp.core.value.learners.contrastive import ContrastiveConfig
from rlp.core.value.learners.regression import RegressionConfig
from rlp.core.value.learners.td import TDConfig
from rlp.core.world_model.runtime import pick_device
from rlp.data import LatentCache
from rlp.logging import logger


def _run(cfg: DictConfig) -> None:
    args = OmegaConf.merge(cfg, cfg.core.value)
    args.embed_dim = args.embedding_dim
    args.rep_dim = args.representation_dim

    device = pick_device(args.device)
    cache = LatentCache.load(args.cache)
    logger.info(f"Loaded cache: {len(cache.z)} latents dim={cache.latent_dim} on {device}")

    learner = "shuffled" if (args.learner == "regression" and args.labels == "shuffled") else args.learner

    if args.learner == "regression":
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
        module: nn.Module = learners.regression.fit(cache, regression_cfg, device)
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
        )
        module = learners.td.fit(cache, td_cfg, device)
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
