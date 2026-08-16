"""Composed, in-process RLP replication pipeline.

One command reproduces the paper's full RLP training stack on top of a frozen
pretrained world model:

1. ``cache``     — encode the offline dataset into a per-frame (fs1) latent cache
2. ``subsample`` — horizon-match the cache to the planner's frameskip (fs5)
3. ``actions``   — extract the action-only h5 the planner trainer indexes
4. ``value``     — offline goal-conditioned quasimetric critic (TD + HER + expectile)
5. ``planner``   — the RLP plan refiner, trained actor-critic through the frozen WM

Example (OGBench Cube on the tracked LeWM checkpoint)::

    pixi run train model=rlp wm=assets/core/world_model/lewm_cube \
        dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance \
        name=cube_lewm planner.amax=1.6

Stages write reusable artifacts (caches, h5) into ``cache_directory`` and
checkpoints into the run's ``checkpoints/`` directory. Re-runs can skip
completed stages, e.g. ``skip=[cache,subsample,actions]`` to iterate on the value
or planner recipe against existing caches. Per-stage hyperparameters are
overridable through the ``value.*`` and ``planner.*`` subtrees; their defaults
are the paper's OGBench Cube recipe.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlp.config import dispatch, get_config_root, run_hydra
from rlp.logging import logger
from rlp.run import save_stage_config


def _stage(parent: DictConfig, index: int, name: str, config_name: str, **values: object) -> object:
    """Compose a stage and execute it inside the pipeline's parent run."""
    # run_hydra closes its Hydra context after composing the root config, so
    # each stage composes under its own context.
    with initialize_config_dir(config_dir=str(get_config_root()), version_base=None):
        stage = compose(config_name=config_name)
    with open_dict(stage):
        stage.run = OmegaConf.create(OmegaConf.to_container(parent.run, resolve=True))
    for key, value in values.items():
        OmegaConf.update(stage, key, value, merge=False)
    stage_directory = Path(parent.run.stages) / f"{index:02d}-{name}"
    save_stage_config(stage, stage_directory)
    logger.info(f"Pipeline stage started: {name}")
    result = dispatch(stage)
    logger.info(f"Pipeline stage completed: {name}")
    return result


def _overrides(subtree: DictConfig) -> dict[str, object]:
    flat = OmegaConf.to_container(subtree, resolve=True)
    if not isinstance(flat, dict):
        raise TypeError("stage override subtree must be a mapping")
    return {str(key): value for key, value in flat.items() if value is not None}


def _run(cfg: DictConfig) -> None:
    raw_skip = cfg.skip if isinstance(cfg.skip, str) else " ".join(cfg.skip)
    skip = {name.strip() for name in raw_skip.replace(",", " ").split() if name.strip()}
    unknown = skip - {"cache", "subsample", "actions", "value", "planner"}
    if unknown:
        raise ValueError(f"unknown skip stages: {sorted(unknown)}")
    cache_directory = Path(str(cfg.cache_directory)).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_fs1 = str(cache_directory / f"{cfg.name}_fs1.pt")
    cache_fs5 = str(cache_directory / f"{cfg.name}_fs{cfg.frameskip}.pt")
    actions_h5 = str(cache_directory / f"{cfg.name}_actions.h5")
    value_checkpoint = str(Path(cfg.run.checkpoints) / "value_td")
    stage_index = 0

    def run_stage(name: str, config_name: str, **values: object) -> object:
        nonlocal stage_index
        stage_index += 1
        return _stage(cfg, stage_index, name, config_name, **values)

    if "cache" not in skip:
        run_stage(
            "cache",
            "tools/cache_latents",
            wm=str(cfg.wm),
            dataset=str(cfg.dataset),
            out=cache_fs1,
            state_key=cfg.state_key,
            train_res=cfg.train_res,
            max_episodes=cfg.train_episodes,
            device=cfg.device,
        )
    if "subsample" not in skip:
        run_stage(
            "subsample",
            "tools/subsample_cache",
            inp=cache_fs1,
            out=cache_fs5,
            frameskip=cfg.frameskip,
        )
    if "actions" not in skip:
        run_stage(
            "actions",
            "tools/build_action_h5",
            dataset=str(cfg.dataset),
            output=actions_h5,
        )
    if "value" not in skip:
        run_stage(
            "value",
            "train/metric",
            cache=cache_fs1,
            learner="td",
            device=cfg.device,
            seed=cfg.seed,
            **{"output.checkpoint": "value_td"},
            **_overrides(cfg.value),
        )
    if "planner" not in skip:
        planner_overrides = _overrides(cfg.planner)
        # `amax` and `iterations` are planner-architecture knobs, not trainer
        # flags; route them onto the composed planner group where lip_ac
        # reads them.
        amax = planner_overrides.pop("amax", None)
        if amax is not None:
            planner_overrides["core.planner.action_limit"] = amax
        iterations = planner_overrides.pop("iterations", None)
        if iterations is not None:
            planner_overrides["core.planner.iterations"] = iterations
        run_stage(
            "planner",
            "train/lip_ac",
            cache=cache_fs5,
            cache_td=cache_fs1,
            h5=actions_h5,
            wm=str(cfg.wm),
            init_value=value_checkpoint,
            seed=cfg.seed,
            **{"output.planner_checkpoint": "planner.pt", "output.value_checkpoint": "value_ac"},
            **planner_overrides,
        )
    logger.success(
        "RLP pipeline finished. Evaluate with: "
        f"pixi run eval core/solver=lip core.solver.actor_path={Path(cfg.run.checkpoints) / 'planner.pt'}"
    )


def main() -> object:
    return run_hydra(dispatch, config_name="train/rlp")


if __name__ == "__main__":
    main()
