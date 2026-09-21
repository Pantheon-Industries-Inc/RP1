"""Composed, in-process RLP replication pipeline.

One command reproduces the paper's full RLP training stack on top of a frozen
pretrained world model:

1. ``cache``     — encode the offline dataset into a per-frame (fs1) latent cache
2. ``subsample`` — horizon-match the cache to the planner's frameskip (fs5)
3. ``actions``   — extract the action-only h5 the planner trainer indexes
4. ``value``     — offline goal-conditioned quasimetric critic (TD + HER + expectile)
5. ``planner``   — the RLP plan refiner, trained actor-critic through the frozen WM

Example (OGBench Cube on the tracked LeWM checkpoint)::

    pixi run training model=rlp training.wm=assets/core/world_model/lewm_cube \
        training.dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance \
        training.name=cube_lewm training.planner.action_limit=1.6

Stages write reusable artifacts (caches, h5) into ``cache_directory`` and
checkpoints into the run's ``checkpoints/`` directory. Re-runs can select
stages, e.g. ``training.stages=[value,planner]`` to iterate on the value
or planner recipe against existing caches. Per-stage hyperparameters are
overridable through the ``value.*`` and ``planner.*`` subtrees; their defaults
are the paper's OGBench Cube recipe.
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rlp.utils.config import dispatch, get_config_root, phase_config
from rlp.utils.logging import logger
from rlp.utils.run import save_stage_config

STAGES = ("cache", "subsample", "actions", "value", "planner")


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


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "training")
    stages = set(args.stages)
    unknown = stages - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {list(STAGES)}")
    cache_directory = Path(str(args.cache_directory)).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_fs1 = str(cache_directory / f"{args.name}_fs1.pt")
    cache_fs5 = str(cache_directory / f"{args.name}_fs{args.frameskip}.pt")
    actions_h5 = str(cache_directory / f"{args.name}_actions.h5")
    value_checkpoint = str(Path(cfg.run.checkpoints) / "value_td")
    teacher = cfg.get("teacher")
    if teacher:
        # joint teacher x actor early stopping: train the planner against a chosen teacher snapshot
        # (a metric dir saved by an earlier value stage) instead of this run's value_td
        value_checkpoint = str(Path(str(teacher)).expanduser())
        if "value" not in skip:
            raise ValueError("teacher=<dir> requires skip=[...,value]: the value stage would be trained and ignored")
        if not Path(value_checkpoint).exists():
            raise FileNotFoundError(f"teacher metric dir not found: {value_checkpoint}")
        logger.info(f"Planner teacher override: {value_checkpoint}")
    stage_index = 0

    def run_stage(name: str, config_name: str, **values: object) -> object:
        nonlocal stage_index
        stage_index += 1
        return _stage(cfg, stage_index, name, config_name, **values)

    if "cache" in stages:
        run_stage(
            "cache",
            "training/data/job/cache_latents",
            **{
                "preparation.wm": str(args.wm),
                "preparation.dataset": str(args.dataset),
                "preparation.out": cache_fs1,
                "preparation.state_key": args.state_key,
                "preparation.train_res": args.train_res,
                "preparation.max_episodes": args.train_episodes,
                "runtime.device": args.device,
            },
        )
    if "subsample" in stages:
        run_stage(
            "subsample",
            "training/data/job/subsample_cache",
            **{
                "preparation.inp": cache_fs1,
                "preparation.out": cache_fs5,
                "preparation.frameskip": args.frameskip,
            },
        )
    if "actions" in stages:
        run_stage(
            "actions",
            "training/data/job/build_action_h5",
            **{"preparation.dataset": str(args.dataset), "preparation.output": actions_h5},
        )
    window_frames = args.value.get("window_frames")
    window_lag = args.value.get("window_lag")
    windowed = window_frames is not None and int(window_frames) > 1
    if windowed and (window_lag is None or int(window_lag) != int(args.frameskip)):
        # imagined latents are one action block apart at plan time, so a
        # window teacher trained at any other spacing would never be queried
        # with the windows it was trained on
        raise ValueError(
            f"value.window_frames={window_frames} requires value.window_lag == "
            f"frameskip ({args.frameskip}), got {window_lag}"
        )
    if "value" in stages:
        value_overrides = _overrides(args.value)
        # `depth` is a value-architecture knob (MRN head), not a trainer flag;
        # route it onto the composed value group where training/metric reads it.
        value_depth = value_overrides.pop("depth", None)
        if value_depth is not None:
            value_overrides["core.agent.value.depth"] = value_depth
        run_stage(
            "value",
            "training/phases/agent/metric",
            **{
                "training.cache": cache_fs1,
                "training.learner": "td",
                "runtime.device": args.device,
                "runtime.seed": args.seed,
            },
            **{"output.checkpoint": "value_td"},
            **{
                (key if key.startswith("core.") else f"training.{key}"): value for key, value in value_overrides.items()
            },
        )
    if "planner" in stages:
        planner_overrides = _overrides(args.planner)
        # `action_limit` and `iterations` are planner-architecture knobs, not
        # trainer flags; route them onto the composed planner group where
        # lip_ac reads them.
        action_limit = planner_overrides.pop("action_limit")
        planner_overrides["core.agent.planner.action_limit"] = action_limit
        iterations = planner_overrides.pop("iterations", None)
        if iterations is not None:
            planner_overrides["core.agent.planner.iterations"] = iterations
        run_stage(
            "planner",
            "training/phases/agent/lip_ac",
            **{
                "training.cache": cache_fs5,
                "training.cache_td": cache_fs1,
                "training.h5": actions_h5,
                "training.wm": str(args.wm),
                "training.init_value": value_checkpoint,
                # a windowed value stage hands the planner a windowed init_value;
                # forward the lag so lip_ac can validate it against the action block
                "training.window_lag": window_lag,
                "runtime.seed": args.seed,
            },
            **{"output.planner_checkpoint": "planner.pt", "output.value_checkpoint": "value_ac"},
            **{
                (key if key.startswith("core.") else f"training.{key}"): value
                for key, value in planner_overrides.items()
            },
        )
    logger.success(
        "RLP pipeline finished. Evaluate with: "
        f"pixi run inference core/solver=lip core.solver.checkpoint.path={Path(cfg.run.checkpoints) / 'planner.pt'}"
    )
