"""The agent pipeline: every stage of training rp1 on a frozen world model, in one run.

1. ``cache``     -- encode the dataset into a latent cache, one row per primitive step
2. ``subsample`` -- keep one row per action block of ``frameskip`` steps
3. ``actions``   -- extract the action h5 the planner trainer indexes
4. ``value``     -- the offline goal-conditioned quasimetric value (``metric`` phase)
5. ``planner``   -- the planner, trained actor-critic through the frozen world model (``rp1_ac`` phase)

Caches and the action h5 go to ``cache_directory`` and are reused across runs, so
``training.stages=[value,planner]`` iterates on the recipe without re-encoding.
``training.value.*`` and ``training.planner.*`` override the two phase configs.
``training.teacher=<value>`` trains the planner against an existing value, such
as a snapshot the value stage saved with ``save_every``, instead of this run's.

Example::

    pixi run posttrain training.wm=assets/core/world_model/cube_lewm \
        training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance \
        training.name=cube_lewm training.planner.action_limit=1.6
"""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from rp1.utils.config import dispatch, get_config_root, phase_config
from rp1.utils.logging import logger
from rp1.utils.run import save_stage_config

STAGES = ("cache", "subsample", "actions", "value", "planner")
# stage overrides that configure the value and planner architectures rather than their trainers
VALUE_ARCHITECTURE = ("depth", "head", "symmetric", "eikonal_weight")
PLANNER_ARCHITECTURE = ("action_limit", "iterations")


def _stage(parent: DictConfig, index: int, name: str, config_name: str, **values: object) -> object:
    """Compose a stage and execute it inside the pipeline's parent run."""
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


def _route(overrides: dict[str, object], architecture: tuple[str, ...], group: str) -> dict[str, object]:
    """Stage overrides as dotted keys: architecture keys onto ``group``, the rest onto the trainer."""
    return {(f"{group}.{key}" if key in architecture else f"training.{key}"): value for key, value in overrides.items()}


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "training")
    stages = set(args.stages)
    unknown = stages - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {list(STAGES)}")
    cache_directory = Path(str(args.cache_directory)).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_fs1 = str(cache_directory / f"{args.name}_fs1.pt")
    # a phase-multiplexed cache is a different artefact from the single-phase one
    phases = f"p{args.actor_phases}" if args.actor_phases > 1 else ""
    cache_fs5 = str(cache_directory / f"{args.name}_fs{args.frameskip}{phases}.pt")
    actions_h5 = str(cache_directory / f"{args.name}_actions.h5")
    value_checkpoint = str(Path(cfg.run.checkpoints) / "value_td")
    if args.teacher is not None:
        if "value" in stages:
            raise ValueError("training.teacher replaces the value stage; drop `value` from training.stages")
        value_checkpoint = str(Path(str(args.teacher)).expanduser())
        if not Path(value_checkpoint).exists():
            raise FileNotFoundError(f"teacher value not found: {value_checkpoint}")
        logger.info(f"Planner teacher: {value_checkpoint}")
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
                "preparation.phases": args.actor_phases,
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
        # imagined latents are one action block apart at plan time, so a window
        # value trained at any other spacing never sees the windows it was trained on
        raise ValueError(
            f"value.window_frames={window_frames} requires value.window_lag == "
            f"frameskip ({args.frameskip}), got {window_lag}"
        )
    if "value" in stages:
        run_stage(
            "value",
            "training/phases/agent/metric",
            **{
                "training.cache": cache_fs1,
                "runtime.device": args.device,
                "runtime.seed": args.seed,
            },
            **{"output.checkpoint": "value_td"},
            **_route(_overrides(args.value), VALUE_ARCHITECTURE, "core.agent.value"),
        )
    if "planner" in stages:
        planner_overrides = _overrides(args.planner)
        if planner_overrides.get("grounding") is not None:
            # the grounding probe is fitted on the dataset's state column
            planner_overrides["state_h5"] = str(args.dataset)
        run_stage(
            "planner",
            "training/phases/agent/rp1_ac",
            **{
                "training.cache": cache_fs5,
                "training.cache_td": cache_fs1,
                "training.h5": actions_h5,
                "training.wm": str(args.wm),
                "training.init_value": value_checkpoint,
                # rp1_ac checks a window value's lag against the action block
                "training.window_lag": window_lag,
                "runtime.seed": args.seed,
            },
            **{"output.planner_checkpoint": "planner.pt", "output.value_checkpoint": "value_ac"},
            **_route(planner_overrides, PLANNER_ARCHITECTURE, "core.agent.planner"),
        )
    planner = Path(cfg.run.checkpoints) / "planner.pt"
    logger.success(
        f"rp1 pipeline finished. Evaluate with: pixi run evaluate core/agent/solver=rp1 "
        f"core.agent.solver.checkpoint.path={planner}"
    )
