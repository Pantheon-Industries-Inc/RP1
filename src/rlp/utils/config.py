from __future__ import annotations

import os
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from rlp.utils.logging import configured_logging, logger
from rlp.utils.run import RunMetadata, RunPaths, save_config


def get_config_root() -> Path:
    candidates: list[Path] = []
    if override := os.environ.get("RLP_CONFIG_DIR"):
        candidates.append(Path(override))
    candidates.extend(
        [
            Path(__file__).resolve().parents[3] / "configs",
            Path(sysconfig.get_path("data")) / "share" / "rlp" / "configs",
        ]
    )
    for root in candidates:
        if root.is_dir():
            return root
    searched = ", ".join(map(str, candidates))
    raise FileNotFoundError(f"RLP Hydra config directory not found; searched: {searched}")


def validate_config(cfg: DictConfig) -> None:
    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")

    runtime = cfg.get("runtime")
    if runtime is not None and int(runtime.seed) < 0:
        raise ValueError("runtime.seed must be non-negative")

    data = cfg.get("data")
    if data is not None and "train_split" in data and not 0.0 < float(data.train_split) < 1.0:
        raise ValueError("data.train_split must be between zero and one")

    logging = cfg.get("logging")
    if logging is not None and "wandb" in logging:
        mode = str(logging.wandb.mode)
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"Unsupported logging.wandb.mode: {mode}")

    core = cfg.get("core")
    if core is not None and "policy" in core:
        if core.policy.kind not in {"random", "no_move", "world_model"}:
            raise ValueError(f"Unsupported core.policy.kind: {core.policy.kind}")
        if core.policy.kind == "world_model" and not core.policy.checkpoint:
            raise ValueError("core.policy.checkpoint is required for a world-model policy")
    if core is not None and "value" in core and "kind" in core.value:
        if core.value.kind not in {"latent", "metric"}:
            raise ValueError(f"Unsupported core.value.kind: {core.value.kind}")
        if core.value.kind == "metric" and not core.value.checkpoints:
            raise ValueError("core.value.checkpoints must not be empty for a metric value")

    benchmark = cfg.get("benchmark")
    if benchmark is not None:
        if int(benchmark.num_episodes) <= 0:
            raise ValueError("benchmark.num_episodes must be positive")
        episode_range = benchmark.get("episode_range")
        if episode_range:
            try:
                low, high = map(int, str(episode_range).split(":"))
            except ValueError as error:
                raise ValueError("benchmark.episode_range must use LO:HI syntax") from error
            if low < 0 or high <= low:
                raise ValueError("benchmark.episode_range must satisfy 0 <= LO < HI")

    planning = cfg.get("planning")
    if planning is not None:
        for key in ("horizon", "receding_horizon", "action_block"):
            if int(planning[key]) <= 0:
                raise ValueError(f"planning.{key} must be positive")
        if int(planning.budget) <= 0:
            raise ValueError("planning.budget must be positive")
        if int(planning.budget) < int(planning.receding_horizon):
            raise ValueError("planning.budget must be at least planning.receding_horizon")
        deadline = planning.get("deadline")
        if deadline is not None:
            if int(deadline) <= 0:
                raise ValueError("planning.deadline must be positive")
            if int(deadline) > int(planning.budget):
                raise ValueError("planning.deadline must fall within the planning budget")


def phase_config(cfg: DictConfig, phase: str, *components: DictConfig) -> DictConfig:
    """Flatten the public config sections for legacy algorithm internals."""
    sections: list[object] = [OmegaConf.to_container(cfg[phase], resolve=False)]
    for name in ("runtime", "planning"):
        if name in cfg:
            sections.append(OmegaConf.to_container(cfg[name], resolve=False))
    sections.extend(OmegaConf.to_container(component, resolve=False) for component in components)
    args = OmegaConf.merge(*sections)
    if not isinstance(args, DictConfig):
        raise TypeError(f"{phase} configuration must be a mapping")
    for name in ("data", "environment", "output", "run"):
        if name in cfg:
            args[name] = OmegaConf.to_container(cfg[name], resolve=False)
    return args


def dispatch(cfg: DictConfig) -> object:
    validate_config(cfg)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cast(object, call(cfg.entrypoint, cfg=cfg, _recursive_=False))


def run_hydra[ResultT](
    task: Callable[[DictConfig], ResultT],
    *,
    config_name: str,
    selector: tuple[str, str] | None = None,
) -> ResultT:
    overrides = list(sys.argv[1:])
    if selector is not None:
        field, group = selector
        prefix = f"{field}="
        matches = [(index, arg.removeprefix(prefix)) for index, arg in enumerate(overrides) if arg.startswith(prefix)]
        if len(matches) > 1:
            raise SystemExit(f"Specify {field}=<name> only once")
        if matches:
            index, name = matches[0]
            if not name or "/" in name or name.startswith("."):
                raise SystemExit(f"Invalid {field} selection: {name!r}")
            config_name = f"{group}/{name}"
            del overrides[index]
    with initialize_config_dir(config_dir=str(get_config_root()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    paths = RunPaths.create(cfg.logging.run_root)
    paths.attach(cfg)
    level = str(cfg.logging.level)
    with configured_logging(paths.log, level):
        logger.info(f"Run directory: {paths.directory}")
        metadata = None
        try:
            metadata = RunMetadata(paths, cfg)
            save_config(cfg, paths.config)
            result = task(cfg)
        except KeyboardInterrupt:
            logger.warning("Run interrupted")
            if metadata is not None:
                metadata.finish("interrupted", "KeyboardInterrupt")
            raise SystemExit(130) from None
        except BaseException as error:  # noqa: BLE001 - command boundary records every failed lifecycle
            logger.opt(exception=error).error("Run failed")
            if metadata is not None:
                metadata.finish("failed", f"{type(error).__name__}: {error}")
            raise SystemExit(1) from None
        assert metadata is not None
        metadata.finish("succeeded")
        logger.info("Run completed")
        return result


__all__ = ["dispatch", "get_config_root", "phase_config", "run_hydra", "validate_config"]
