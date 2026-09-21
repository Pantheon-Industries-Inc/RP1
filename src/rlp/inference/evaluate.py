import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.policy import BasePolicy
from torch import nn

from rlp.core.agent.policy import NoMovePolicy, PlanConfig, WorldModelPolicy
from rlp.core.agent.value import MetricCost, as_planning_cost
from rlp.core.world_model.featurize import image_transform
from rlp.data.base import Array, Dataset, episode_index
from rlp.environment import World
from rlp.training.harness.checkpointing import load_metric, load_pretrained
from rlp.utils.config import dispatch, run_hydra
from rlp.utils.device import pick_device
from rlp.utils.logging import logger


def get_episodes_length(dataset: Dataset, episodes: Array) -> Array:
    episode_idx = episode_index(dataset)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def get_dataset(cfg: DictConfig, dataset_name: str) -> Dataset:
    raw_dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.data.cache_dir,
        keys_to_cache=list(cfg.data.keys_to_cache),
    )
    if not callable(getattr(raw_dataset, "get_col_data", None)) or not hasattr(raw_dataset, "column_names"):
        raise TypeError(f"dataset {dataset_name!r} does not provide the expected column interface")
    return cast(Dataset, raw_dataset)


def run(cfg: DictConfig) -> None:
    if cfg.planning.horizon * cfg.planning.action_block > cfg.planning.budget:
        raise ValueError("planning horizon x action block must not exceed the evaluation budget")
    device = pick_device(cfg.runtime.device)

    environment = cast(dict[str, Any], OmegaConf.to_container(cfg.environment, resolve=True))
    environment["max_episode_steps"] = 2 * cfg.planning.budget
    world = World(**environment, image_shape=(cfg.benchmark.image_size, cfg.benchmark.image_size))

    img_dtype = torch.bfloat16 if cfg.runtime.bfloat16 else torch.float32
    image = image_transform(cfg.benchmark.image_size, cfg.benchmark.train_resolution, img_dtype)
    transform: dict[str, Callable[[object], torch.Tensor]] = {"pixels": image, "goal": image}

    dataset = get_dataset(cfg, cfg.data.path)
    # dataset.stats: source of the action/proprio z-scoring stats. Defaults
    # to the eval dataset; point it at the checkpoint's TRAINING dataset when
    # they differ (training normalizes with train-set stats, so eval must
    # match — e.g. OGBench play-data retrains vs expert-h5 eval).
    stats_name = cfg.data.stats or cfg.data.path
    if str(stats_name) == str(cfg.data.path):
        stats_dataset = dataset
    else:
        stats_dataset = get_dataset(cfg, stats_name)
        logger.info(f"Using normalization stats from {stats_name}; evaluation tasks and goals from {cfg.data.path}")
    row_episodes = episode_index(dataset)
    ep_indices = np.unique(row_episodes)
    episode_range = cfg.benchmark.episode_range
    if episode_range:
        low, high = map(int, str(episode_range).split(":"))
        ep_indices = ep_indices[(ep_indices >= low) & (ep_indices < high)]
        if len(ep_indices) < cfg.benchmark.num_episodes:
            raise ValueError(
                f"episode range {episode_range} has {len(ep_indices)} episodes; need {cfg.benchmark.num_episodes}"
            )
        logger.info(f"Restricted evaluation pool to episodes [{low}, {high}): {len(ep_indices)} eligible")

    process: dict[str, Any] = {}
    for col in cfg.data.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy_kind = cfg.core.agent.policy.kind

    if policy_kind == "world_model":
        model = load_pretrained(cfg.core.agent.policy.checkpoint)
        if cfg.runtime.bfloat16:
            model = model.to(torch.bfloat16)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        dynamic_model = cast(Any, model)  # Stable-WM exposes architecture-specific runtime attributes.
        dynamic_model.interpolate_pos_encoding = True
        if cfg.runtime.compile:
            encoder_attr = "backbone" if hasattr(model, "backbone") else "encoder"
            encoder = getattr(model, encoder_attr)
            if not callable(encoder):
                raise TypeError(f"world-model {encoder_attr} must be callable")
            setattr(
                model,
                encoder_attr,
                torch.compile(encoder),
            )
            predictor = getattr(model, "predictor", None)
            if not callable(predictor):
                raise TypeError("world-model predictor must be callable")
            dynamic_model.predictor = torch.compile(predictor)
        planning_raw = OmegaConf.to_container(cfg.planning, resolve=True)
        if not isinstance(planning_raw, dict):
            raise TypeError("planning configuration must resolve to a mapping")
        planning = cast(dict[str, Any], planning_raw)
        config = PlanConfig(**planning)

        # optional plan-score metric hook: replace (or blend with) the raw
        # latent cost by a trained value function (ported from pod snapshot2)
        planning_cost = as_planning_cost(model)
        if not isinstance(planning_cost, nn.Module):
            raise TypeError("planning cost must also be a torch module")
        cost_model: nn.Module = planning_cost
        metric_paths = list(cfg.core.agent.value.checkpoints)
        if cfg.core.agent.value.kind == "metric":
            loaded_metrics = [load_metric(path, device=device) for path in metric_paths]
            if not loaded_metrics:
                raise ValueError("metric planning requires at least one checkpoint")
            cost_model = MetricCost(
                cost_model,
                loaded_metrics[0],
                cfg.core.agent.value.mode,
                lam=float(cfg.core.agent.value.blend_weight),
                metrics=loaded_metrics,
                deadline_mode=str(cfg.core.agent.value.deadline_mode),
            )
            logger.info(f"Plan-score metrics={metric_paths} mode={cfg.core.agent.value.mode}")

        solver = hydra.utils.instantiate(
            cfg.core.agent.solver,
            model=cost_model,
            device=device,
            seed=cfg.runtime.seed,
        )
        policy: BasePolicy = WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    else:
        policy = NoMovePolicy() if policy_kind == "no_move" else swm.policy.RandomPolicy()

    video_directory = Path(cfg.run.videos)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.benchmark.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row's episode_idx to its max_start_idx (flatten index
    # columns: lance serves them as (N,1), h5 as (N,)).
    _row_epi = row_episodes
    _row_step = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    max_start_per_row = np.full(_row_epi.shape, -1, dtype=np.int64)
    for ep_id, maximum in max_start_idx_dict.items():
        max_start_per_row[_row_epi == ep_id] = maximum

    valid_mask = (max_start_per_row >= 0) & (_row_step <= max_start_per_row)
    valid_indices = np.nonzero(valid_mask)[0]
    logger.info(f"Found {int(valid_mask.sum())} valid evaluation starting points")

    # optional hard-task filter (TwoRoom): keep only (start, goal) pairs on
    # opposite sides of the dividing wall (wall center 112; axis inferred
    # from the first door center in the observation vector)
    if cfg.benchmark.cross_wall:
        st_all = np.asarray(dataset.get_col_data("state"))
        first_door = np.asarray(dataset.get_col_data("observation"))[0, 4:6]
        axis = 0 if abs(float(first_door[0]) - 112.0) < 1e-3 else 1
        off = cfg.benchmark.goal_offset_steps
        s0 = st_all[valid_indices]
        s1 = st_all[valid_indices + off]
        cross = np.sign(s0[:, axis] - 112.0) != np.sign(s1[:, axis] - 112.0)
        valid_indices = valid_indices[cross]
        logger.info(f"Found {len(valid_indices)} cross-wall starting points on wall axis {axis}")

    if len(valid_indices) < cfg.benchmark.num_episodes:
        raise ValueError(f"Need {cfg.benchmark.num_episodes} valid evaluation starts; found {len(valid_indices)}")
    g = np.random.default_rng(cfg.runtime.seed)
    # Sort increasingly to avoid issues with HDF5Dataset indexing.
    random_episode_indices = np.sort(g.choice(valid_indices, size=cfg.benchmark.num_episodes, replace=False))

    logger.info(f"Selected evaluation row indices: {random_episode_indices.tolist()}")

    # index columns may be writer-managed (lance): use column access, not rows.
    # Flatten (lance serves (N,1)) and cast to int (lance stores these as
    # float32; the reader indexes offsets with them and requires int).
    eval_episodes = row_episodes[random_episode_indices].astype(np.int64)
    eval_start_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)[random_episode_indices].astype(np.int64)

    # Diagnostic hook: solvers that need to know WHICH dataset task each env is
    # running (e.g. the oracle-subgoal arm, which reads true future latents)
    # get the per-env (episode, start) list here. No-op for every other solver.
    _solver = getattr(policy, "solver", None)
    if _solver is not None and hasattr(_solver, "set_task_context"):
        _solver.set_task_context(eval_episodes.tolist(), eval_start_idx.tolist())

    world.set_policy(policy)

    logger.info(f"Saving evaluation videos to {video_directory}")

    autocast_ctx = torch.autocast(
        device_type=device if device != "mps" else "cpu",
        dtype=torch.bfloat16,
        enabled=cfg.runtime.bfloat16,
    )

    if cfg.runtime.compile:
        logger.info("Warming up compiled model")
        warmup_autocast_ctx = torch.autocast(
            device_type=device if device != "mps" else "cpu",
            dtype=torch.bfloat16,
            enabled=cfg.runtime.bfloat16,
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.benchmark.goal_offset_steps,
                eval_budget=cfg.planning.budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=cast(
                    list[dict[Any, Any]] | None,
                    OmegaConf.to_container(cfg.benchmark.callables, resolve=True),
                ),
                video=video_directory,
            )
        logger.info("Compiled model warmup completed")

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.benchmark.goal_offset_steps,
            eval_budget=cfg.planning.budget,
            episodes_idx=eval_episodes.tolist(),
            callables=cast(
                list[dict[Any, Any]] | None,
                OmegaConf.to_container(cfg.benchmark.callables, resolve=True),
            ),
            video=video_directory,
        )
    end_time = time.time()

    logger.info(f"Evaluation metrics: {metrics}")
    logger.info(f"Videos saved to {video_directory}")

    results_path = Path(cfg.run.metrics) / cfg.output.filename
    results_path.write_text(f"metrics: {metrics}\nevaluation_time_seconds: {end_time - start_time}\n")
    logger.info(f"Evaluation results saved to {results_path}")


def main() -> object:
    return run_hydra(dispatch, config_dir="inference", config_name="evaluate")


if __name__ == "__main__":
    main()
