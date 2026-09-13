"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "egl")

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from stable_worldmodel.policy import BasePolicy
from torch import nn
from torchvision.transforms import v2 as transforms

from rlp.config import dispatch, run_hydra
from rlp.core.policy import NoMovePolicy, PlanConfig, WorldModelPolicy
from rlp.core.value import as_planning_cost
from rlp.core.world_model import load_pretrained
from rlp.core.world_model.runtime import pick_device
from rlp.data.protocols import Array, Dataset
from rlp.environment import World
from rlp.logging import logger


def img_transform(cfg: DictConfig, dtype: torch.dtype = torch.float32) -> Callable[[object], torch.Tensor]:
    stats = cast(dict[str, Sequence[float]], spt.data.dataset_stats.ImageNet)
    steps = [
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ]
    # eval.train_res: bottleneck images through the checkpoint's native
    # training resolution (e.g. 64 for OGBench play retrains, whose 64px
    # frames were upscaled to 224 during training) so the model sees the
    # image domain it was trained on. null/img_size = no-op.
    train_res = cfg.evaluation.train_resolution
    if train_res and int(train_res) != int(cfg.evaluation.image_size):
        steps.append(transforms.Resize(size=int(train_res)))
    steps.append(transforms.Resize(size=cfg.evaluation.image_size))
    return cast(Callable[[object], torch.Tensor], transforms.Compose(steps))


def episode_col(dataset: Dataset) -> str:
    """Episode-index column name. Lance keeps 'episode_idx'/'step_idx' as
    writer-managed index columns outside column_names but serves them via
    get_col_data; h5 eval sets list 'ep_idx' (or 'episode_idx') explicitly."""
    return "ep_idx" if "ep_idx" in dataset.column_names else "episode_idx"


def get_episodes_length(dataset: Dataset, episodes: Array) -> Array:
    col_name = episode_col(dataset)

    # lance serves index columns as (N,1); h5 as (N,). Flatten so the boolean
    # mask below is 1-D regardless of source format.
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    lengths: list[int] = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg: DictConfig, dataset_name: str) -> Dataset:
    raw_dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.data.cache_dir,
        keys_to_cache=list(cfg.data.keys_to_cache),
    )
    if not callable(getattr(raw_dataset, "get_col_data", None)) or not hasattr(raw_dataset, "column_names"):
        raise TypeError(f"dataset {dataset_name!r} does not provide the expected column interface")
    return cast(Dataset, raw_dataset)


def _run(cfg: DictConfig) -> None:
    """Run evaluation of dinowm vs random policy."""
    if cfg.planning.horizon * cfg.planning.action_block > cfg.evaluation.budget:
        raise ValueError("planning horizon x action block must not exceed the evaluation budget")
    device = pick_device(cfg.runtime.device)

    # create world environment
    environment = cast(dict[str, Any], OmegaConf.to_container(cfg.environment, resolve=True))
    environment["max_episode_steps"] = 2 * cfg.evaluation.budget
    world = World(
        **environment,
        image_shape=(cfg.evaluation.image_size, cfg.evaluation.image_size),
        # planning.history_len used to be a dead key (nothing stacked frames from it);
        # it now sets the WM history published as `pixels_hist`, one action block apart
        history_frames=int(cfg.planning.get("history_len", 1) or 1),
        history_lag=int(cfg.planning.action_block),
    )

    # create the transform
    img_dtype = torch.bfloat16 if cfg.runtime.bfloat16 else torch.float32
    transform: dict[str, Callable[[object], torch.Tensor]] = {
        "pixels": img_transform(cfg, img_dtype),
        "pixels_hist": img_transform(cfg, img_dtype),  # same normalise + CHW permute as `pixels`
        "goal": img_transform(cfg, img_dtype),
    }

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
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_range = cfg.evaluation.episode_range
    if episode_range:
        parts = str(episode_range).split(":")
        if len(parts) != 2:
            raise ValueError("evaluation.episode_range must use LO:HI syntax")
        low, high = map(int, parts)
        if low < 0 or high <= low:
            raise ValueError("evaluation.episode_range must satisfy 0 <= LO < HI")
        ep_indices = ep_indices[(ep_indices >= low) & (ep_indices < high)]
        if len(ep_indices) < cfg.evaluation.num_episodes:
            raise ValueError(
                f"episode range {episode_range} has {len(ep_indices)} episodes; need {cfg.evaluation.num_episodes}"
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

    # -- run evaluation
    policy_kind = cfg.core.policy.kind

    if policy_kind == "world_model":
        model = load_pretrained(cfg.core.policy.checkpoint)
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
        metric_paths = list(cfg.core.value.checkpoints)
        if cfg.core.value.kind == "metric":
            from rlp.core.value import load_metric

            loaded_metrics = [load_metric(path, device=device) for path in metric_paths]
            if not loaded_metrics:
                raise ValueError("metric planning requires at least one checkpoint")
            from rlp.core.value import MetricCost

            cost_model = MetricCost(
                cost_model,
                loaded_metrics[0],
                cfg.core.value.mode,
                lam=float(cfg.core.value.blend_weight),
                metrics=loaded_metrics,
                deadline_mode=str(cfg.core.value.deadline_mode),
            )
            logger.info(f"Plan-score metrics={metric_paths} mode={cfg.core.value.mode}")

        solver = hydra.utils.instantiate(
            cfg.core.solver,
            model=cost_model,
            device=device,
            seed=cfg.runtime.seed,
        )
        policy: BasePolicy = WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    else:
        policy = NoMovePolicy() if policy_kind == "no_move" else swm.policy.RandomPolicy()

    video_directory = Path(cfg.run.videos)

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.evaluation.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row's episode_idx to its max_start_idx (flatten index
    # columns: lance serves them as (N,1), h5 as (N,)).
    _row_epi = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    _row_step = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    max_start_per_row = np.full(_row_epi.shape, -1, dtype=np.int64)
    for ep_id, maximum in max_start_idx_dict.items():
        max_start_per_row[_row_epi == ep_id] = maximum

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = (max_start_per_row >= 0) & (_row_step <= max_start_per_row)
    valid_indices = np.nonzero(valid_mask)[0]
    logger.info(f"Found {int(valid_mask.sum())} valid evaluation starting points")

    # optional hard-task filter (TwoRoom): keep only (start, goal) pairs on
    # opposite sides of the dividing wall (wall center 112; axis inferred
    # from the first door center in the observation vector)
    if cfg.evaluation.cross_wall:
        st_all = np.asarray(dataset.get_col_data("state"))
        first_door = np.asarray(dataset.get_col_data("observation"))[0, 4:6]
        axis = 0 if abs(float(first_door[0]) - 112.0) < 1e-3 else 1
        off = cfg.evaluation.goal_offset_steps
        s0 = st_all[valid_indices]
        s1 = st_all[valid_indices + off]
        cross = np.sign(s0[:, axis] - 112.0) != np.sign(s1[:, axis] - 112.0)
        valid_indices = valid_indices[cross]
        logger.info(f"Found {len(valid_indices)} cross-wall starting points on wall axis {axis}")

    if len(valid_indices) < cfg.evaluation.num_episodes:
        raise ValueError(f"Need {cfg.evaluation.num_episodes} valid evaluation starts; found {len(valid_indices)}")
    g = np.random.default_rng(cfg.runtime.seed)
    # Sort increasingly to avoid issues with HDF5Dataset indexing.
    random_episode_indices = np.sort(g.choice(valid_indices, size=cfg.evaluation.num_episodes, replace=False))

    logger.info(f"Selected evaluation row indices: {random_episode_indices.tolist()}")

    # index columns may be writer-managed (lance): use column access, not rows.
    # Flatten (lance serves (N,1)) and cast to int (lance stores these as
    # float32; the reader indexes offsets with them and requires int).
    eval_episodes = np.asarray(dataset.get_col_data(col_name)).reshape(-1)[random_episode_indices].astype(np.int64)
    eval_start_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)[random_episode_indices].astype(np.int64)

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
                goal_offset=cfg.evaluation.goal_offset_steps,
                eval_budget=cfg.evaluation.budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(cfg.evaluation.callables, resolve=True),
                video=video_directory,
            )
        logger.info("Compiled model warmup completed")

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.evaluation.goal_offset_steps,
            eval_budget=cfg.evaluation.budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.evaluation.callables, resolve=True),
            video=video_directory,
        )
    end_time = time.time()

    logger.info(f"Evaluation metrics: {metrics}")
    logger.info(f"Videos saved to {video_directory}")

    results_path = Path(cfg.run.metrics) / cfg.output.filename
    results_path.write_text(f"metrics: {metrics}\nevaluation_time_seconds: {end_time - start_time}\n")
    logger.info(f"Evaluation results saved to {results_path}")


def run() -> object:
    """Launch world-model evaluation with the repository-level Hydra config."""
    return run_hydra(dispatch, config_name="eval/lewm")


if __name__ == "__main__":
    run()
