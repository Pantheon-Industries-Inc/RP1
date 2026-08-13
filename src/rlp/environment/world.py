"""RLP's dataset-evaluation world behavior.

This extension still imports Stable-WM's private dataset helpers because 0.1.1
does not expose recording or resize hooks. Keep the imports isolated here so a
future public upstream evaluation hook can replace this subclass directly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from PIL import Image
from stable_worldmodel import World as _World
from stable_worldmodel.plot import save_panel_videos
from stable_worldmodel.world.world import _apply_callables, _extract_init_goal

from rlp.logging import logger


class EvaluationResult(TypedDict):
    success_rate: float
    episode_successes: np.ndarray
    seeds: object


def _resize_images_like_env(images: np.ndarray, env_pixels: np.ndarray) -> np.ndarray:
    target_shape = env_pixels.shape[2:]
    if images.shape[1:] == target_shape:
        return images
    target_h, target_w = target_shape[:2]
    resized = [
        np.asarray(Image.fromarray(image).resize((target_w, target_h), Image.Resampling.BILINEAR)) for image in images
    ]
    return np.stack(resized).astype(images.dtype, copy=False)


class World(_World):
    """Stable-WM World with reproducible RLP dataset reset and recording."""

    def __init__(self, env_name: str, *args: Any, **kwargs: Any) -> None:
        self.record_path = kwargs.pop("record_path", None)
        super().__init__(env_name, *args, **kwargs)

    def _evaluate_from_dataset(
        self,
        dataset: Any,
        episodes_idx: Sequence[int],
        start_steps: Sequence[int],
        goal_offset: int,
        eval_budget: int,
        callables: dict[str, Callable[..., Any]] | None,
        video: str | Path | None,
        mode: str,
    ) -> EvaluationResult:
        n = len(episodes_idx)
        if n != self.num_envs:
            raise ValueError(f"{n} dataset episodes for {self.num_envs} environments")
        init_state, goal_state, dataset_videos = _extract_init_goal(dataset, episodes_idx, start_steps, goal_offset)
        self.reset(seed=init_state.get("seed"))
        if callables:
            merged = {**init_state, **goal_state}
            for i in range(n):
                env_init = {key: value[i] for key, value in merged.items()}
                _apply_callables(self.envs.envs[i].unwrapped, callables, env_init)

        if "pixels" in self.infos:
            for mapping, key in ((init_state, "pixels"), (goal_state, "goal")):
                if key in mapping:
                    mapping[key] = _resize_images_like_env(mapping[key], self.infos["pixels"])
            dataset_videos = [_resize_images_like_env(item, self.infos["pixels"]) for item in dataset_videos]
            shape_prefix = self.infos["pixels"].shape[:2]
        else:
            shape_prefix = next(
                value.shape[:2]
                for value in self.infos.values()
                if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 2
            )
        for source in (init_state, goal_state):
            for key, value in source.items():
                if key in self.infos or key in goal_state:
                    self.infos[key] = np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:]).copy()

        goal_snapshot = {key: self.infos[key].copy() for key in goal_state}
        record_path = self.record_path
        record_cols = ("pixels", "action", "qpos", "qvel")
        record_buffers: list[defaultdict[str, list[np.ndarray]]] | None = (
            [defaultdict(list) for _ in range(n)] if record_path else None
        )
        record_done = np.zeros(n, dtype=bool)
        results: EvaluationResult = {
            "success_rate": 0.0,
            "episode_successes": np.zeros(n, dtype=bool),
            "seeds": init_state.get("seed"),
        }
        frames: defaultdict[int, list[np.ndarray]] | None = defaultdict(list) if video else None

        def on_step(world: World) -> None:
            world.infos.update(deepcopy(goal_snapshot))
            if record_buffers is not None:
                for column in record_cols:
                    if column not in world.infos:
                        continue
                    data = world.infos[column]
                    if not isinstance(data, (np.ndarray, torch.Tensor)):
                        continue
                    if data.ndim > 1 and data.shape[1] == 1:
                        data = data.squeeze(1)
                    for i in range(n):
                        if not record_done[i]:
                            value = data[i]
                            if torch.is_tensor(value):
                                value = value.detach().cpu().numpy()
                            record_buffers[i][column].append(np.asarray(value).copy())
                if world.terminateds is None or world.truncateds is None:
                    raise RuntimeError("world step did not populate termination arrays")
                record_done[:] |= world.terminateds | world.truncateds
            if world.terminateds is None:
                raise RuntimeError("world step did not populate termination flags")
            results["episode_successes"] |= world.terminateds
            if frames is not None:
                for i in range(n):
                    frame = world.infos["pixels"][i]
                    frames[i].append(np.asarray(frame[-1] if frame.ndim > 3 else frame).copy())

        self._run(max_steps=eval_budget, mode=mode, on_step=on_step)
        if record_buffers is not None:
            from stable_worldmodel.data.format import get_format

            stats = {"kept": 0, "dropped": 0}

            def episodes() -> Iterator[dict[str, list[np.ndarray]]]:
                for buffer in record_buffers:
                    episode = {key: list(values) for key, values in buffer.items()}
                    if len(episode.get("action", ())) < 25:
                        stats["dropped"] += 1
                        continue
                    episode["action"].append(episode["action"].pop(0))
                    stats["kept"] += 1
                    yield episode

            with get_format("lance").open_writer(record_path) as writer:
                writer.write_episodes(episodes())
            logger.info(f"Recorded dataset kept={stats['kept']} dropped={stats['dropped']} path={record_path}")

        results["success_rate"] = float(results["episode_successes"].sum()) / n * 100.0
        if frames and video is not None:
            save_panel_videos(
                Path(video),
                {"agent": frames, "dataset": dataset_videos, "goal": goal_state["goal"]},
            )
        return results


__all__ = ["World"]
