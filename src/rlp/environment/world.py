"""RLP's dataset-evaluation world behavior.

This extension still imports Stable-WM's private dataset helpers because 0.1.1
does not expose recording or resize hooks. Keep the imports isolated here so a
future public upstream evaluation hook can replace this subclass directly.
"""

from __future__ import annotations

import os
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
        # First-hit scoring at an explicit tolerance, replacing the
        # environment's own termination test. Reacher's qpos-match task
        # hardcodes a 0.05 rad threshold, so the tau=0.1 column of the paper's
        # Reacher table is unreachable through termination alone; with this set
        # success latches on the first step whose worst joint is within
        # `success_threshold` of the goal state.
        self.success_threshold: float | None = kwargs.pop("success_threshold", None)
        self.success_key: str = kwargs.pop("success_key", "qpos")
        if self.success_threshold is not None:
            self.success_threshold = float(self.success_threshold)
        # World-model history at plan time. Upstream infos carry ONE frame, so a
        # planner that wants the WM's 3-frame window pads by tiling the current
        # frame and zeroing the action history -- at every replan, not just the
        # episode start (PUSHT_DIAG E29). With history_frames=K this publishes
        # `pixels_hist` (n, K, H, W, C): the frames at lags 0, lag, 2*lag, ...
        # (lag = the action block, so they are one WM step apart), and
        # `action_hist` (n, K-1, lag, a): the primitive actions of the K-1 past
        # blocks. Both are padded with the earliest available frame / zeros at
        # the episode start. Separate keys, so the upstream CEM path -- which
        # splits `pixels.size(2)` actions off its own candidate -- is untouched.
        self.history_frames: int = int(kwargs.pop("history_frames", 0) or 0)
        self.history_lag: int = int(kwargs.pop("history_lag", 5) or 5)
        super().__init__(env_name, *args, **kwargs)

    @staticmethod
    def _final_frame(value: np.ndarray) -> np.ndarray:
        """Drop the per-env history axis, keeping the current step."""
        return value[:, -1] if value.ndim > 2 else value

    @classmethod
    def threshold_hits(cls, current: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
        """Per-environment first-hit test: worst coordinate within ``threshold``."""
        deviation = np.abs(
            cls._final_frame(np.asarray(current, dtype=np.float64))
            - cls._final_frame(np.asarray(target, dtype=np.float64))
        )
        return np.asarray(deviation.max(axis=-1) < threshold)

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
        if self.success_threshold is not None:
            missing = [key for key in (self.success_key, f"goal_{self.success_key}") if key not in self.infos]
            if missing:
                raise KeyError(f"first-hit scoring needs {missing} in the evaluation state")
            logger.info(f"First-hit scoring on |{self.success_key} - goal| < {self.success_threshold}")
        record_path = self.record_path
        # pos_agent/block_pose are PushT's exact pose fields: recording them lets an
        # analysis evaluate the env's own success test along a rollout instead of
        # decoding poses from latents (a probe's error exceeds the 20 px tolerance)
        record_cols = ("pixels", "action", "qpos", "qvel", "pos_agent", "block_pose")
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
        # lagged history buffers: the last (K-1)*lag+1 frames and (K-1)*lag actions per env
        hist_k, hist_lag = self.history_frames, self.history_lag
        hist_px: list[list[np.ndarray]] = [[] for _ in range(n)]
        hist_ac: list[list[np.ndarray]] = [[] for _ in range(n)]

        def publish_history(world: World) -> None:
            px_now = self._final_frame(np.asarray(world.infos["pixels"]))  # (n, H, W, C)
            ac_now = np.asarray(world.infos["action"])
            ac_now = ac_now[:, -1] if ac_now.ndim > 2 else ac_now  # (n, a)
            span = (hist_k - 1) * hist_lag
            out_px = np.empty((n, hist_k) + px_now.shape[1:], dtype=px_now.dtype)
            out_ac = np.zeros((n, hist_k - 1, hist_lag, ac_now.shape[-1]), dtype=np.float32)
            for i in range(n):
                hist_px[i].append(px_now[i])
                hist_px[i] = hist_px[i][-(span + 1) :]
                hist_ac[i].append(ac_now[i])
                hist_ac[i] = hist_ac[i][-span:] if span else []
                buf = hist_px[i]
                for j in range(hist_k):  # j = 0 oldest ... K-1 newest, lagged by `lag`
                    idx = len(buf) - 1 - (hist_k - 1 - j) * hist_lag
                    out_px[i, j] = buf[max(idx, 0)]
                acs = np.asarray(hist_ac[i], dtype=np.float32)
                if len(acs):  # right-align the primitive actions we have into the block slots
                    out_ac[i].reshape(-1, ac_now.shape[-1])[-len(acs) :] = acs
            world.infos["pixels_hist"] = out_px
            world.infos["action_hist"] = out_ac

        def on_step(world: World) -> None:
            world.infos.update(deepcopy(goal_snapshot))
            if hist_k > 1 and "pixels" in world.infos and "action" in world.infos:
                publish_history(world)
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
            if self.success_threshold is None:
                if world.terminateds is None:
                    raise RuntimeError("world step did not populate termination flags")
                results["episode_successes"] |= world.terminateds
            else:
                results["episode_successes"] |= self.threshold_hits(
                    world.infos[self.success_key],
                    goal_snapshot[f"goal_{self.success_key}"],
                    self.success_threshold,
                )
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
                    if len(episode.get("action", ())) < int(os.environ.get("RLP_RECORD_MIN_LEN", "25")):
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
