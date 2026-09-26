"""The physics-grounding term for the planner trainer, calibrated on the offline data.

:class:`rp1.core.agent.value.grounding.GroundingPenalty` needs, for every planning
problem, the real agent position at the plan's start and the command of the
primitive step before it. Both come from the dataset's ``state`` column and the
action h5, row-aligned with the dense latent cache.
"""

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from rp1.core.agent.value.grounding import ACTION_SCALE, calibrate_pusht_grounding
from rp1.data import LatentCache
from rp1.utils.logging import logger


class Grounding:
    """The calibrated penalty and the offline states and commands that anchor it."""

    def __init__(
        self,
        a: DictConfig,
        dense: LatentCache,
        actions: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        device: str,
    ) -> None:
        if a.grounding != "pusht":
            raise ValueError(f"unsupported grounding {a.grounding!r}; expected 'pusht'")
        if not a.state_h5:
            raise ValueError("grounding needs state_h5, the dataset h5 with its 'state' column")
        rows = int(dense.z.shape[0])
        with h5py.File(a.state_h5, "r") as file:
            episodes = np.asarray(file["episode_idx"][:rows]).reshape(-1)
            if len(episodes) != rows or not np.array_equal(episodes, dense.episode_idx.numpy()):
                raise RuntimeError(f"{a.state_h5} rows do not align with the dense cache")
            self.state = np.asarray(file["state"][:], dtype=np.float32)
        if self.state.shape[0] < actions.shape[0]:
            raise RuntimeError(f"{a.state_h5} has fewer rows ({self.state.shape[0]}) than the action h5")
        # the probe and thresholds describe the encoder and the environment, so they are
        # calibrated on the whole dense cache whatever episode cap the learners train under
        penalty, report = calibrate_pusht_grounding(
            dense.z,
            self.state[:rows],
            actions[:rows],
            dense.episode_idx.numpy(),
            amu=mean,
            astd=std,
            frameskip=a.frameskip,
            horizon=a.horizon,
            margin=a.ground_margin,
            tau=a.ground_tau,
            ref=a.ground_ref,
            weight=a.ground_weight,
            deadzone=a.ground_deadzone,
            quantile=a.ground_quantile,
            seed=a.seed,
        )
        logger.info("Grounding calibrated: " + " ".join(f"{key}={value:.4g}" for key, value in report.items()))
        self.penalty = penalty.to(device)
        self.commands = np.nan_to_num(actions).astype(np.float32) * ACTION_SCALE
        self.device = device

    def anchors(self, rows: np.ndarray, has_previous: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Agent position (px) at each plan's first action-h5 row, and the command (px) before it."""
        agent = torch.from_numpy(self.state[rows, :2]).to(self.device)
        previous = np.where(has_previous[:, None], self.commands[np.maximum(rows - 1, 0)], 0.0)
        return agent, torch.from_numpy(previous.astype(np.float32)).to(self.device)


__all__ = ["Grounding"]
