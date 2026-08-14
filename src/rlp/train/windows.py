"""Latent-window sampler for planner training on a frozen world model.

Mirrors the sampling contract of :mod:`rlp.train.lip_ac` — three-frame latent
history from a frameskip-matched (fs5) cache, the two preceding real action
blocks from the action h5, and a goal drawn either from the same episode
within ``max_delta`` blocks or (with probability ``p_cross``) from a different
episode. ``lip_ac`` keeps its own inline copy so the paper pipeline's RNG
stream stays frozen; this module serves the baseline trainers
(:mod:`rlp.train.dmpo`).

The h5 read is nan-aware: several public datasets (cube, reacher) pad every
episode's terminal step with NaN actions, and plain ``mean``/``std`` would
poison every normalized action (see the reacher caveat in the replication
sheet).
"""

from contextlib import suppress
from dataclasses import dataclass

import h5py

with suppress(ImportError):
    import hdf5plugin  # noqa: F401  (registers HDF5 compression filters, e.g. cube h5)

import numpy as np
import torch

from rlp.data import LatentCache

__all__ = ["WindowBatch", "WindowSampler"]


@dataclass(frozen=True)
class WindowBatch:
    """One training batch of planning problems."""

    z_hist: torch.Tensor  # (B, 3, D) latent history
    a_hist: torch.Tensor  # (B, 2, a_dim) preceding real action blocks
    z_goal: torch.Tensor  # (B, D) goal latent
    a_ref: torch.Tensor  # (B, H, a_dim) the data's next-H real action blocks


class WindowSampler:
    """Draw planning problems from a latent cache plus its action h5."""

    def __init__(
        self,
        cache: str,
        h5: str,
        horizon: int,
        max_delta: int = 10,
        p_cross: float = 0.3,
        frameskip: int = 5,
        device: str | torch.device = "cpu",
        mmap: bool = True,
        seed: int = 0,
    ) -> None:
        self.horizon = int(horizon)
        self.max_delta = int(max_delta)
        self.p_cross = float(p_cross)
        self.frameskip = int(frameskip)
        self.device = device
        self.rng = np.random.default_rng(seed)

        latents = LatentCache.load(cache, mmap=mmap)
        self.z = latents.z.to(device).float()
        episodes = latents.episodes()
        keys = [key for key in episodes if len(episodes[key]) > self.max_delta + 4]
        if not keys:
            raise ValueError(f"no episode in {cache} is longer than max_delta + 4 = {self.max_delta + 4}")
        self.ep_rows = {episode: np.asarray(rows) for episode, rows in episodes.items() if episode in set(keys)}
        self.ep_ids = np.array(keys)

        with h5py.File(h5, "r") as handle:
            actions = handle["action"][:]
            self.ep_off = handle["ep_offset"][:]
            self.ep_len = handle["ep_len"][:] if "ep_len" in handle else None
        self.action_mean = np.nanmean(actions, 0)
        self.action_std = np.nanstd(actions, 0) + 1e-6
        self.actions = ((actions - self.action_mean) / self.action_std).astype(np.float32)
        self.a_dim = int(actions.shape[-1]) * self.frameskip

    @property
    def latent_dim(self) -> int:
        return int(self.z.shape[-1])

    def _block(self, episode: int, index: int) -> np.ndarray:
        offset = self.frameskip * index
        if self.ep_len is not None:
            offset = min(offset, max(0, int(self.ep_len[episode]) - self.frameskip))
        start = int(self.ep_off[episode] + offset)
        return np.asarray(self.actions[start : start + self.frameskip]).reshape(-1)

    def sample(self, batch: int) -> WindowBatch:
        z_hist: list[torch.Tensor] = []
        a_hist: list[np.ndarray] = []
        z_goal: list[torch.Tensor] = []
        a_ref: list[np.ndarray] = []
        for _ in range(batch):
            episode = int(self.ep_ids[self.rng.integers(len(self.ep_ids))])
            rows = self.ep_rows[episode]
            length = len(rows)
            t = int(self.rng.integers(2, length - 2))
            z_hist.append(torch.stack([self.z[rows[t - 2]], self.z[rows[t - 1]], self.z[rows[t]]]))
            a_hist.append(np.stack([self._block(episode, t - 2), self._block(episode, t - 1)]))
            # clamp at length - 2, the last FULL in-episode block: block
            # length - 1 starts at the episode's final primitive step, so its
            # frameskip-long read spills into the next episode.
            a_ref.append(np.stack([self._block(episode, min(t + k, length - 2)) for k in range(self.horizon)]))
            if self.rng.random() < self.p_cross:
                other = int(self.ep_ids[self.rng.integers(len(self.ep_ids))])
                other_rows = self.ep_rows[other]
                z_goal.append(self.z[other_rows[self.rng.integers(len(other_rows))]])
            else:
                delta = int(self.rng.integers(1, self.max_delta + 1))
                z_goal.append(self.z[rows[min(t + delta, length - 1)]])
        return WindowBatch(
            z_hist=torch.stack(z_hist),
            a_hist=torch.from_numpy(np.stack(a_hist)).to(self.device),
            z_goal=torch.stack(z_goal),
            a_ref=torch.from_numpy(np.stack(a_ref)).to(self.device).float(),
        )
