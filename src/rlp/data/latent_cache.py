"""Latent cache: encode logged trajectories once with a frozen world model.

All three metric learners (regression / TD / contrastive) train on *latents*
``z_t = f(o_t)`` produced by a frozen encoder. Encoding is the only expensive
step, so we run it once over a logged dataset and persist a compact cache of
``(z, episode_idx, step_idx[, state])``. The learners then iterate the cache
cheaply on CPU/MPS.

The builder is intentionally decoupled from any specific world model via a
``featurizer`` callable, so the same code serves both the lightweight state-WM
and the pixel LeWM checkpoint.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rlp.logging import logger

from .protocols import Dataset, RowBatch


@dataclass
class LatentCache:
    """Container of per-step latents with episode/step bookkeeping.

    Attributes:
        z: ``(N, D)`` float32 latents in trajectory order.
        episode_idx: ``(N,)`` int64 episode id per row.
        step_idx: ``(N,)`` int64 within-episode timestep per row.
        state: optional ``(N, S)`` ground-truth task state (for the oracle).
        meta: free-form metadata (env id, wm name, latent dim, ...).
    """

    z: torch.Tensor
    episode_idx: torch.Tensor
    step_idx: torch.Tensor
    state: torch.Tensor | None = None
    meta: dict[str, object] | None = None

    @property
    def latent_dim(self) -> int:
        return self.z.shape[1]

    def episodes(self) -> dict[int, np.ndarray]:
        """Map each episode id to its row indices, sorted by ``step_idx``."""
        ep = self.episode_idx.numpy()
        st = self.step_idx.numpy()
        out: dict[int, np.ndarray] = {}
        for e in np.unique(ep):
            rows = np.nonzero(ep == e)[0]
            out[int(e)] = rows[np.argsort(st[rows])]
        return out

    def windowed(self, frames: int, lag: int = 1) -> LatentCache:
        """Return a cache whose latent rows concatenate causal frame windows.

        Missing history at an episode start is left-padded with that episode's
        first row, matching the planner-side window convention.
        """
        if frames < 1 or lag < 1:
            raise ValueError("frames and lag must be positive")
        if frames == 1:
            return self
        output = torch.empty((len(self.z), self.latent_dim * frames), dtype=torch.float32)
        offsets = np.arange(frames - 1, -1, -1) * lag
        for rows in self.episodes().values():
            positions = np.arange(len(rows))[:, None] - offsets[None]
            positions = np.maximum(positions, 0)
            source = torch.as_tensor(rows[positions], dtype=torch.long)
            output[torch.as_tensor(rows, dtype=torch.long)] = self.z[source].flatten(start_dim=1).float()
        metadata = dict(self.meta or {})
        metadata.update(window_frames=frames, window_lag=lag, source_latent_dim=self.latent_dim)
        return type(self)(output, self.episode_idx, self.step_idx, self.state, metadata)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "z": self.z.cpu(),
                "episode_idx": self.episode_idx.cpu(),
                "step_idx": self.step_idx.cpu(),
                "state": None if self.state is None else self.state.cpu(),
                "meta": self.meta or {},
            },
            path,
        )
        logger.success(f"Saved latent cache ({len(self.z)} rows, dim={self.latent_dim}) to {path}")

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool | None = None) -> LatentCache:
        """Load a cache, optionally mapping tensor storage instead of copying it.

        ``mmap=None`` follows ``RLP_CACHE_MMAP``. Callers can explicitly opt
        out for small caches or filesystems where mapping is undesirable.
        """
        if mmap is None:
            mmap = os.environ.get("RLP_CACHE_MMAP") == "1"
        d = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
        return cls(
            z=d["z"].float(),
            episode_idx=d["episode_idx"].long(),
            step_idx=d["step_idx"].long(),
            state=None if d.get("state") is None else d["state"].float(),
            meta=d.get("meta", {}),
        )


def _episode_col(dataset: Dataset) -> np.ndarray:
    """Fetch the per-row episode index, robust to lance hiding it from
    ``column_names`` (try ``episode_idx`` then ``ep_idx``)."""
    for name in ("episode_idx", "ep_idx"):
        try:
            return np.asarray(dataset.get_col_data(name))
        except (KeyError, ValueError, NotImplementedError):
            continue
    raise KeyError("dataset exposes neither 'episode_idx' nor 'ep_idx'")


def encode_dataset(
    dataset: Dataset,
    featurizer: Callable[[RowBatch], torch.Tensor],
    *,
    batch_size: int = 256,
    state_key: str | None = None,
    meta: dict[str, object] | None = None,
) -> LatentCache:
    """Encode every row of ``dataset`` into a :class:`LatentCache`.

    Args:
        dataset: a stable-worldmodel dataset exposing ``get_col_data`` and
            ``get_row_data``.
        featurizer: maps a batch of raw rows (dict of numpy arrays) to a
            ``(B, D)`` latent tensor. Encapsulates the frozen WM + transforms.
        batch_size: rows per encode call.
        state_key: optional column to store as ground-truth state for the oracle.
        meta: metadata to attach to the cache.
    """
    # NB: lance hides episode_idx/step_idx from ``column_names`` even though
    # ``get_col_data`` serves them, so probe directly rather than membership-test.
    episode_idx = _episode_col(dataset).reshape(-1).astype(np.int64)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    n = len(episode_idx)

    z_chunks: list[torch.Tensor] = []
    state_chunks: list[np.ndarray] = []
    total_batches = max(1, (n + batch_size - 1) // batch_size)
    log_interval = max(1, total_batches // 20)
    for batch_index, start in enumerate(range(0, n, batch_size), start=1):
        idx = list(range(start, min(start + batch_size, n)))
        rows = dataset.get_row_data(idx)
        with torch.no_grad():
            z = featurizer(rows).float().cpu()
        z_chunks.append(z)
        if state_key is not None:
            state_chunks.append(np.asarray(rows[state_key]).reshape(len(idx), -1))
        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
            logger.info(f"Encoding latents batch={batch_index}/{total_batches} rows={min(start + batch_size, n)}/{n}")

    z = torch.cat(z_chunks, dim=0)
    state = torch.from_numpy(np.concatenate(state_chunks, axis=0)).float() if state_key is not None else None
    return LatentCache(
        z=z,
        episode_idx=torch.from_numpy(episode_idx),
        step_idx=torch.from_numpy(step_idx),
        state=state,
        meta=meta or {},
    )


__all__ = ["LatentCache", "encode_dataset"]
