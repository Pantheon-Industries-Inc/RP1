"""Small shared utilities used by multiple training entrypoints."""

from __future__ import annotations

import math

import numpy as np
import torch


def cosine_interpolate(base: float, final: float | None, step: int, total: int) -> float:
    """Cosine interpolation from ``base`` to ``final`` over ``total`` steps."""
    if final is None or total <= 0:
        return base
    step = min(step, total)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * step / total))


def sample_windows(
    episodes: dict[int, np.ndarray],
    values: np.ndarray | torch.Tensor,
    actions: np.ndarray | torch.Tensor,
    window: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample aligned contiguous value/action windows from episode rows."""
    valid = [rows for rows in episodes.values() if len(rows) >= window]
    if not valid:
        raise ValueError(f"no episode contains a window of length {window}")
    value_out = torch.empty((batch_size, window, values.shape[1]), dtype=torch.float32)
    action_out = torch.empty((batch_size, window, actions.shape[1]), dtype=torch.float32)
    for batch_index in range(batch_size):
        rows = valid[int(rng.integers(len(valid)))]
        start = int(rng.integers(len(rows) - window + 1))
        indices = np.asarray(rows[start : start + window])
        value_out[batch_index] = torch.as_tensor(values[indices], dtype=torch.float32)
        action_out[batch_index] = torch.as_tensor(actions[indices], dtype=torch.float32)
    return value_out, action_out


__all__ = ["cosine_interpolate", "sample_windows"]
