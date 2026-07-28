"""Trajectory pair/transition samplers over a :class:`LatentCache`.

The paper emphasises that "sampling is part of the method, not an implementation
detail": horizon-matched supervision draws *balanced, full-horizon* temporal
separations so the metric sees the same long-range reachability scale that the
terminal selector faces at planning time.

Three samplers, one per learner:

* :class:`BalancedHorizonPairSampler` -- regression. Pairs ``(z_i, z_j)`` with
  label ``|t_i - t_j|``; separations balanced across buckets up to the full
  episode horizon (or a ``max_delta`` cap, for the paper's ablation).
* :class:`TransitionSampler` -- offline TD. Logged transitions ``(z_t, z_{t+1})``
  with a hindsight goal ``z_g`` (a later state in the same episode) and a
  ``done`` flag (goal reached at ``t+1``).
* :class:`GeometricFutureSampler` -- contrastive. Anchor ``z_s`` and positive
  future ``z_{s+k}`` with ``k ~ Geom(1 - gamma)``; negatives are in-batch.
"""

from __future__ import annotations

import numpy as np
import torch

from .latent_cache import LatentCache


class _BaseSampler:
    def __init__(self, cache: LatentCache, seed: int = 0, min_len: int = 2):
        self.cache = cache
        self.z = cache.z
        self.rng = np.random.default_rng(seed)
        eps = cache.episodes()
        # keep only episodes long enough to form a pair/transition
        self.episodes = {e: rows for e, rows in eps.items() if len(rows) >= min_len}
        self.ep_ids = np.array(sorted(self.episodes.keys()))
        self.ep_lens = np.array([len(self.episodes[e]) for e in self.ep_ids])
        self.max_len = int(self.ep_lens.max())
        assert len(self.ep_ids) > 0, "no episodes long enough to sample"


class BalancedHorizonPairSampler(_BaseSampler):
    """Balanced full-horizon temporal-separation pairs (regression supervision).

    Args:
        n_buckets: number of separation buckets used to equalise coverage.
        max_delta: optional cap on the temporal separation (the paper's
            ``max-Delta`` ablation; ``None`` = full episode horizon).
        random_order: if ``True`` randomly swap ``(z_i, z_j)`` per pair so the
            head is encouraged to be symmetric.
    """

    def __init__(
        self,
        cache: LatentCache,
        n_buckets: int = 10,
        max_delta: int | None = None,
        random_order: bool = True,
        seed: int = 0,
    ):
        super().__init__(cache, seed=seed, min_len=2)
        self.n_buckets = n_buckets
        self.max_delta = max_delta
        self.random_order = random_order

    def _sample_delta(self, L: int) -> int:
        hi = L - 1
        if self.max_delta is not None:
            hi = min(hi, self.max_delta)
        if hi < 1:
            return 1
        # balanced over buckets: choose a bucket overlapping [1, hi], then uniform within
        edges = np.linspace(1, hi + 1, self.n_buckets + 1)
        b = self.rng.integers(0, self.n_buckets)
        lo_b, hi_b = edges[b], edges[b + 1]
        delta = int(self.rng.integers(int(np.floor(lo_b)), max(int(np.ceil(hi_b)), int(np.floor(lo_b)) + 1)))
        return int(np.clip(delta, 1, hi))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        i_idx = np.empty(batch_size, dtype=np.int64)
        j_idx = np.empty(batch_size, dtype=np.int64)
        labels = np.empty(batch_size, dtype=np.float32)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            delta = self._sample_delta(L)
            t = int(self.rng.integers(0, L - delta))
            ri, rj = rows[t], rows[t + delta]
            if self.random_order and self.rng.random() < 0.5:
                ri, rj = rj, ri
            i_idx[b], j_idx[b], labels[b] = ri, rj, delta
        return {
            "z_i": self.z[i_idx],
            "z_j": self.z[j_idx],
            "label": torch.from_numpy(labels),
        }


class TransitionSampler(_BaseSampler):
    """Logged transitions with hindsight goals for offline TD.

    Returns ``z_t, z_tp1, z_g`` and a ``done`` flag (goal reached at ``t+1``).
    Goals are later states in the same episode; a fraction are random states
    from any episode (treated as not-done) for negative coverage.
    """

    def __init__(self, cache: LatentCache, p_random_goal: float = 0.0, seed: int = 0):
        super().__init__(cache, seed=seed, min_len=2)
        self.p_random_goal = p_random_goal
        self.n = len(cache.z)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        t_idx = np.empty(batch_size, dtype=np.int64)
        tp1_idx = np.empty(batch_size, dtype=np.int64)
        g_idx = np.empty(batch_size, dtype=np.int64)
        done = np.zeros(batch_size, dtype=np.float32)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            t = int(self.rng.integers(0, L - 1))
            t_idx[b], tp1_idx[b] = rows[t], rows[t + 1]
            if self.rng.random() < self.p_random_goal:
                g_idx[b] = int(self.rng.integers(0, self.n))
                done[b] = 0.0
            else:
                # hindsight goal: a state at or after t+1 in the same episode
                g_off = int(self.rng.integers(t + 1, L))
                g_idx[b] = rows[g_off]
                done[b] = 1.0 if g_off == t + 1 else 0.0
        return {
            "z_t": self.z[t_idx],
            "z_tp1": self.z[tp1_idx],
            "z_g": self.z[g_idx],
            "done": torch.from_numpy(done),
        }


class NStepGoalSampler(_BaseSampler):
    """HER n-step transitions with balanced full-horizon hindsight goals.

    For each draw returns ``(z_t, z_tn, z_g, n_eff, reached, dist)`` where:
      * ``z_tn`` is the state ``n_eff = min(n, L-1-t)`` steps ahead (n-step bootstrap target),
      * ``z_g`` is a **hindsight goal**: with prob ``p_cross`` a random cross-episode state
        (enables Bellman stitching of long/cross-room pairs), else a future state in the
        same episode at a **balanced full-horizon** offset ``δ`` (paper's lesson applied to HER),
      * ``reached`` / ``dist``: if the in-episode goal is within ``n_eff`` steps, the exact
        distance ``δ`` is known (Monte-Carlo target); otherwise bootstrap from ``z_tn``.
    """

    def __init__(self, cache, n_step=1, p_cross=0.2, n_buckets=10, balanced=True, seed=0, max_delta=None):
        super().__init__(cache, seed=seed, min_len=2)
        self.n = n_step
        self.max_delta = max_delta
        self.p_cross = p_cross
        self.n_buckets = n_buckets
        self.balanced = balanced
        self.n_total = len(cache.z)

    def _offset(self, hi):
        if hi < 1:
            return 1
        if not self.balanced:
            return int(self.rng.integers(1, hi + 1))
        edges = np.linspace(1, hi + 1, self.n_buckets + 1)
        b = self.rng.integers(0, self.n_buckets)
        lo_b, hi_b = int(np.floor(edges[b])), max(int(np.ceil(edges[b + 1])), int(np.floor(edges[b])) + 1)
        return int(np.clip(self.rng.integers(lo_b, hi_b), 1, hi))

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        t_idx = np.empty(batch_size, np.int64)
        tn_idx = np.empty(batch_size, np.int64)
        g_idx = np.empty(batch_size, np.int64)
        n_eff = np.empty(batch_size, np.float32)
        reached = np.zeros(batch_size, np.float32)
        dist = np.zeros(batch_size, np.float32)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            t = int(self.rng.integers(0, L - 1))
            ne = min(self.n, L - 1 - t)
            t_idx[b], tn_idx[b], n_eff[b] = rows[t], rows[t + ne], ne
            if self.rng.random() < self.p_cross:
                g_idx[b] = int(self.rng.integers(0, self.n_total))  # cross-episode goal
            else:
                _hi = L - 1 - t
                if self.max_delta is not None:
                    _hi = min(_hi, self.max_delta)
                delta = self._offset(_hi)
                g_idx[b] = rows[t + delta]
                if delta <= ne:                      # goal reached within the n-step window
                    reached[b], dist[b] = 1.0, float(delta)
        out = {
            "z_t": self.z[t_idx], "z_tn": self.z[tn_idx], "z_g": self.z[g_idx],
            "n_eff": torch.from_numpy(n_eff), "reached": torch.from_numpy(reached),
            "dist": torch.from_numpy(dist),
        }
        # Row indices, so a learner can look up ground-truth state (qpos) and
        # build a reward from it. The distance learners never needed this: a
        # quasimetric target is pure step counting.
        out["t_idx"] = torch.from_numpy(t_idx)
        out["tn_idx"] = torch.from_numpy(tn_idx)
        out["g_idx"] = torch.from_numpy(g_idx)
        return out


class GeometricFutureSampler(_BaseSampler):
    """Anchor / geometric-future-goal pairs for contrastive value learning.

    ``k ~ Geom(1 - gamma)`` (clipped to the episode end). Negatives are taken
    in-batch by the contrastive loss (every other goal in the batch).
    """

    def __init__(self, cache: LatentCache, gamma: float = 0.99, seed: int = 0):
        super().__init__(cache, seed=seed, min_len=2)
        self.gamma = gamma

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        s_idx = np.empty(batch_size, dtype=np.int64)
        g_idx = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
            rows = self.episodes[e]
            L = len(rows)
            t = int(self.rng.integers(0, L - 1))
            k = int(self.rng.geometric(1.0 - self.gamma))  # >= 1
            g = min(t + k, L - 1)
            s_idx[b], g_idx[b] = rows[t], rows[g]
        return {"z_s": self.z[s_idx], "z_g": self.z[g_idx]}


__all__ = [
    "BalancedHorizonPairSampler",
    "TransitionSampler",
    "NStepGoalSampler",
    "GeometricFutureSampler",
]
