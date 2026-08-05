"""MetricCost: drop-in terminal cost wrapping a frozen world model + a metric.

This implements the **Costable** protocol so it plugs directly into any solver
(``CEMSolver`` etc.) in place of the world model. It reuses the frozen WM's
``encode`` / ``rollout`` to obtain the predicted terminal latent and the goal
latent (exactly what ``LeWM.get_cost`` already computes), then applies a learned
TRM metric as the terminal cost.

Modes (paper terminology):

* ``latent``      -- passthrough baseline ``c_lat = ||z_hat_T - z_g||^2`` (the
                     mismatched Euclidean cost the paper repairs).
* ``replacement`` -- ``m_phi(z_hat_T, z_g)`` (TRM replaces the cost).
* ``hybrid``      -- ``std(c_lat) + lambda * std(m_phi)`` with per-row (per-env,
                     over candidates) standardisation to prevent scale dominance.
* ``shuffled``    -- like ``replacement`` but with a head trained on shuffled
                     temporal labels (negative control).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, cast

import torch
from torch import nn

from .protocols import TensorInfo, ValueMetric
from .stable_worldmodel import as_planning_cost


class MetricCost(nn.Module):
    """Wrap a frozen base WM and a metric module behind the cost interface.

    Args:
        base_wm: a frozen world model exposing ``get_cost`` that, as a side
            effect, populates ``info_dict['predicted_emb']`` (rollout) and
            ``info_dict['goal_emb']`` (encoded goal) -- true for ``LeWM`` and the
            lightweight state-WM in this repo.
        metric: a module with ``cost(z_pred, z_goal) -> Tensor`` (regression head,
            TD value, or contrastive critic). ``None`` only for ``latent`` mode.
        mode: one of ``latent | replacement | hybrid | shuffled``.
        lam: hybrid weighting ``lambda`` on the standardised metric term.
    """

    def __init__(
        self,
        base_wm: nn.Module,
        metric: nn.Module | None,
        mode: str = "replacement",
        lam: float = 1.0,
        metrics: Sequence[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        assert mode in {"latent", "replacement", "hybrid", "shuffled"}, mode
        if mode != "latent":
            assert metric is not None, f"mode={mode} requires a metric module"
        self.base = as_planning_cost(base_wm)
        self.metric = metric
        self.metrics = nn.ModuleList(metrics or ([] if metric is None else [metric]))
        self.mode = mode
        self.lam = lam

    def parameters(self, *args: Any, **kwargs: Any) -> Iterator[nn.Parameter]:
        return self.base.parameters(*args, **kwargs)

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        """Per-row (over candidate dim) standardisation."""
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def _metric_inputs(self, info_dict: TensorInfo) -> tuple[torch.Tensor, torch.Tensor]:
        """Build terminal/history inputs matching the metric checkpoint width."""
        if self.metric is None:
            raise RuntimeError("metric input requested in latent-only mode")
        predicted = info_dict["predicted_emb"]
        goal_raw = info_dict["goal_emb"]
        base_dim = predicted.shape[-1]
        # Generic one-frame metrics predate explicit checkpoint metadata.
        # RLP's persisted metrics expose ``latent_dim`` so wider history
        # inputs can be reconstructed without inspecting implementation layers.
        metric_dim = int(getattr(self.metric, "latent_dim", base_dim))
        if metric_dim % base_dim:
            raise ValueError(f"metric latent_dim={metric_dim} is not a multiple of world-model latent dim={base_dim}")
        context = metric_dim // base_dim
        if context < 1:
            raise ValueError(f"invalid metric context width: {context}")

        pred = predicted[..., -1, :]
        goal = goal_raw[..., -1, :]
        if context == 2:
            if predicted.shape[-2] < 2:
                raise ValueError("two-frame delta metric requires at least two predicted frames")
            previous = predicted[..., -2, :]
            pred = torch.cat([pred, pred - previous], dim=-1)
            goal = torch.cat([goal, torch.zeros_like(goal)], dim=-1)
        elif context > 2:
            if predicted.shape[-2] < context:
                raise ValueError(
                    f"{context}-frame metric requires {context} predicted frames; got {predicted.shape[-2]}"
                )
            pred = predicted[..., -context:, :].flatten(start_dim=-2)
            goal = torch.cat([goal] * context, dim=-1)

        if goal.ndim < pred.ndim:
            goal = goal.unsqueeze(1)
        return pred, goal.expand_as(pred)

    def _metric_terminal_cost(self, info_dict: TensorInfo) -> torch.Tensor:
        """Apply the learned metric to the cached terminal/goal latents.

        Expects ``predicted_emb`` (B, S, T, D) and ``goal_emb`` in either
        ``(B, T, D)`` or Stable-WM's ``(B, 1, T, D)`` form.
        """
        pred, goal = self._metric_inputs(info_dict)
        costs = [cast(ValueMetric, metric).cost(pred.float(), goal.float()) for metric in self.metrics]
        return torch.stack(costs).amax(dim=0)

    @torch.inference_mode()
    def get_cost(self, info_dict: TensorInfo, action_candidates: torch.Tensor) -> torch.Tensor:
        # base.get_cost computes c_lat AND populates predicted_emb / goal_emb.
        c_lat = self.base.get_cost(info_dict, action_candidates)
        if self.mode == "latent":
            return c_lat
        m = self._metric_terminal_cost(info_dict)
        if self.mode in ("replacement", "shuffled"):
            return m
        # hybrid
        return self._standardize(c_lat) + self.lam * self._standardize(m)

    # criterion mirrors the Costable protocol (used by some solvers/diagnostics)
    def criterion(self, info_dict: TensorInfo) -> torch.Tensor:
        if self.mode == "latent":
            return self.base.criterion(info_dict)
        m = self._metric_terminal_cost(info_dict)
        if self.mode in ("replacement", "shuffled"):
            return m
        return self._standardize(self.base.criterion(info_dict)) + self.lam * self._standardize(m)


__all__ = ["MetricCost"]
