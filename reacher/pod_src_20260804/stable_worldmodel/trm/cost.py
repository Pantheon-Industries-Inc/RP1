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

import torch
from torch import nn


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

    def __init__(self, base_wm: nn.Module, metric: nn.Module | None, mode: str = "replacement", lam: float = 1.0):
        super().__init__()
        assert mode in {"latent", "replacement", "hybrid", "shuffled"}, mode
        if mode != "latent":
            assert metric is not None, f"mode={mode} requires a metric module"
        self.base = base_wm
        self.metric = metric
        self.mode = mode
        self.lam = lam

    def parameters(self, *a, **k):  # so solvers can infer device/dtype from base
        return self.base.parameters(*a, **k)

    @staticmethod
    def _standardize(x: torch.Tensor) -> torch.Tensor:
        """Per-row (over candidate dim) standardisation."""
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def _metric_terminal_cost(self, info_dict: dict) -> torch.Tensor:
        """Apply the learned metric to the cached terminal/goal latents.

        Expects ``predicted_emb`` (B, S, T, D) and ``goal_emb`` (B, T, D) to have
        been populated by a prior ``base.get_cost`` call.
        """
        pred = info_dict["predicted_emb"][..., -1, :]          # (B, S, D)
        goal = info_dict["goal_emb"][..., -1, :]               # (B, D)
        goal = goal.unsqueeze(1).expand_as(pred)               # (B, S, D)
        return self.metric.cost(pred, goal)                    # (B, S)

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
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
    def criterion(self, info_dict: dict) -> torch.Tensor:
        if self.mode == "latent":
            return self.base.criterion(info_dict)
        m = self._metric_terminal_cost(info_dict)
        if self.mode in ("replacement", "shuffled"):
            return m
        return self._standardize(self.base.criterion(info_dict)) + self.lam * self._standardize(m)


__all__ = ["MetricCost"]
