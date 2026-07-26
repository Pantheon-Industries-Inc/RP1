"""Metric learners for TRM. Each ``fit(cache, cfg, device)`` returns a module
exposing ``cost(z_pred, z_goal)`` (lower == more reachable)."""

from . import contrastive, qrl, regression, td

__all__ = ["regression", "td", "contrastive", "qrl"]
