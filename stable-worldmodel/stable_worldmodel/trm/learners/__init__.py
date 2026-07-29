"""Metric learners for TRM. Each ``fit(cache, cfg, device)`` returns a module
exposing ``cost(z_pred, z_goal)`` (lower == more reachable)."""

from . import contrastive, dwell, regression, reward, td

# ``reward`` is the odd one out: it does not expose ``cost``. It fits a
# goal-conditioned reward R(z, z_g) -> [0, 1] for PWM-style first-order policy
# extraction, where the actor objective needs a per-step reward in imagination.
__all__ = ["regression", "td", "contrastive", "dwell", "reward"]
