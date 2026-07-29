"""Metric learners for TRM. Each ``fit(cache, cfg, device)`` returns a module
exposing ``cost(z_pred, z_goal)`` (lower == more reachable)."""

from . import contrastive, dwell, regression, td

# ``reward`` is deliberately NOT imported here. It fits a goal-conditioned
# reward R(z, z_g) -> [0, 1] for PWM-style first-order policy extraction, and it
# needs ``..samplers``; importing it eagerly makes initialising this package
# re-enter the parent ``trm`` package, which imports ``learners`` in turn:
#     ImportError: cannot import name 'dwell' from partially initialized module
# That broke every `+metric=` eval (the only path that imports ``trm``) while
# leaving latent-cost evals working, so it failed asymmetrically and looked like
# a solver bug. Import it directly instead:
#     from stable_worldmodel.trm.learners.reward import RewardConfig, fit
__all__ = ["regression", "td", "contrastive", "dwell"]
