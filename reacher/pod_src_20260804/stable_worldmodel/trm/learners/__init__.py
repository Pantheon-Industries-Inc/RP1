"""Metric learners for TRM. Each ``fit(cache, cfg, device)`` returns a module
exposing ``cost(z_pred, z_goal)`` (lower == more reachable)."""

from . import contrastive, regression, td

# Only learners on the *evaluation* path belong in this eager list. Everything
# else is imported lazily by ``trm.io.load_metric`` at the point of use, because
# any name listed here is a hard dependency of every ``+metric=`` eval:
#
# * ``dwell`` (discounted-dwell value, a discarded reacher experiment) was listed
#   here while ``dwell.py`` was untracked, so ``git archive`` -- which ships only
#   tracked files -- sent pods an ``__init__`` referencing a module it had not
#   sent. Python reports the missing submodule as
#       ImportError: cannot import name 'dwell' from partially initialized
#       module 'stable_worldmodel.trm.learners' (circular import)
#   which reads like a cycle, and it killed every TD eval while latent-cost evals
#   kept working (``eval_wm.py`` imports ``trm`` only in the metric branch), so
#   the failure looked like a solver bug.
# * ``reward`` (goal-conditioned R(z, z_g) for PWM-style policy extraction) needs
#   ``..samplers``, so importing it here re-enters the parent ``trm`` package --
#   a genuine cycle, same symptom.
#
# Import either one directly instead:
#     from stable_worldmodel.trm.learners.dwell import DwellConfig
__all__ = ["regression", "td", "contrastive"]
