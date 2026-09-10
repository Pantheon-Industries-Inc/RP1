"""The agent's value function -- Trajectory Reachability Metrics (TRM).

Post-hoc repair of latent world-model terminal costs for MPC, following
"Beyond Euclidean Proximity: Repairing Latent World Models with Horizon-Matched
Trajectory Reachability Metrics" (arXiv 2605.22164), extended with three ways to
learn the metric (regression / offline TD / contrastive) and deployment beyond
TwoRoom.

This is what :class:`~rlp.core.solver.LIPSolver` scores rollout terminals with;
the frozen-latent cache it trains on lives in :mod:`rlp.data`.

Public surface:
    * ``PairwiseMetricHead``    -- the pairwise scalar head.
    * ``MetricCost``            -- Costable wrapper (latent/replacement/hybrid/shuffled).
    * ``learners``              -- regression / td / contrastive ``fit`` functions.
    * samplers, oracle, diagnostics.
"""

from . import diagnostics, learners, oracle, samplers
from .contact import ContactPenalty, StateProbe
from .cost import MetricCost
from .head import IQEHead, L2WindowCost, PairwiseMetricHead, QuasimetricHead, pair_features
from .io import build_metric, load_metric, save_metric
from .stable_worldmodel import LatentGoalCost, as_planning_cost

__all__ = [
    "ContactPenalty",
    "StateProbe",
    "PairwiseMetricHead",
    "QuasimetricHead",
    "IQEHead",
    "L2WindowCost",
    "pair_features",
    "MetricCost",
    "LatentGoalCost",
    "as_planning_cost",
    "save_metric",
    "load_metric",
    "build_metric",
    "learners",
    "samplers",
    "oracle",
    "diagnostics",
]
