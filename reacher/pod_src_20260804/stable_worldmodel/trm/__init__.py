"""Trajectory Reachability Metrics (TRM).

Post-hoc repair of latent world-model terminal costs for MPC, following
"Beyond Euclidean Proximity: Repairing Latent World Models with Horizon-Matched
Trajectory Reachability Metrics" (arXiv 2605.22164), extended with three ways to
learn the metric (regression / offline TD / contrastive) and deployment beyond
TwoRoom.

Public surface:
    * ``PairwiseMetricHead``    -- the pairwise scalar head.
    * ``MetricCost``            -- Costable wrapper (latent/replacement/hybrid/shuffled).
    * ``LatentCache`` / ``encode_dataset`` -- frozen-latent caching.
    * ``learners``              -- regression / td / contrastive ``fit`` functions.
    * samplers, oracle, diagnostics.
"""

from . import diagnostics, learners, oracle, samplers
from .cost import MetricCost
from .head import PairwiseMetricHead, pair_features
from .io import build_metric, load_metric, save_metric
from .latent_cache import LatentCache, encode_dataset

__all__ = [
    "PairwiseMetricHead",
    "pair_features",
    "MetricCost",
    "LatentCache",
    "encode_dataset",
    "save_metric",
    "load_metric",
    "build_metric",
    "learners",
    "samplers",
    "oracle",
    "diagnostics",
]
