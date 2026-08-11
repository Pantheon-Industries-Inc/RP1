"""Same-Candidate Selection Audit (SCSA).

For a fixed set of candidate plans at a planning state, SCSA compares the
*metric-under-test* ranking against an *oracle* terminal-quality ranking on the
**same candidates** (both are costs: lower == better). This isolates the
terminal-cost interface from the sampler/optimiser. Metrics (per the paper):

* ``spearman``  -- rank correlation of candidate costs vs oracle costs
  (higher == the metric orders candidates like the oracle).
* ``best_rank_pct`` -- percentile rank, under the oracle, of the candidate the
  metric selects (0 == metric picked the oracle-best; lower is better).
* ``regret`` -- oracle cost of the metric-selected candidate minus the oracle
  best (0 == optimal selection).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torchmetrics.functional import spearman_corrcoef


def _to_np(x: ArrayLike | torch.Tensor) -> NDArray[Any]:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    prediction = torch.as_tensor(a, dtype=torch.float64)
    target = torch.as_tensor(b, dtype=torch.float64)
    if torch.unique(prediction).numel() < 2 or torch.unique(target).numel() < 2:
        return float("nan")
    return float(spearman_corrcoef(prediction, target))


def scsa_pointwise(metric_costs: ArrayLike | torch.Tensor, oracle_costs: ArrayLike | torch.Tensor) -> dict[str, float]:
    """SCSA stats for one planning state (1-D arrays of equal length)."""
    m = _to_np(metric_costs).reshape(-1)
    o = _to_np(oracle_costs).reshape(-1)
    sel = int(np.argmin(m))
    best_oracle = float(o.min())
    sel_oracle = float(o[sel])
    best_rank_pct = float((o < sel_oracle).mean())  # fraction strictly better
    return {
        "spearman": _spearman(m, o),
        "best_rank_pct": best_rank_pct,
        "regret": sel_oracle - best_oracle,
        "selected_oracle_cost": sel_oracle,
        "oracle_best_cost": best_oracle,
    }


def scsa_aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    """Average SCSA stats over planning states (ignoring NaNs)."""
    if not records:
        return {}
    keys = records[0].keys()
    return {k: float(np.nanmean([r[k] for r in records])) for k in keys}


__all__ = ["scsa_pointwise", "scsa_aggregate"]
