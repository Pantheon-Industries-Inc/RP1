"""Weight averaging over deployable planner snapshots ("model soup").

`planner.ckpt_every` leaves a trail of deployable snapshots behind a run, and
the selection pass keeps exactly one of them. Averaging the top few instead is
the standard variance reducer on that axis: it ships ONE actor that still runs
a single K-iteration pass, so it changes nothing about the deploy budget.

Only the actor tensors under ``"sd"`` are averaged. Every other field in the
payload describes the architecture and must agree across the inputs, otherwise
the average would not be loadable -- a mismatch is an error, never a silent
pick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

__all__ = ["average_planner_payloads", "average_planner_checkpoints"]

# Fields that may differ between snapshots of one run without making the
# average meaningless (they are provenance, not architecture).
_IGNORED = frozenset({"step", "val", "source"})


def _architecture(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "sd" and k not in _IGNORED}


def average_planner_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Average the actor tensors of deployable planner payloads.

    Raises ``ValueError`` if the payloads disagree on architecture, on the set
    of parameter names, or on any tensor's shape or dtype.
    """
    if not payloads:
        raise ValueError("no planner payloads to average")
    head, *rest = payloads
    reference = _architecture(head)
    for index, other in enumerate(rest, start=1):
        if _architecture(other) != reference:
            raise ValueError(f"payload {index} disagrees on architecture with payload 0")
    states = [p["sd"] for p in payloads]
    names = set(states[0])
    for index, state in enumerate(states[1:], start=1):
        if set(state) != names:
            raise ValueError(f"payload {index} has a different parameter set than payload 0")
    if len(payloads) == 1:
        return {**reference, "sd": {k: v.detach().clone() for k, v in states[0].items()}}
    averaged: dict[str, torch.Tensor] = {}
    for name in states[0]:
        tensors = [state[name] for state in states]
        first = tensors[0]
        for index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != first.shape or tensor.dtype != first.dtype:
                raise ValueError(f"'{name}' differs in shape/dtype between payload 0 and {index}")
        if first.is_floating_point():
            stacked = torch.stack([t.detach().to(torch.float64) for t in tensors])
            averaged[name] = stacked.mean(dim=0).to(first.dtype)
        else:
            # Integer buffers (counters, masks) have no meaningful mean; they
            # must already agree, or the snapshots are not interchangeable.
            for index, tensor in enumerate(tensors[1:], start=1):
                if not torch.equal(tensor, first):
                    raise ValueError(f"non-float tensor '{name}' differs between payload 0 and {index}")
            averaged[name] = first.detach().clone()
    return {**reference, "sd": averaged}


def average_planner_checkpoints(paths: list[Path], output: Path) -> dict[str, Any]:
    """Load, average and write deployable planner checkpoints."""
    payloads = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    merged = average_planner_payloads(payloads)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)
    return merged
