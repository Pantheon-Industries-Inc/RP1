"""Temporal objectives shared by LIP training and deployment."""

from typing import Protocol

import torch


class ValueFunction(Protocol):
    def __call__(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor: ...


def trajectory_value(
    value: ValueFunction,
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    start: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Score an imagined trajectory under a supported temporal objective."""
    terminal = value(trajectory[:, -1], goal)
    if mode == "terminal":
        return terminal
    initial = value(start, goal)
    if mode == "tel-exact":
        return terminal - initial
    if mode != "tel-stopprev":
        raise ValueError(f"unsupported temporal objective: {mode}")
    horizon = trajectory.shape[1]
    values = value(
        trajectory.reshape(-1, trajectory.shape[-1]),
        goal.repeat_interleave(horizon, dim=0),
    ).view(trajectory.shape[0], horizon)
    previous = torch.cat([initial[:, None], values[:, :-1]], dim=1).detach()
    return (values - previous).sum(dim=1)


__all__ = ["ValueFunction", "trajectory_value"]
