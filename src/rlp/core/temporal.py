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


def windowed_terminal_value(
    value: ValueFunction,
    trajectory: torch.Tensor,
    goal: torch.Tensor,
    context: int,
) -> torch.Tensor:
    """Terminal value of a ``context``-frame window, as ``MetricCost`` forms it.

    Window critics (the Reacher three-frame quasimetric) take ``context * D``
    inputs. This reproduces the deploy-side convention of
    :meth:`rlp.core.value.cost.MetricCost._metric_inputs` on a differentiable
    imagined trajectory: the last ``context`` frames concatenated (frames before
    the trajectory start clamped to its first frame), the goal tiled to match,
    and the two-frame case expressed as ``[current, current - previous]``.
    """
    if context < 1:
        raise ValueError(f"invalid value context width: {context}")
    if context == 1:
        return value(trajectory[:, -1], goal)
    horizon = trajectory.shape[1]
    indices = (torch.arange(context, device=trajectory.device) + horizon - context).clamp_min(0)
    window = trajectory.index_select(1, indices)
    if context == 2:
        current, previous = window[:, -1], window[:, -2]
        return value(torch.cat([current, current - previous], dim=-1), torch.cat([goal, torch.zeros_like(goal)], -1))
    return value(window.flatten(start_dim=1), goal.repeat(1, context))


__all__ = ["ValueFunction", "trajectory_value", "windowed_terminal_value"]
