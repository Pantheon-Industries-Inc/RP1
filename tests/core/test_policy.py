from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlp.core.policy import PlanConfig, WorldModelPolicy


class _Space:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _Env:
    num_envs = 1
    action_space = _Space((1, 1))
    single_action_space = _Space((1,))


class _Solver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.remaining: list[list[int]] = []

    def configure(self, **kwargs: Any) -> None:
        self.configuration = kwargs

    def set_align_remaining(self, remaining: list[int]) -> None:
        self.remaining.append(remaining)

    def __call__(self, info: dict[str, Any], *, init_action: torch.Tensor | None) -> dict[str, torch.Tensor]:
        self.calls.append({"info": info, "init_action": init_action})
        offset = 10 * len(self.calls)
        actions = torch.tensor([[[offset + 1.0, offset + 2.0], [offset + 3.0, offset + 4.0], [0.0, 0.0]]])
        return {"actions": actions}


def _info(frame: float) -> dict[str, np.ndarray]:
    return {
        "pixels": np.full((1, 1, 2), frame, dtype=np.float32),
        "proprio": np.full((1, 1, 1), frame, dtype=np.float32),
        "terminated": np.array([False]),
    }


def test_policy_keeps_real_history_and_updates_deadline_on_replan() -> None:
    solver = _Solver()
    config = PlanConfig(horizon=3, receding_horizon=1, history_len=3, action_block=2, warm_start=True, deadline=5)
    policy = WorldModelPolicy(solver=solver, config=config)
    policy.set_env(_Env())

    first = policy.get_action(_info(1.0))
    buffered = policy.get_action(_info(2.0))
    replanned = policy.get_action(_info(3.0))

    assert first.item() == 11.0
    assert buffered.item() == 12.0
    assert replanned.item() == 21.0
    assert solver.remaining == [[3], [2]]
    assert len(solver.calls) == 2

    first_history = solver.calls[0]["info"]["pixels_hist"]
    second_history = solver.calls[1]["info"]["pixels_hist"]
    assert torch.equal(first_history[0, :, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(second_history[0, :, 0], torch.tensor([1.0, 1.0, 3.0]))
    assert torch.equal(solver.calls[0]["info"]["_align_remaining"], torch.tensor([3]))
    assert torch.equal(solver.calls[1]["info"]["_align_remaining"], torch.tensor([2]))
    assert solver.calls[1]["init_action"] is not None


def test_unwrap_encoder_peels_cost_wrappers() -> None:
    import torch
    from torch import nn

    from rlp.core.solver.lip import unwrap_encoder
    from rlp.core.value.stable_worldmodel import LatentGoalCost

    class WM(nn.Module):
        def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return info

    wm = WM()
    assert unwrap_encoder(wm) is wm
    assert unwrap_encoder(LatentGoalCost(wm)) is wm

    class OuterCost(nn.Module):  # MetricCost-shaped: inner stack at .base
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

    assert unwrap_encoder(OuterCost(LatentGoalCost(wm))) is wm
