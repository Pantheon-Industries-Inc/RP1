from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from gymnasium.spaces import Box

from rp1.core.agent.solver import RP1Solver
from rp1.training.harness.checkpointing import load_planner, load_wm


def _solver(world_model: Path, planner: Path, **overrides: Any) -> RP1Solver:
    settings: dict[str, Any] = {
        "rp1_select": "last",
        "init_mode": "zero",
        "init_samples": 64,
        "init_scale": 1.5,
        "cem_init_steps": 30,
        "iters_override": None,
        "graphed": False,
        "graph_warmup_iters": 5,
    }
    solver = RP1Solver(
        model=load_wm(str(world_model)),
        checkpoint=load_planner(str(planner), None),
        batch_size=2,
        num_samples=8,
        var_scale=1.0,
        n_steps=0,
        topk=2,
        lam=1.0,
        device="cpu",
        seed=0,
        **{**settings, **overrides},
    )
    solver.configure(
        action_space=Box(-1, 1, shape=(2, 5)),
        n_envs=2,
        config=SimpleNamespace(horizon=5, action_block=5, receding_horizon=5),
    )
    return solver


def _observation() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "pixels": torch.randn(2, 3, 3, 224, 224, generator=generator),
        "goal": torch.randn(2, 1, 3, 224, 224, generator=generator),
    }


def test_plans_are_deterministic_and_inside_the_action_limit(cube_world_model: Path, cube_planner: Path) -> None:
    solver = _solver(cube_world_model, cube_planner)
    first = solver.solve(_observation())["actions"]
    second = solver.solve(_observation())["actions"]
    assert first.shape == (2, 5, 25)
    assert torch.equal(first, second)
    assert first.abs().max() <= solver.actor.action_limit


def test_zero_iterations_emit_the_initial_plan(cube_world_model: Path, cube_planner: Path) -> None:
    solver = _solver(cube_world_model, cube_planner, iters_override=0)
    assert torch.count_nonzero(solver.solve(_observation())["actions"]) == 0
