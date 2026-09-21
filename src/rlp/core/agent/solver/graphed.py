"""CUDA-graph-captured LIP refinement — an opt-in inference mode.

The LIP decision is launch-bound, not compute-bound: ~25k sequential kernel
launches at 0.004% of GPU peak. This module records the per-iteration score
computation (world-model unroll -> trajectory value, forward AND backward)
into CUDA graphs once, then replays them per refinement iteration. Measured on
an architecture-matched benchmark (job 4840, H200): 272.6 -> 29.3 ms/decision
at batch 1, final-plan deviation vs the eager production path 5.96e-8 (one
float32 ulp; identical to the trajectory-reuse deviation — capture itself adds
zero), fp32 throughout.

The gradient unroll also produces the actor's value input, so the duplicate
scoring unroll the audit flagged as dead work is never executed.

Constraints inherited from CUDA graphs:
  * Static shapes: one capture per (batch, horizon, action_dim). New batch
    sizes trigger a fresh capture (seconds); shrinking eval batches must be
    padded by the caller (audit fix #12) or left on the eager path.
  * The captured tensors' addresses are fixed: inputs are staged by copying
    into static buffers; the world model and value weights may be updated
    in place but must not be reallocated after capture.

Two captures live here, one per planner family:
  * :class:`GraphedRefinement` — LIP/RLP: score AND backward for one plan per
    problem, replayed per refinement iteration.
  * :class:`GraphedCost` — the learned-optimizer solvers (L2O-MPC, DMPO):
    forward-only cost of ``N`` sampled plans per problem, replayed per inner
    iteration. No backward exists on that path, so this is a plain
    ``CUDAGraph`` capture rather than ``make_graphed_callables``.

This file is intentionally self-contained and imported lazily by the solvers
only when ``graphed`` is enabled — the default eager path is untouched.
"""

from collections.abc import Callable
from typing import Any

import torch

from rlp.core.agent.value.temporal import ValueFunction, trajectory_value
from rlp.core.world_model.rollout import rollout_traj
from rlp.utils.logging import logger

__all__ = ["GraphedCost", "GraphedRefinement"]


class GraphedRefinement:
    """Per-batch-size cache of CUDA-graphed LIP score functions.

    One instance lives on the solver. ``bind()`` stages a decision's context
    (latent history, action history, goal) into the static buffers, capturing
    graphs on first use of a batch size; ``step()`` then serves one refinement
    iteration: score, detached trajectory features, and the exact value
    gradient — one forward-graph replay plus one backward-graph replay.
    """

    def __init__(
        self,
        wm: Any,
        value: ValueFunction,
        temporal_objective: str,
        horizon: int,
        action_dim: int,
        latent_dim: int,
        device: torch.device | str,
        warmup_iters: int,
    ) -> None:
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise ValueError("graphed LIP inference requires a CUDA device")
        self._wm = wm
        self._value = value
        self._objective = temporal_objective
        self._horizon = horizon
        self._action_dim = action_dim
        self._latent_dim = latent_dim
        self._device = torch.device(device)
        self._warmup_iters = warmup_iters
        # batch -> (graphed score fn, zh_s, ah_s, zg_s static buffers)
        self._captured: dict[int, tuple[Callable[..., Any], torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._bound: int | None = None

    def _capture(self, batch: int) -> None:
        logger.info(f"GraphedRefinement: capturing CUDA graphs for batch {batch} (one-time, takes seconds)")
        zh_s = torch.zeros(batch, 3, self._latent_dim, device=self._device)
        ah_s = torch.zeros(batch, 2, self._action_dim, device=self._device)
        zg_s = torch.zeros(batch, self._latent_dim, device=self._device)
        wm, value, objective = self._wm, self._value, self._objective

        def score_fn(A_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            traj = rollout_traj(wm, zh_s, ah_s, A_in)
            score = trajectory_value(value, traj, zg_s, zh_s[:, -1], objective)
            # second output is detached: actor features only, never on the tape
            return score, traj.detach()

        sample = torch.zeros(batch, self._horizon, self._action_dim, device=self._device, requires_grad=True)
        graphed = torch.cuda.make_graphed_callables(  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
            score_fn, (sample,), num_warmup_iters=self._warmup_iters
        )
        self._captured[batch] = (graphed, zh_s, ah_s, zg_s)

    def bind(self, z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor) -> None:
        """Stage one decision's context; captures graphs for a new batch size."""
        batch = z_hist.shape[0]
        if batch not in self._captured:
            self._capture(batch)
        _, zh_s, ah_s, zg_s = self._captured[batch]
        zh_s.copy_(z_hist)
        ah_s.copy_(a_hist)
        zg_s.copy_(z_goal)
        self._bound = batch

    def step(self, plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One refinement iteration: (score, detached trajectory, dscore/dplan)."""
        if self._bound is None or plan.shape[0] != self._bound:
            raise RuntimeError("GraphedRefinement.step called without a matching bind()")
        graphed, *_ = self._captured[self._bound]
        plan_in = plan.detach().requires_grad_(True)
        score, traj_f = graphed(plan_in)
        (grad_plan,) = torch.autograd.grad(score.sum(), plan_in)
        return score.detach(), traj_f, grad_plan

    def score(self, plan: torch.Tensor) -> torch.Tensor:
        """Score a plan (selection unroll) through the same captured graph."""
        if self._bound is None or plan.shape[0] != self._bound:
            raise RuntimeError("GraphedRefinement.score called without a matching bind()")
        graphed, *_ = self._captured[self._bound]
        score, _ = graphed(plan.detach().requires_grad_(True))
        detached: torch.Tensor = score.detach()
        return detached


class GraphedCost:
    """Per-row-count cache of CUDA-graphed sampled-plan cost evaluations.

    The learned-optimizer solvers spend their whole decision in one shape:
    ``rows = batch * num_samples`` plans, each unrolled ``horizon`` blocks
    through the frozen world model and scored by the frozen critic, with no
    gradient anywhere. That is a fixed-shape, side-effect-free forward pass —
    the ideal CUDA-graph candidate, and it is where a launch-bound solver
    spends its wall-clock.

    ``costs()`` stages the decision's context and plans into static buffers and
    replays the graph, capturing on first use of a row count (seconds, once).
    The returned tensor is cloned: the graph's output buffer is overwritten by
    the next replay.
    """

    def __init__(
        self,
        score: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        horizon: int,
        action_dim: int,
        latent_dim: int,
        device: torch.device | str,
        warmup_iters: int = 3,
    ) -> None:
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise ValueError("graphed cost evaluation requires a CUDA device")
        self._score = score
        self._horizon = int(horizon)
        self._action_dim = int(action_dim)
        self._latent_dim = int(latent_dim)
        self._device = torch.device(device)
        self._warmup_iters = int(warmup_iters)
        # rows -> (graph, z_hist, a_hist, z_goal, plans, costs) static tensors
        self._captured: dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _capture(self, rows: int) -> None:
        logger.info(f"GraphedCost: capturing a CUDA graph for {rows} rows (one-time, takes seconds)")
        z_hist = torch.zeros(rows, 3, self._latent_dim, device=self._device)
        a_hist = torch.zeros(rows, 2, self._action_dim, device=self._device)
        z_goal = torch.zeros(rows, self._latent_dim, device=self._device)
        plans = torch.zeros(rows, self._horizon, self._action_dim, device=self._device)

        # Warm up on a side stream first: capture records whatever allocations
        # and cuBLAS handles already exist, so the first calls must not be
        # inside the capture.
        stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self._warmup_iters):
                self._score(z_hist, a_hist, z_goal, plans)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
        with torch.cuda.graph(graph):
            costs = self._score(z_hist, a_hist, z_goal, plans)
        self._captured[rows] = (graph, z_hist, a_hist, z_goal, plans, costs)

    def costs(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
        plans: torch.Tensor,
    ) -> torch.Tensor:
        """Cost of every row of ``plans``; inputs already expanded to rows."""
        rows = plans.shape[0]
        if rows not in self._captured:
            self._capture(rows)
        graph, z_hist_s, a_hist_s, z_goal_s, plans_s, costs = self._captured[rows]
        z_hist_s.copy_(z_hist)
        a_hist_s.copy_(a_hist)
        z_goal_s.copy_(z_goal)
        plans_s.copy_(plans)
        graph.replay()  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
        return costs.clone()
