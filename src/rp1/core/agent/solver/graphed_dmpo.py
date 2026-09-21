"""CUDA-graph-captured DMPO cost evaluation — an opt-in inference mode.

DMPO's decision is the mirror image of rp1's: rp1 is launch-bound (~80
dependent kernel groups on a batch of one, which is why
:mod:`rp1.core.agent.solver.graphed` buys it ~9x), while DMPO rolls ``N`` sampled
plans through the world model as **one wide forward batch** — 5 dependent
world-model steps at batch ``B*N``, no backward. Graph capture removes launch
overhead, so the expected gain here is small; this module exists to measure
that rather than to assume it.

What is captured: the sampled-plan cost function
``plans (B, N, H, a) -> costs (B, N)`` — the world-model unroll plus the value
head, forward only. The learned update itself is two small MLP calls and is
left eager.

Constraints inherited from CUDA graphs:
  * Static shapes: one capture per ``(batch, num_samples)``. The sample count
    is fixed by the trained checkpoint, so in practice only the env batch
    varies; a changed batch triggers a fresh capture (seconds).
  * Captured tensor addresses are fixed: the decision's context is staged by
    copying into static buffers, and the world model / value weights must not
    be reallocated after capture.

Imported lazily by :class:`rp1.core.agent.solver.DMPOSolver` only when ``graphed``
is enabled; the default eager path is untouched.
"""

from collections.abc import Callable
from typing import Any

import torch

from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, windowed_terminal_value
from rp1.core.world_model.rollout import rollout_traj
from rp1.utils.logging import logger

__all__ = ["GraphedSampledCost"]


class GraphedSampledCost:
    """Per-batch cache of CUDA-graphed DMPO cost evaluations."""

    def __init__(
        self,
        wm: Any,
        value: ValueFunction,
        horizon: int,
        action_dim: int,
        latent_dim: int,
        num_samples: int,
        device: torch.device | str,
        temporal_objective: str = "terminal",
        value_context: int = 1,
        warmup_iters: int = 3,
    ) -> None:
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise ValueError("graphed DMPO inference requires a CUDA device")
        self._wm = wm
        self._value = value
        self._horizon = horizon
        self._action_dim = action_dim
        self._latent_dim = latent_dim
        self._num_samples = num_samples
        self._device = torch.device(device)
        self._objective = temporal_objective
        self._context = value_context
        self._warmup_iters = warmup_iters
        # batch -> (graphed cost fn, zh_s, ah_s, zg_s) staged over B*N rows
        self._captured: dict[int, tuple[Callable[..., Any], torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._bound: int | None = None

    def _capture(self, batch: int) -> None:
        rows = batch * self._num_samples
        logger.info(f"GraphedSampledCost: capturing for batch {batch} x {self._num_samples} samples (one-time)")
        zh_s = torch.zeros(rows, 3, self._latent_dim, device=self._device)
        ah_s = torch.zeros(rows, 2, self._action_dim, device=self._device)
        zg_s = torch.zeros(rows, self._latent_dim, device=self._device)
        wm, value, context, objective = self._wm, self._value, self._context, self._objective

        def cost_fn(flat_plans: torch.Tensor) -> torch.Tensor:
            trajectory = rollout_traj(wm, zh_s, ah_s, flat_plans)
            if context > 1:
                return windowed_terminal_value(value, trajectory, zg_s, context)
            return trajectory_value(value, trajectory, zg_s, zh_s[:, -1], objective)

        sample = torch.zeros(rows, self._horizon, self._action_dim, device=self._device, requires_grad=True)
        graphed = torch.cuda.make_graphed_callables(  # type: ignore[no-untyped-call]  # PyTorch 2.7 stub is untyped.
            cost_fn, (sample,), num_warmup_iters=self._warmup_iters
        )
        self._captured[batch] = (graphed, zh_s, ah_s, zg_s)

    def bind(self, z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor) -> None:
        """Stage one decision's context, repeated across the sample dimension."""
        batch = z_hist.shape[0]
        if batch not in self._captured:
            self._capture(batch)
        _, zh_s, ah_s, zg_s = self._captured[batch]
        repeat = self._num_samples
        zh_s.copy_(z_hist.repeat_interleave(repeat, dim=0))
        ah_s.copy_(a_hist.repeat_interleave(repeat, dim=0))
        zg_s.copy_(z_goal.repeat_interleave(repeat, dim=0))
        self._bound = batch

    def __call__(self, plans: torch.Tensor) -> torch.Tensor:
        """Cost of every sampled plan: ``(B, N, H, a) -> (B, N)``."""
        if self._bound is None or plans.shape[0] != self._bound:
            raise RuntimeError("GraphedSampledCost called without a matching bind()")
        batch, samples = plans.shape[0], plans.shape[1]
        if samples != self._num_samples:
            raise RuntimeError(f"captured for {self._num_samples} samples, got {samples}")
        graphed, *_ = self._captured[self._bound]
        flat = plans.reshape(batch * samples, plans.shape[2], plans.shape[3]).detach().requires_grad_(True)
        costs: torch.Tensor = graphed(flat)
        return costs.detach().view(batch, samples)
