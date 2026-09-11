"""L2OSolver — plan with a trained L2O-MPC learned-optimizer checkpoint.

*Learning to Optimize in Model Predictive Control* (Sacks & Boots, ICRA 2022)
keeps MPC's sample-rollout-reduce loop and learns the whole reduction;
:class:`L2ONet` holds the learned update and this solver supplies the rollouts
and the cost.

Per decision the solver spends ``num_samples * iters`` forward world-model
unrolls and no backward pass — the shipped configuration (64 x 4 = 256)
matches DMPO's deployed budget exactly, so the two learned-optimizer rows
differ only in method, not compute. The checkpoint's sample count is
authoritative: the network consumes the ``N`` costs positionally, so ``N``
and the horizon cannot be changed after training. The iteration count can be
(``iters``), as with DMPO.

Costs come from the checkpoint's own goal-conditioned value through the frozen
world model — the same critic every other planner row is scored against, so an
L2O-vs-DMPO-vs-RLP table isolates the *planner*.

Warm start: L2O-MPC has no learned shift model; the previous decision's
unexecuted tail is shift-forwarded the standard DMD-MPC way. Under this
repository's open-loop protocol (``receding_horizon == horizon``) nothing
survives the shift and the warm start is inert by construction.

Trainer: :mod:`rlp.train.l2o`.
"""

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rlp.logging import logger

from ..planner.l2o import L2ONet
from ..rollout import rollout_traj
from ..temporal import trajectory_value, windowed_terminal_value
from ..world_model.protocols import LatentWorldModel
from .lip import EncoderWorldModel, ValueFunction, unwrap_encoder


class L2OCheckpoint(TypedDict):
    kind: str
    z_dim: int
    horizon: int
    a_dim: int
    iters: int
    num_samples: int
    value: str
    sd: dict[str, torch.Tensor]
    amax: NotRequired[float]
    init_std: NotRequired[float]
    hidden: NotRequired[int]
    learn_std: NotRequired[bool]
    gate_bias: NotRequired[float]
    halton: NotRequired[bool]
    seed_val: NotRequired[int]
    temporal_objective: NotRequired[str]
    value_context: NotRequired[int]


__all__ = ["L2OSolver"]


class L2OSolver(CEMSolver):
    """Sample-and-reduce planning with L2O-MPC's learned update rule."""

    def __init__(
        self,
        *args: Any,
        actor_path: str = "",
        value_path: str | None = None,
        iters: int | None = None,
        cost_chunk: int = 0,
        graphed: bool | str = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from rlp.core.value import load_metric

        self.cost_chunk = int(cost_chunk)
        if graphed not in (True, False, "verify"):
            raise ValueError(f"graphed must be true, false, or 'verify' (got {graphed!r})")
        self.graphed = graphed
        self._graph: Any = None

        raw = torch.load(actor_path, map_location=self.device, weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError("L2O checkpoint must contain a mapping")
        ck = cast(L2OCheckpoint, raw)
        if ck.get("kind") != "l2o":
            raise ValueError(f"L2OSolver: unsupported checkpoint kind {ck.get('kind')!r}")
        self.net = L2ONet(
            horizon=ck["horizon"],
            a_dim=ck["a_dim"],
            num_samples=ck["num_samples"],
            hidden=ck.get("hidden", 1024),
            amax=ck.get("amax", 2.5),
            init_std=ck.get("init_std", 1.0),
            learn_std=ck.get("learn_std", False),
            gate_bias=ck.get("gate_bias", 0.0),
            halton=ck.get("halton", True),
            seed_val=ck.get("seed_val", 0),
        ).to(self.device)
        self.net.load_state_dict(ck["sd"])
        self.net.eval()
        self.net.requires_grad_(False)
        self._actor_horizon = int(ck["horizon"])
        self.iters = int(ck["iters"] if iters is None else iters)
        requested_samples: int = self.num_samples
        if requested_samples != self.net.num_samples:
            logger.info(
                f"L2O sample count is fixed by the checkpoint: using {self.net.num_samples} "
                f"(config asked for {requested_samples})"
            )
        self.temporal_objective = ck.get("temporal_objective", "terminal")

        value_reference = value_path or ck["value"]
        if value_path is None and not Path(value_reference).exists():
            recorded = Path(value_reference)
            actor_directory = Path(actor_path).resolve().parent
            for candidate in (actor_directory / recorded.name, actor_directory / recorded.stem):
                if candidate.exists():
                    logger.info(f"L2O value fallback: {value_reference} missing, using sibling {candidate}")
                    value_reference = str(candidate)
                    break
        value_module = load_metric(value_reference, device=self.device)
        value_module.eval()
        self.value = cast(ValueFunction, value_module)
        latent_dim = int(getattr(value_module, "latent_dim", ck["z_dim"]))
        self.value_context = int(ck.get("value_context", max(latent_dim // int(ck["z_dim"]), 1)))
        if self.value_context > 1 and self.temporal_objective != "terminal":
            raise ValueError("window critics support the terminal objective only")

    @property
    def horizon(self) -> int:
        return int(self._actor_horizon)

    def _base(self) -> torch.nn.Module:
        return unwrap_encoder(self.model)

    # ------------------------------------------------------------- encoding
    def _encode(self, info_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent history ``(B, 3, D)`` and goal latent ``(B, D)``."""
        wm = cast(EncoderWorldModel, self._base())
        with torch.no_grad():
            px = info_dict["pixels"].to(self.device, dtype=self.dtype)
            enc_in = {"pixels": px}
            if getattr(wm, "wants_proprio", False):
                pro = info_dict.get("proprio")
                if pro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'proprio'")
                pro = torch.as_tensor(np.asarray(pro), dtype=torch.float32, device=self.device)
                enc_in["proprio"] = pro.reshape(px.shape[0], px.shape[1], -1)
            z_hist = wm.encode(enc_in)["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            genc_in = {"pixels": gx}
            if getattr(wm, "wants_proprio", False):
                gpro = info_dict.get("goal_state")
                if gpro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'goal_state'")
                gpro = torch.as_tensor(np.asarray(gpro), dtype=torch.float32, device=self.device)
                gpro = gpro.reshape(gx.shape[0], -1)[:, -2:]
                genc_in["proprio"] = gpro.unsqueeze(1).expand(-1, gx.shape[1], -1)
            z_goal = wm.encode(genc_in)["emb"][:, -1].float()
        return z_hist, z_goal

    def _cost_fn(
        self,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        wm = cast(LatentWorldModel, self._base())
        chunk = self.cost_chunk

        def score_rows(zh: torch.Tensor, ah: torch.Tensor, zg: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
            """Cost of one flattened batch of plans — the whole inner-loop hot path."""
            traj = rollout_traj(wm, zh, ah, flat)
            if self.value_context > 1:
                return windowed_terminal_value(self.value, traj, zg, self.value_context)
            return trajectory_value(self.value, traj, zg, zh[:, -1], self.temporal_objective)

        if self.graphed and self._graph is None:
            from .graphed import GraphedCost

            self._graph = GraphedCost(
                score_rows,
                horizon=self.horizon,
                action_dim=self.action_dim,
                latent_dim=z_hist.shape[-1],
                device=self.device,
            )

        def cost(plans: torch.Tensor) -> torch.Tensor:
            batch, samples = plans.shape[0], plans.shape[1]
            flat = plans.reshape(batch * samples, plans.shape[2], plans.shape[3])
            zh = z_hist.repeat_interleave(samples, dim=0)
            ah = a_hist.repeat_interleave(samples, dim=0)
            zg = z_goal.repeat_interleave(samples, dim=0)
            if self.graphed and chunk <= 0:
                # the captured graph owns the whole row count; chunking would
                # change the shape per call and defeat the capture
                graphed_costs = cast(torch.Tensor, self._graph.costs(zh, ah, zg, flat))
                if self.graphed == "verify":
                    eager = score_rows(zh, ah, zg, flat)
                    deviation = float((graphed_costs - eager).abs().max().item())
                    logger.info(f"L2O graphed verify: max_deviation {deviation:.3e}")
                return graphed_costs.view(batch, samples)
            size = flat.shape[0] if chunk <= 0 else chunk
            scored: list[torch.Tensor] = [
                score_rows(
                    zh[start : start + size],
                    ah[start : start + size],
                    zg[start : start + size],
                    flat[start : start + size],
                )
                for start in range(0, flat.shape[0], size)
            ]
            return torch.cat(scored).view(batch, samples)

        return cost

    # ---------------------------------------------------------------- solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        start_time = time.time()
        z_hist, z_goal = self._encode(info_dict)
        encode_seconds = time.time() - start_time
        batch = z_hist.shape[0]
        a_hist = torch.zeros(batch, 2, self.action_dim, device=self.device)
        mean, std = self.net.initial(batch, self.device)

        with torch.no_grad():
            if init_action is not None and init_action.shape[1] > 0:
                # previous decision's unexecuted tail, right-aligned in the plan
                previous = torch.zeros_like(mean)
                tail = init_action.to(device=self.device, dtype=mean.dtype)[:, : self.horizon]
                previous[:, self.horizon - tail.shape[1] :] = tail
                executed = self.horizon - tail.shape[1]
                mean, std = self.net.warm_start(previous, std, executed)

            cost_fn = self._cost_fn(z_hist, a_hist, z_goal)
            mean, std, _ = self.net.plan(cost_fn, mean, std, self.iters)
            final = cost_fn(mean.unsqueeze(1)).squeeze(1)

        plan = mean.detach().to(self.dtype).cpu()
        total_seconds = time.time() - start_time
        logger.info(
            f"L2O solve completed in {total_seconds:.4f} seconds "
            f"(encode {encode_seconds:.4f}, plan {total_seconds - encode_seconds:.4f})"
        )
        return {
            "actions": plan,
            "mean": [plan],
            "var": [std.detach().to(self.dtype).cpu()],
            "costs": final.detach().float().cpu().tolist(),
        }

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "set_align_remaining"):
            model.set_align_remaining(remaining_chunks)
