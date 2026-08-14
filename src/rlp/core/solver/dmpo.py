"""DMPOSolver — plan with a trained DMPO learned-optimizer checkpoint.

*Deep Model Predictive Optimization* (Sacks et al., ICRA 2024) keeps MPC's
sample-rollout-reduce loop and learns the reduction; :class:`DMPONet` holds the
learned pieces and this solver supplies the rollouts and the cost.

Per decision the solver spends ``num_samples * iters`` forward world-model
unrolls and no backward pass, against CEM/MPPI's ``300 x 30 = 9,000`` forward
unrolls and RLP/LIP's ``9 forward + 8 backward``. The checkpoint's sample count
is authoritative: the actor consumes the ``N`` costs positionally, so ``N`` and
the horizon cannot be changed after training. The iteration count can be
(the paper varies it at test time) via ``iters``.

Costs come from the checkpoint's own goal-conditioned value through the frozen
world model — the same cost the optimizer was trained on, and the same critic
the LIP solver plans against, so a DMPO-vs-RLP table isolates the *planner*.

Warm start: the learned shift model consumes the previous decision's plan tail
that :class:`rlp.core.policy.WorldModelPolicy` hands over as ``init_action``.
Under this repository's open-loop protocol (``receding_horizon == horizon``)
nothing survives the shift-forward and the warm start is inert by
construction; evaluate with ``planning.receding_horizon=1`` for the
closed-loop regime DMPO was published in.

Trainer: :mod:`rlp.train.dmpo`.
"""

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rlp.logging import logger

from ..planner.dmpo import DMPONet
from ..rollout import rollout_traj
from ..temporal import trajectory_value, windowed_terminal_value
from ..world_model.protocols import LatentWorldModel
from .lip import EncoderWorldModel, ValueFunction, unwrap_encoder


class DMPOCheckpoint(TypedDict):
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
    temperature: NotRequired[float]
    step_size: NotRequired[float]
    scale_costs: NotRequired[bool]
    gated: NotRequired[bool]
    residual: NotRequired[bool]
    learn_std: NotRequired[bool]
    use_shift: NotRequired[bool]
    gate_activation: NotRequired[str]
    halton: NotRequired[bool]
    seed_val: NotRequired[int]
    temporal_objective: NotRequired[str]
    value_context: NotRequired[int]


__all__ = ["DMPOSolver"]


class DMPOSolver(CEMSolver):
    """Sample-and-reduce planning with DMPO's learned update rule."""

    def __init__(
        self,
        *args: Any,
        actor_path: str = "",
        value_path: str | None = None,
        iters: int | None = None,
        mppi_mode: bool = False,
        cost_chunk: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from rlp.core.value import load_metric

        # mppi_mode runs the same loop with the learned heads bypassed — the
        # hand-written MPPI update DMPO learns a residual on (reference flag
        # ``is_mppi``). It is the ablation the paper reports against and the
        # equivalence test in tests/core/test_dmpo.py.
        self.mppi_mode = bool(mppi_mode)
        self.cost_chunk = int(cost_chunk)

        raw = torch.load(actor_path, map_location=self.device, weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError("DMPO checkpoint must contain a mapping")
        ck = cast(DMPOCheckpoint, raw)
        if ck.get("kind") != "dmpo":
            raise ValueError(f"DMPOSolver: unsupported checkpoint kind {ck.get('kind')!r}")
        self.net = DMPONet(
            horizon=ck["horizon"],
            a_dim=ck["a_dim"],
            num_samples=ck["num_samples"],
            hidden=ck.get("hidden", 256),
            amax=ck.get("amax", 2.5),
            init_std=ck.get("init_std", 1.0),
            temperature=ck.get("temperature", 0.05),
            step_size=ck.get("step_size", 0.8),
            scale_costs=ck.get("scale_costs", True),
            gated=ck.get("gated", True),
            residual=ck.get("residual", True),
            learn_std=ck.get("learn_std", True),
            use_shift=ck.get("use_shift", True),
            gate_activation=ck.get("gate_activation", "tanh"),
            halton=ck.get("halton", True),
            seed_val=ck.get("seed_val", 0),
        ).to(self.device)
        self.net.load_state_dict(ck["sd"])
        self.net.eval()
        self.net.requires_grad_(False)
        self._actor_horizon = int(ck["horizon"])
        self.iters = int(ck["iters"] if iters is None else iters)
        # The actor reads the N costs positionally, so the trained sample count
        # is authoritative; the config value is informational only.
        requested_samples: int = self.num_samples
        if requested_samples != self.net.num_samples:
            logger.info(
                f"DMPO sample count is fixed by the checkpoint: using {self.net.num_samples} "
                f"(config asked for {requested_samples})"
            )
        self.temporal_objective = ck.get("temporal_objective", "terminal")

        value_reference = value_path or ck["value"]
        if value_path is None and not Path(value_reference).exists():
            recorded = Path(value_reference)
            actor_directory = Path(actor_path).resolve().parent
            for candidate in (actor_directory / recorded.name, actor_directory / recorded.stem):
                if candidate.exists():
                    logger.info(f"DMPO value fallback: {value_reference} missing, using sibling {candidate}")
                    value_reference = str(candidate)
                    break
        value_module = load_metric(value_reference, device=self.device)
        value_module.eval()
        self.value = cast(ValueFunction, value_module)
        # window critics score a stack of the last `context` imagined frames,
        # matching MetricCost's deploy-side convention for the baselines
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

        def cost(plans: torch.Tensor) -> torch.Tensor:
            batch, samples = plans.shape[0], plans.shape[1]
            flat = plans.reshape(batch * samples, plans.shape[2], plans.shape[3])
            zh = z_hist.repeat_interleave(samples, dim=0)
            ah = a_hist.repeat_interleave(samples, dim=0)
            zg = z_goal.repeat_interleave(samples, dim=0)
            size = flat.shape[0] if chunk <= 0 else chunk
            scored: list[torch.Tensor] = []
            for start in range(0, flat.shape[0], size):
                stop = start + size
                traj = rollout_traj(wm, zh[start:stop], ah[start:stop], flat[start:stop])
                if self.value_context > 1:
                    scored.append(windowed_terminal_value(self.value, traj, zg[start:stop], self.value_context))
                else:
                    scored.append(
                        trajectory_value(
                            self.value,
                            traj,
                            zg[start:stop],
                            zh[start:stop, -1],
                            self.temporal_objective,
                        )
                    )
            return torch.cat(scored).view(batch, samples)

        return cost

    # ---------------------------------------------------------------- solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        start_time = time.time()
        z_hist, z_goal = self._encode(info_dict)
        batch = z_hist.shape[0]
        a_hist = torch.zeros(batch, 2, self.action_dim, device=self.device)
        mean, std = self.net.initial(batch, self.device)

        with torch.no_grad():
            if init_action is not None and init_action.shape[1] > 0 and not self.mppi_mode:
                # previous decision's unexecuted tail, right-aligned in the plan
                previous = torch.zeros_like(mean)
                tail = init_action.to(device=self.device, dtype=mean.dtype)[:, : self.horizon]
                previous[:, self.horizon - tail.shape[1] :] = tail
                executed = self.horizon - tail.shape[1]
                mean, std = self.net.warm_start(previous, std, executed)

            cost_fn = self._cost_fn(z_hist, a_hist, z_goal)
            if self.mppi_mode:
                for _ in range(self.iters):
                    plans = self.net.plans(mean, std)
                    mean = self.net.mppi_mean(mean, plans, cost_fn(plans)).clamp(-self.net.amax, self.net.amax)
            else:
                mean, std, _ = self.net.plan(cost_fn, mean, std, self.iters)
            final = cost_fn(mean.unsqueeze(1)).squeeze(1)

        plan = mean.detach().to(self.dtype).cpu()
        logger.info(f"DMPO solve completed in {time.time() - start_time:.4f} seconds")
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
