"""IBPSolver — plan with a trained Imagination-Based Planner checkpoint.

*Learning model-based planning from scratch* (Pascanu et al., 2017,
`arXiv:1707.06170 <https://arxiv.org/abs/1707.06170>`_).  The learned pieces
live in :class:`rlp.core.planner.ibp.IBPNet`; this solver supplies the two
things the paper's agent gets from its own learned interaction network — a
one-block imagination step through the frozen world model, and the cost of an
imagined node under the checkpoint's goal-conditioned critic.

Per decision the solver spends at most ``max_imagine`` one-block unrolls plus
the greedy completion of the committed chain — single-digit-to-low-tens forward
unrolls, no backward pass, against CEM/MPPI's ``300 x 30 = 9,000`` forward
unrolls, DMPO's ``256``, and RLP/LIP's ``9 forward + 8 backward``.  The manager
is evaluated greedily (argmax over routes), so a decision is deterministic
given the observation; ``sample_routes=true`` restores the training-time
stochastic manager.

Costs come from the checkpoint's own critic, the same one the LIP and DMPO
solvers plan against, so an IBP-vs-RLP table isolates the *planner*.

Trainer: :mod:`rlp.train.ibp`.
"""

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rlp.logging import logger

from ..planner.ibp import IBPNet
from ..rollout import rollout_traj
from ..world_model.protocols import LatentWorldModel
from .lip import EncoderWorldModel, ValueFunction, unwrap_encoder


class IBPCheckpoint(TypedDict):
    kind: str
    z_dim: int
    horizon: int
    a_dim: int
    value: str
    sd: dict[str, torch.Tensor]
    amax: NotRequired[float]
    hidden: NotRequired[int]
    memory: NotRequired[int]
    max_imagine: NotRequired[int]
    tau: NotRequired[float]
    value_context: NotRequired[int]


__all__ = ["IBPSolver"]


class IBPSolver(CEMSolver):
    """Imagination-tree planning with IBP's learned manager and controller."""

    def __init__(
        self,
        *args: Any,
        actor_path: str = "",
        value_path: str | None = None,
        max_imagine: int | None = None,
        sample_routes: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from rlp.core.value import load_metric

        self.sample_routes = bool(sample_routes)

        raw = torch.load(actor_path, map_location=self.device, weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError("IBP checkpoint must contain a mapping")
        ck = cast(IBPCheckpoint, raw)
        if ck.get("kind") != "ibp":
            raise ValueError(f"IBPSolver: unsupported checkpoint kind {ck.get('kind')!r}")
        self.net = IBPNet(
            z_dim=ck["z_dim"],
            a_dim=ck["a_dim"],
            horizon=ck["horizon"],
            hidden=ck.get("hidden", 256),
            memory=ck.get("memory", 256),
            amax=ck.get("amax", 2.5),
            max_imagine=ck.get("max_imagine", 10),
        ).to(self.device)
        self.net.load_state_dict(ck["sd"])
        self.net.eval()
        self.net.requires_grad_(False)
        self._actor_horizon = int(ck["horizon"])
        # The imagination budget is a deployment knob in the paper (it reports
        # performance against it); the trained default is the checkpoint's.
        self.max_imagine = int(ck.get("max_imagine", 10) if max_imagine is None else max_imagine)

        value_reference = value_path or ck["value"]
        if value_path is None and not Path(value_reference).exists():
            recorded = Path(value_reference)
            actor_directory = Path(actor_path).resolve().parent
            for candidate in (actor_directory / recorded.name, actor_directory / recorded.stem):
                if candidate.exists():
                    logger.info(f"IBP value fallback: {value_reference} missing, using sibling {candidate}")
                    value_reference = str(candidate)
                    break
        value_module = load_metric(value_reference, device=self.device)
        value_module.eval()
        self.value = cast(ValueFunction, value_module)
        latent_dim = int(getattr(value_module, "latent_dim", ck["z_dim"]))
        context = int(ck.get("value_context", max(latent_dim // int(ck["z_dim"]), 1)))
        if context > 1:
            # A node is a single imagined frame, so a window critic has no
            # window to read at the tree's interior nodes.
            raise ValueError("IBPSolver supports single-frame critics only (value_context == 1)")

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

    # ---------------------------------------------------------------- solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        start_time = time.time()
        z_hist, z_goal = self._encode(info_dict)
        wm = cast(LatentWorldModel, self._base())

        def imagine(z_window: torch.Tensor, a_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return rollout_traj(wm, z_window, a_window, action[:, None])[:, -1]

        def value(z: torch.Tensor) -> torch.Tensor:
            return self.value(z, z_goal)

        batch = z_hist.shape[0]
        a_hist = torch.zeros(batch, 2, self.action_dim, device=self.device)
        with torch.no_grad():
            result = self.net.search(
                z_hist,
                a_hist,
                z_goal,
                imagine,
                value,
                sample=self.sample_routes,
                max_imagine=self.max_imagine,
            )

        plan = result.plan.detach().to(self.dtype).cpu()
        logger.info(
            f"IBP solve completed in {time.time() - start_time:.4f} seconds "
            f"(imagined {result.imagined.mean().item():.1f}, unrolls {result.unrolls.mean().item():.1f})"
        )
        return {
            "actions": plan,
            "mean": [plan],
            "var": [torch.zeros_like(plan)],
            "costs": result.cost.detach().float().cpu().tolist(),
        }

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "set_align_remaining"):
            model.set_align_remaining(remaining_chunks)
