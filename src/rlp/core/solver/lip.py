"""LIPSolver — plan with a trained LIP checkpoint against a frozen value fn.

Two world-model families are supported, selected by the ``kind`` field of the
actor checkpoint (written by the training scripts):

- ``kind='lip'``      pooled-latent WMs (e.g. LeWM): state = (B, 3, D) pooled
  history, rollout via :func:`rlp.core.rollout.rollout_terminal`.
- ``kind='lip4'``     LIPv4 (current default): same rollout as ``lip`` but the
  minimal-input, gate-free update rule — f_theta sees only [A, grad_A V, E];
  state and goal reach the planner exclusively through the value function.
- ``kind='lip_dino'`` patch-token WMs (PreJEPA/DINO): state = (B, 3, P, D)
  patch tokens, action embeddings injected per block, rollout via
  :func:`rlp.core.rollout.rollout_terminal_dino`. Plans are trained/rolled in
  normalized action units and denormalized (stats stored in the checkpoint)
  before execution.

At plan time one refinement pass costs 1 differentiated WM unroll per
iteration (trajectory reuse, default since 2026-08-12): 9 forward + 8 backward
unrolls at K=8 versus 9000 forward for the repo's CEM defaults (300 samples x
30 iterations) — a ~360x compute reduction at equal or better success rate.
Measured wall-clock (H200, fp32): 30.7 ms/decision with graphed=true vs CEM's
218.6 ms graphed / 241.9 ms eager (see docs/lip/README_lip.md).

Trainer: ``rlp/train/lip_ac.py`` (see ``rlp/train/rlp.py`` for the composed
replication pipeline).
"""

import time
from pathlib import Path
from typing import Any, NotRequired, Protocol, TypedDict, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rlp.logging import logger

from ..planner import PlannerNet, PlannerNetRec, PlannerNetV3
from ..rollout import rollout_terminal, rollout_terminal_dino, rollout_traj
from ..temporal import trajectory_value, window_pair, windowed_trajectory_value
from ..world_model.protocols import LatentWorldModel, TokenWorldModel


class ValueFunction(Protocol):
    def __call__(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor: ...


class EncoderWorldModel(LatentWorldModel, Protocol):
    wants_proprio: bool
    obs_key: str
    goal_key: str

    def encode(self, info: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]: ...


class TokenEncoderWorldModel(TokenWorldModel, Protocol):
    def encode(self, info: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]: ...


class EnvironmentCost(Protocol):
    def get_cost(self, info: dict[str, Any], actions: torch.Tensor) -> torch.Tensor: ...


class LIPCheckpoint(TypedDict):
    kind: str
    z_dim: int
    horizon: int
    iters: int
    value: str
    sd: dict[str, torch.Tensor]
    a_dim: NotRequired[int]
    amax: NotRequired[float]
    feed: NotRequired[str]
    hidden: NotRequired[int]
    s_dim: NotRequired[int]
    s0_mode: NotRequired[str]
    width: NotRequired[int]
    layers: NotRequired[int]
    goal_mode: NotRequired[str]
    gd_init: NotRequired[float]
    feat_norm: NotRequired[bool]
    vscale: NotRequired[float]
    iter_mode: NotRequired[str]
    head_mode: NotRequired[str]
    cond_mode: NotRequired[str]
    pre_ln: NotRequired[bool]
    use_zg: NotRequired[bool]
    use_gate: NotRequired[bool]
    use_z0: NotRequired[bool]
    use_grad: NotRequired[bool]
    amu5: NotRequired[torch.Tensor]
    ast5: NotRequired[torch.Tensor]
    temporal_objective: NotRequired[str]
    vnorm: NotRequired[str]
    window_frames: NotRequired[int]
    window_lag: NotRequired[int | None]


# Mech-interp hook: ``probe_directory`` dumps per-replan
# (z0, imagined-terminal latent, its value, goal latent) so an open-loop eval
# yields imagined-vs-actually-reached (consecutive replans) for the A/B probe.
_LIP_PROBE_N = 0


__all__ = [
    "LIPSolver",
    "unwrap_encoder",
]


def unwrap_encoder(model: torch.nn.Module) -> torch.nn.Module:
    """Peel planning-cost wrappers off ``model`` until an encoder WM appears.

    The eval driver hands the solver a cost stack — ``MetricCost`` holds its
    inner model at ``.base`` and ``LatentGoalCost`` at ``.model``, and for
    LeWM/PLDM the two nest (``MetricCost.base`` is a ``LatentGoalCost``). LIP
    needs the raw world model underneath (``encode``/latent rollout).
    """
    candidate: torch.nn.Module = model
    for _ in range(4):
        if callable(getattr(candidate, "encode", None)):
            return candidate
        inner = getattr(candidate, "base", None)
        if inner is None:
            inner = getattr(candidate, "model", None)
        if not isinstance(inner, torch.nn.Module):
            break
        candidate = inner
    if not callable(getattr(candidate, "encode", None)):
        raise TypeError(f"{type(model).__name__} does not wrap an encoder world model")
    return candidate


class LIPSolver(CEMSolver):
    """Plan with a trained LIP checkpoint; optionally refine with MPPI.

    With ``n_steps=0`` (the benchmarked configuration) the solver is fully
    deterministic: K learned refinement iterations, execute the plan. With
    ``restarts=R > 1`` it runs R noisy-initialized refinements per env and
    picks by terminal value (argmin-V; ``robust_m > 0`` averages the value of
    m perturbed copies instead). ``n_steps > 0`` additionally runs MPPI
    (softmax-weighted, temperature ``lam``) seeded from the LIP plan.

    The plan length (horizon) always comes from the actor checkpoint.
    """

    def __init__(
        self,
        *args: Any,
        actor_path: str = "",
        value_path: str | None = None,
        lam: float = 1.0,
        restarts: int = 1,
        restart_noise: float = 0.5,
        lip_select: str = "last",
        robust_m: int = 0,
        init_mode: str = "zero",
        init_samples: int = 64,
        init_scale: float = 1.5,
        reuse_trajectory: bool | str = True,
        graphed: bool | str = False,
        record_probes: bool = False,
        probe_directory: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from rlp.core.value import load_metric

        self.lam = lam
        self.restarts = int(restarts)
        self.restart_noise = restart_noise
        self.lip_select = lip_select
        self.robust_m = int(robust_m)
        # value-guided initialization: A(0) = argmin-E over a candidate set of
        # RAW plans (zero + iid-Gaussian + time-tiled Gaussian), scored once
        # before refinement. Unlike `restarts` (noise around zero, argmin over
        # REFINED plans — the configuration that hurt on v2WM), selection here
        # happens on unrefined samples, exactly CEM's iteration-0, and the
        # refinement trajectory stays single and deterministic.
        self.init_mode = init_mode
        self.init_samples = int(init_samples)
        self.init_scale = float(init_scale)
        # Default TRUE since 2026-08-12: the gradient unroll's trajectory is
        # reused as the actor features (9 fwd + 8 bwd instead of 17 fwd +
        # 8 bwd). The discarded unroll was a bit-exact duplicate — verified by
        # reuse_trajectory="verify" (max_difference == 0 per iteration) and by
        # end-task parity on the reacher winners (jobs 4927/4930: 3 of 4
        # report cells identical, 1 differs by a single episode in 450).
        # Pass reuse_trajectory=false to reproduce pre-2026-08-12 evals.
        self.reuse_trajectory = reuse_trajectory
        # Opt-in CUDA-graphed inference (see solver/graphed.py). False = the
        # untouched eager path; True = graph-captured refinement (implies
        # trajectory reuse); "verify" = graphed, plus an eager recomputation
        # each iteration with the max deviation logged. Lazily initialised on
        # first solve so construction stays checkpoint-only.
        self.graphed = graphed
        self._graphed_refinement: Any = None
        self.probe_directory = Path(probe_directory) if record_probes and probe_directory else None
        if self.probe_directory is not None:
            self.probe_directory.mkdir(parents=True, exist_ok=True)

        raw_checkpoint = torch.load(actor_path, map_location=self.device, weights_only=False)
        if not isinstance(raw_checkpoint, dict):
            raise TypeError("LIP checkpoint must contain a mapping")
        ck = cast(LIPCheckpoint, raw_checkpoint)
        if ck.get("vnorm", "none") != "none":
            # The port's PlannerNet consumes the raw critic value E. Loading a
            # vnorm-trained state_dict here would deploy the actor on an input
            # scale it never saw — in_dim is identical, so it fails silently and
            # only shows up as a mysteriously weak success rate.
            raise ValueError(
                f"checkpoint was trained with vnorm={ck['vnorm']!r}; this "
                "PlannerNet has no vnorm support and would silently feed raw E. "
                "Evaluate through scripts/sky/overlays/lip.py, or port the flag."
            )
        self.kind = ck.get("kind")
        if self.kind not in (
            "lip",
            "lip2",
            "lip3",
            "lip4",
            "lip4r",
            "lip_dino",
        ):
            raise ValueError(f"LIPSolver: unsupported checkpoint kind {self.kind!r}")
        self.feed = ck.get("feed", "none") if self.kind == "lip2" else "none"
        self.actor: PlannerNet | PlannerNetRec | PlannerNetV3
        if self.kind == "lip4r":
            self.actor = PlannerNetRec(
                ck["z_dim"],
                horizon=ck["horizon"],
                a_dim=ck.get("a_dim", 25),
                hidden=ck.get("hidden", 512),
                s_dim=ck.get("s_dim", 256),
                amax=ck.get("amax", 2.5),
                use_z0=ck.get("use_z0", False),
                use_zg=ck.get("use_zg", False),
                use_grad=ck.get("use_grad", True),
                s0_mode=ck.get("s0_mode", "zero"),
            ).to(self.device)
        elif self.kind == "lip3":
            self.actor = PlannerNetV3(
                ck["z_dim"],
                horizon=ck["horizon"],
                a_dim=ck.get("a_dim", 25),
                width=ck.get("width", 256),
                layers=ck.get("layers", 2),
                amax=ck.get("amax", 2.5),
                n_iters=ck["iters"],
                goal_mode=ck.get("goal_mode", "diff"),
                gd_init=ck.get("gd_init", 0.0),
                feat_norm=ck.get("feat_norm", False),
                vscale=ck.get("vscale", 25.0),
                iter_mode=ck.get("iter_mode", "emb"),
                head_mode=ck.get("head_mode", "gate"),
                cond_mode=ck.get("cond_mode", "token"),
                use_gate=ck.get("use_gate", True),
                pre_ln=ck.get("pre_ln", False),
            ).to(self.device)
        else:
            # lip4 checkpoints store the flags explicitly; the flipped ck.get
            # defaults only guard hand-rolled checkpoints of each kind
            v4 = self.kind == "lip4"
            self.actor = PlannerNet(
                ck["z_dim"],
                horizon=ck["horizon"],
                a_dim=ck.get("a_dim", 25),
                amax=ck.get("amax", 2.5),
                feed=self.feed,
                use_zg=ck.get("use_zg", not v4),
                use_gate=ck.get("use_gate", not v4),
                use_z0=ck.get("use_z0", not v4),
                use_grad=ck.get("use_grad", True),
            ).to(self.device)
        self.actor.load_state_dict(ck["sd"])
        self.actor.eval()
        self._actor_horizon = ck["horizon"]  # plan length must match the trained net
        self.lip_iters = ck["iters"]
        self.temporal_objective = ck.get("temporal_objective", "terminal")
        if self.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
            raise ValueError(f"unsupported temporal objective: {self.temporal_objective}")
        value_reference = value_path or ck["value"]
        if value_path is None and not Path(value_reference).exists():
            # The training run records an absolute value path; when evaluating
            # on another machine, fall back to the sibling the checkpoints were
            # copied out with (planner.pt next to value_ac). Vendored values
            # are artifact directories named by the recorded file's stem, since
            # a directory ending in .pt breaks upstream checkpoint resolution.
            recorded = Path(value_reference)
            actor_directory = Path(actor_path).resolve().parent
            for candidate in (actor_directory / recorded.name, actor_directory / recorded.stem):
                if candidate.exists():
                    logger.info(f"LIP value fallback: {value_reference} missing, using sibling {candidate}")
                    value_reference = str(candidate)
                    break
        value_module = load_metric(value_reference, device=self.device)
        value_module.eval()
        self.lip_value = cast(ValueFunction, value_module)
        # m-frame window values score a stack of the last `vframes` imagined
        # frames with the (static) goal frame duplicated to match. Trainers
        # record `window_frames` in the planner checkpoint; older checkpoints
        # referencing a windowed value fall back to the declared latent widths.
        value_dim = int(getattr(value_module, "latent_dim", ck["z_dim"]))
        declared_frames = ck.get("window_frames")
        self.vframes = int(declared_frames) if declared_frames is not None else max(value_dim // int(ck["z_dim"]), 1)
        if self.vframes > 1:
            logger.info(f"LIP window value: {self.vframes} frames")
            if self.kind == "lip_dino":
                raise ValueError("windowed values are not supported for lip_dino checkpoints")
            if self.graphed:
                raise ValueError("graphed LIP inference does not support windowed values")
            gm = getattr(self.actor, "goal_mode", None)
            if gm == "sep" or (gm == "vonly" and getattr(self.actor, "cond", None) is None):
                raise ValueError("windowed values do not support per-step value conditioning (goal_mode=sep/vonly)")
        if self.kind == "lip_dino":
            if "amu5" not in ck or "ast5" not in ck:
                raise ValueError("lip_dino checkpoint lacks action statistics")
            self._amu = ck["amu5"].to(self.device).float()
            self._ast = ck["ast5"].to(self.device).float()

    @property
    def horizon(self) -> int:
        ah = getattr(self, "_actor_horizon", None)
        return int(ah if ah else self._config.horizon)

    def _base(self) -> torch.nn.Module:
        return unwrap_encoder(self.model)

    def _score(self, trajectory: torch.Tensor, goal: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Trajectory score; stacks the last ``vframes`` imagined frames and
        duplicates the observed goal frame when the value is windowed."""
        if self.vframes > 1:
            return windowed_trajectory_value(
                self.lip_value, trajectory, goal, history, self.vframes, self.temporal_objective
            )
        return trajectory_value(self.lip_value, trajectory, goal, history[:, -1], self.temporal_objective)

    def _value_init(
        self,
        wm: LatentWorldModel,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        zg: torch.Tensor,
    ) -> torch.Tensor:
        """A(0) = argmin-E over {zero, iid-Gaussian, time-tiled Gaussian} raw plans.

        The tiled half (one action block repeated across the horizon) covers
        sustained-motion plans — the long-transport family the zero-init
        refinement cannot reach when the value gradient is flat at A=0.
        """
        B, H, adim = z_hist.shape[0], self.horizon, self.action_dim
        amax = float(self.actor.amax)
        n_iid = self.init_samples // 2
        n_tile = self.init_samples - n_iid
        iid = self.init_scale * torch.randn(B, n_iid, H, adim, device=self.device, generator=self.torch_gen)
        tile = self.init_scale * torch.randn(B, n_tile, 1, adim, device=self.device, generator=self.torch_gen)
        cand = torch.cat(
            [
                torch.zeros(B, 1, H, adim, device=self.device),
                iid,
                tile.expand(B, n_tile, H, adim),
            ],
            dim=1,
        )
        cand = cand.clamp(-amax, amax)
        C = cand.shape[1]
        with torch.no_grad():
            zg_c = zg.repeat_interleave(C, dim=0)
            if self.vframes > 1:
                # window values need the full trajectory to stack terminal frames
                trajectory = rollout_traj(
                    wm,
                    z_hist.repeat_interleave(C, dim=0),
                    a_hist.repeat_interleave(C, dim=0),
                    cand.reshape(B * C, H, adim),
                )
                E = self.lip_value(*window_pair(trajectory.float(), zg_c, self.vframes)).view(B, C)
            else:
                term = rollout_terminal(
                    wm,
                    z_hist.repeat_interleave(C, dim=0),
                    a_hist.repeat_interleave(C, dim=0),
                    cand.reshape(B * C, H, adim),
                )
                E = self.lip_value(term.float(), zg_c).view(B, C)
        return cand[torch.arange(B, device=self.device), E.argmin(dim=1)]

    # ------------------------------------------------------------------ lip
    def _graphed_for(self, wm: Any, z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor) -> Any:
        """Lazily build the CUDA-graphed refinement and stage this decision.

        Kept out of ``__init__`` so solver construction never touches CUDA
        graphs; the first solve pays a one-time capture per batch size.
        """
        if self._graphed_refinement is None:
            from .graphed import GraphedRefinement

            if self.reuse_trajectory:
                logger.info("graphed LIP inference already reuses the gradient unroll; reuse_trajectory is redundant")
            self._graphed_refinement = GraphedRefinement(
                wm,
                self.lip_value,
                self.temporal_objective,
                self.horizon,
                self.action_dim,
                z_hist.shape[-1],
                self.device,
            )
        self._graphed_refinement.bind(z_hist, a_hist, z_goal)
        return self._graphed_refinement

    def _proposal_lip(self, info_dict: dict[str, Any], n_envs: int) -> torch.Tensor:
        encode_start = time.time()
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
            enc = wm.encode(enc_in)
            z_hist = enc["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:  # pad short history at episode start
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            genc_in = {"pixels": gx}
            if getattr(wm, "wants_proprio", False):
                gpro = info_dict.get("goal_state")
                if gpro is None:
                    raise KeyError(
                        "proprio-variant WM: info_dict lacks 'goal_state' (goal agent position for the goal latent)"
                    )
                gpro = torch.as_tensor(np.asarray(gpro), dtype=torch.float32, device=self.device)
                gpro = gpro.reshape(gx.shape[0], -1)[:, -2:]  # last frame's (x, y)
                genc_in["proprio"] = gpro.unsqueeze(1).expand(-1, gx.shape[1], -1)
            zg = wm.encode(genc_in)["emb"][:, -1].float()

        # everything above is observation/goal encoding; everything below is
        # planning. The split matters: only the refinement is graph-captured,
        # so a whole-decision number is not comparable to a refinement number.
        self._last_encode_seconds = time.time() - encode_start
        R = max(1, self.restarts)
        B = n_envs
        zh_r = z_hist.repeat_interleave(R, dim=0)
        zg_r = zg.repeat_interleave(R, dim=0)
        z0_r = zh_r[:, -1]
        a_hist = torch.zeros(B * R, 2, self.action_dim, device=self.device)
        A = self.restart_noise * torch.randn(
            B * R,
            self.horizon,
            self.action_dim,
            device=self.device,
            generator=self.torch_gen,
        )
        A[::R] = 0.0  # one zero-init restart per env
        if self.init_mode == "value" and R == 1:
            A = self._value_init(wm, z_hist, a_hist, zg)
        s = self.actor.init_state(A.shape[0], z0_r) if isinstance(self.actor, PlannerNetRec) else None
        graphed_ref = self._graphed_for(wm, zh_r, a_hist, zg_r) if self.graphed else None
        buf: list[torch.Tensor] = []
        for k_it in range(self.lip_iters):
            if graphed_ref is not None:
                # one forward-graph + one backward-graph replay; the gradient
                # unroll's trajectory doubles as the actor features (reuse)
                E_graphed, traj_f, gA = graphed_ref.step(A)
                if self.graphed == "verify":
                    with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                        A_ref = A.detach().requires_grad_(True)
                        traj_ref = rollout_traj(wm, zh_r, a_hist, A_ref)
                        score_ref = trajectory_value(self.lip_value, traj_ref, zg_r, z0_r, self.temporal_objective)
                        (gA_ref,) = torch.autograd.grad(score_ref.sum(), A_ref)
                    logger.info(
                        f"Graphed verification iteration={k_it} "
                        f"gradient={float((gA - gA_ref).abs().max()):.3e} "
                        f"score={float((E_graphed - score_ref.detach()).abs().max()):.3e} "
                        f"trajectory={float((traj_f - traj_ref.detach()).abs().max()):.3e}"
                    )
            else:
                with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                    A_in = A.detach().requires_grad_(True)
                    traj = rollout_traj(wm, zh_r, a_hist, A_in)
                    score = self._score(traj, zg_r, zh_r)
                    (gA,) = torch.autograd.grad(score.sum(), A_in)
            with torch.no_grad():
                if graphed_ref is not None:
                    E = E_graphed
                else:
                    reuse = self.reuse_trajectory
                    if reuse == "verify":
                        traj_f = rollout_traj(wm, zh_r, a_hist, A)
                        difference = (traj_f - traj.detach()).abs().max().item()
                        logger.info(f"Reuse verification iteration={k_it} max_difference={difference:.3e}")
                    elif reuse and reuse != "0":
                        traj_f = traj.detach()
                    else:
                        traj_f = rollout_traj(wm, zh_r, a_hist, A)
                    E = self._score(traj_f, zg_r, zh_r)
                vtraj = None
                gm = getattr(self.actor, "goal_mode", None)
                if gm == "sep" or (gm == "vonly" and getattr(self.actor, "cond", None) is None):
                    Bh, H = traj_f.shape[0], traj_f.shape[1]
                    vtraj = self.lip_value(
                        traj_f.reshape(Bh * H, -1),
                        zg_r.repeat_interleave(H, dim=0),
                    ).view(Bh, H)
                if self.kind == "lip4r":
                    if not isinstance(self.actor, PlannerNetRec) or s is None:
                        raise RuntimeError("lip4r checkpoint did not create a recurrent actor state")
                    A, s = self.actor(A, gA, E, z0_r, zg_r, s, traj_f, k=k_it, vtraj=vtraj)
                else:
                    if isinstance(self.actor, PlannerNetRec):
                        raise RuntimeError("non-recurrent checkpoint created a recurrent actor")
                    A = self.actor(A, gA, E, z0_r, zg_r, traj_f, k=k_it, vtraj=vtraj)
                buf.append(A.clone())
        with torch.no_grad():
            cands = torch.stack(buf if self.lip_select == "buffer" else buf[-1:], dim=1)
            C = cands.shape[1]
            cf = cands.reshape(B * R * C, self.horizon, self.action_dim)
            zh_c = zh_r.repeat_interleave(C, dim=0)
            ah_c = a_hist.repeat_interleave(C, dim=0)
            zg_c = zg_r.repeat_interleave(C, dim=0)
            if self.robust_m > 0:  # value of m perturbed copies (robust argmin)
                scores = torch.zeros(cf.shape[0], device=cf.device)
                amax = self.actor.amax
                for _ in range(self.robust_m):
                    pert = (cf + 0.1 * torch.randn_like(cf)).clamp(-amax, amax)
                    perturbed = rollout_traj(wm, zh_c, ah_c, pert)
                    scores = scores + self._score(perturbed, zg_c, zh_c)
                Ef = (scores / self.robust_m).view(B, R * C)
            elif graphed_ref is not None and C == 1:
                # selection unroll through the same captured graph (shape matches)
                Ef = graphed_ref.score(cf).view(B, R * C)
            else:
                final_trajectory = rollout_traj(wm, zh_c, ah_c, cf)
                Ef = self._score(final_trajectory, zg_c, zh_c).view(B, R * C)
            best = Ef.argmin(dim=1)
            A = cands.view(B, R * C, self.horizon, self.action_dim)[torch.arange(B, device=self.device), best]
        if self.probe_directory is not None:
            global _LIP_PROBE_N
            with torch.no_grad():
                z_traj = rollout_traj(wm, z_hist, a_hist[:B], A)  # (B,H,D) imagined path
                z_imag = z_traj[:, -1]
                e_imag = self._score(z_traj, zg, z_hist)
                # candidate population under both objectives (probe-only rollout;
                # Ef is the critic energy the deployed argmin actually used)
                traj_c = rollout_traj(wm, zh_c, ah_c, cf)
                lat_pop = (traj_c[:, -1] - zg_c).norm(dim=-1).view(B, R * C)
                lat_traj = (z_traj - zg.unsqueeze(1)).norm(dim=-1)  # (B,H)
            torch.save(
                {
                    "z0": z_hist[:, -1].detach().cpu(),
                    "z_traj": z_traj.detach().cpu(),
                    "z_imag": z_imag.detach().cpu(),
                    "A": A.detach().cpu(),
                    "E": e_imag.detach().cpu(),
                    "zg": zg.detach().cpu(),
                    "Ef_pop": Ef.detach().cpu(),
                    "lat_pop": lat_pop.detach().cpu(),
                    "best": best.detach().cpu(),
                    "lat_traj": lat_traj.detach().cpu(),
                    "cands": cands.view(B, R * C, self.horizon, self.action_dim).detach().cpu(),
                },
                self.probe_directory / f"probe_{_LIP_PROBE_N:04d}.pt",
            )
            _LIP_PROBE_N += 1
        return A.detach().to(self.dtype)

    # ------------------------------------------------------------- lip_dino
    def _proposal_lip_dino(self, info_dict: dict[str, Any], n_envs: int) -> torch.Tensor:
        del n_envs
        wm = cast(TokenEncoderWorldModel, self._base())
        px = info_dict["pixels"].to(self.device).float()[:, -3:]
        B = px.shape[0]
        pro = info_dict.get("proprio")
        pro = (
            pro.to(self.device).float()[:, -3:]
            if pro is not None
            else torch.zeros(B, px.shape[1], 41, device=self.device)
        )
        if px.shape[1] < 3:
            k = 3 - px.shape[1]
            px = torch.cat([px[:, :1].expand(-1, k, -1, -1, -1), px], dim=1)
            pro = torch.cat([pro[:, :1].expand(-1, k, -1), pro], dim=1)
        with torch.no_grad():
            info = {
                "pixels": px,
                "proprio": pro,
                "action": torch.zeros(B, 3, self.action_dim, device=self.device),
            }
            toks = wm.encode(info)["emb"].float()  # (B,3,P,D)
            gx = info_dict["goal"].to(self.device).float()
            genc = wm.encode({"pixels": gx}, emb_keys=[], target="gemb")
            zg = genc["pixels_gemb"][:, -1].mean(dim=1).float()
            z0 = toks[:, -1, :, :384].mean(dim=1)
        A = torch.zeros(B, self.horizon, self.action_dim, device=self.device)
        for _ in range(self.lip_iters):
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                A_in = A.detach().requires_grad_(True)
                zT = rollout_terminal_dino(wm, toks, A_in * self._ast + self._amu)
                (gA,) = torch.autograd.grad(self.lip_value(zT, zg).sum(), A_in)
            with torch.no_grad():
                E = self.lip_value(
                    rollout_terminal_dino(wm, toks, A * self._ast + self._amu),
                    zg,
                )
                if isinstance(self.actor, PlannerNetRec):
                    raise RuntimeError("DINO checkpoint cannot use a recurrent actor")
                A = self.actor(A, gA, E, z0, zg)
        return (A * self._ast + self._amu).to(self.dtype)  # raw action units

    # ------------------------------------------------------------------ solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action
        start_time = time.time()
        outputs: dict[str, Any] = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))

        if self.kind == "lip_dino":
            proposal = self._proposal_lip_dino(info_dict, total_envs)
        else:
            proposal = self._proposal_lip(info_dict, total_envs)
        mean = proposal
        var = self.var_scale * torch.ones_like(mean)

        # optional MPPI refinement around the LIP plan (n_steps=0 -> pure LIP)
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded: dict[str, Any] = {}
            for k, v in info_dict.items():
                vb = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    td = self.dtype if vb.is_floating_point() else None
                    vb = vb.to(device=self.device, dtype=td).unsqueeze(1).expand(bs, self.num_samples, *vb.shape[1:])
                elif isinstance(v, np.ndarray):
                    vb = np.repeat(vb[:, None, ...], self.num_samples, axis=1)
                expanded[k] = vb

            final_cost = [0.0] * bs
            for _ in range(self.n_steps):
                cand = torch.randn(
                    bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                    dtype=self.dtype,
                )
                cand = cand * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                cand[:, 0] = batch_mean
                costs = cast(EnvironmentCost, self.model).get_cost(expanded, cand)
                w = torch.softmax(-costs.float() / self.lam, dim=1)
                batch_mean = (w[..., None, None] * cand.float()).sum(dim=1).to(self.dtype)
                spread = (cand.float() - batch_mean.unsqueeze(1).float()) ** 2
                batch_var = (w[..., None, None] * spread).sum(dim=1).sqrt().clamp(min=0.05).to(self.dtype)
                final_cost = (w * costs.float()).sum(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        total_seconds = time.time() - start_time
        encode_seconds = getattr(self, "_last_encode_seconds", 0.0)
        logger.info(
            f"LIP solve completed in {total_seconds:.4f} seconds "
            f"(encode {encode_seconds:.4f}, plan {total_seconds - encode_seconds:.4f})"
        )
        return outputs
