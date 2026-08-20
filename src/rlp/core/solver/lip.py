import time
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rlp.utils.logging import logger

from ..planner import PlannerNet
from ..rollout import rollout_terminal, rollout_traj
from ..temporal import trajectory_value, window_pair, windowed_trajectory_value
from ..world_model.protocols import LatentWorldModel


class ValueFunction(Protocol):
    def __call__(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor: ...


class EncoderWorldModel(LatentWorldModel, Protocol):
    wants_proprio: bool
    obs_key: str
    goal_key: str

    def encode(self, info: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]: ...


class EnvironmentCost(Protocol):
    def get_cost(self, info: dict[str, Any], actions: torch.Tensor) -> torch.Tensor: ...


def _checkpoint_value(checkpoint: dict[str, Any], key: str, legacy_key: str) -> Any:
    if key in checkpoint:
        return checkpoint[key]
    if legacy_key in checkpoint:
        return checkpoint[legacy_key]
    raise KeyError(f"planner checkpoint is missing {key!r}")


# Mech-interp hook: ``probe_directory`` dumps per-replan
# (z0, imagined-terminal latent, its value, goal latent) so an open-loop eval
# yields imagined-vs-actually-reached (consecutive replans) for the A/B probe.
_LIP_PROBE_N = 0


__all__ = [
    "LIPSolver",
    "unwrap_encoder",
]


def unwrap_encoder(model: Any) -> torch.nn.Module:
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
    deterministic: one pass of K learned refinement iterations from a single
    initialization, then execute the plan. ``n_steps > 0`` additionally runs MPPI
    (softmax-weighted, temperature ``lam``) seeded from the LIP plan.

    The plan length (horizon) always comes from the actor checkpoint.
    """

    def __init__(
        self,
        *args: Any,
        actor_path: str,
        value_path: str | None,
        lam: float,
        lip_select: str,
        init_mode: str,
        init_samples: int,
        init_scale: float,
        cem_init_steps: int,
        iters_override: int | None,
        graphed: bool | str,
        graph_warmup_iters: int,
        record_probes: bool,
        probe_directory: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from rlp.core.value import load_metric

        self.lam = lam
        self.lip_select = lip_select
        # update_rule="gradient" replaces the LEARNED update with a plain
        # gradient step of the same energy, inside the same loop: same init,
        # same _score, same K, same clip, so the only difference is the rule.
        # It answers whether the exploitation E1 measured (a CEM plan worth
        # 79.6 refined down to 68.9) belongs to the learned rule or to the
        # energy landscape any descent direction would follow.
        if update_rule not in {"learned", "gradient"}:
            raise ValueError(f"unsupported update rule: {update_rule}")
        self.update_rule = update_rule
        self.gd_lr = float(gd_lr)
        # band-limited updates (PUSHT_DIAG E25): every refinement step is
        # projected onto the first `band_limit` temporal DCT modes before it is
        # applied. Built from zero the refiner's plans are DC-dominated
        # (high/low power 0.45-0.62) like a real action sequence, but the
        # correction it applies to a good plan is high-frequency chatter
        # (1.66-1.94) that buys nothing. This constrains only the update's
        # temporal bandwidth -- the energy, the world model and the data are
        # untouched, and it is not a behaviour-cloning anchor.
        self.band_limit = None if band_limit in (None, 0) else int(band_limit)
        self._band_proj: torch.Tensor | None = None
        # rlp.core.policy.WorldModelPolicy publishes `pixels_hist` (frames one action
        # block apart) and `action_hist` (the primitive actions between them) on every
        # replan once planning.history_len > 1; before E29 this solver ignored both and
        # tiled the current frame. Frames are used whenever present; the real executed
        # action history is opt-in so the two halves can be tested separately (the actor
        # trained on zero-action replay samples, so zeros are in-distribution for it)
        self.use_action_history = bool(use_action_history)
        # physics grounding (rlp.core.grounding): checkpoints trained with it carry
        # the probe and thresholds; the same term joins the deployed energy. The
        # config's ground_weight overrides the checkpoint's (0 = deploy without it).
        self.ground_weight_override = None if ground_weight is None else float(ground_weight)
        self.grounding: GroundingPenalty | None = None
        self._ground_agent0: torch.Tensor | None = None
        self._ground_u_prev: torch.Tensor | None = None
        # value-guided initialization: A(0) = argmin-E over a candidate set of
        # RAW plans (zero + iid-Gaussian + time-tiled Gaussian), scored once
        # before refinement. Selection happens on UNREFINED samples, exactly
        # CEM's iteration-0, so the refinement trajectory itself stays single
        # and deterministic.
        self.init_mode = init_mode
        self.init_samples = int(init_samples)
        self.init_scale = float(init_scale)
        # init_mode="cem": A(0) = the mean of a CEM run on the solver's planning
        # cost (num_samples x cem_init_steps, topk elites), then the K learned
        # refinements as usual. An oracle-initialisation diagnostic: separates
        # "the refiner cannot find the basin" from "the refiner cannot hold a
        # good plan".
        self.cem_init_steps = int(cem_init_steps)
        # iters_override: deploy-time K (None = the checkpoint's trained K).
        # 0 = emit the initial plan untouched; >K = continuation diagnostic.
        self.iters_override = None if iters_override is None else int(iters_override)
        # Opt-in CUDA-graphed inference (see solver/graphed.py). False = the
        # untouched eager path; True = graph-captured refinement;
        # "verify" = graphed, plus an eager recomputation
        # each iteration with the max deviation logged. Lazily initialised on
        # first solve so construction stays checkpoint-only.
        self.graphed = graphed
        self.graph_warmup_iters = graph_warmup_iters
        self._graphed_refinement: Any = None
        self.probe_directory = Path(probe_directory) if record_probes and probe_directory else None
        if self.probe_directory is not None:
            self.probe_directory.mkdir(parents=True, exist_ok=True)

        raw_checkpoint = torch.load(actor_path, map_location=self.device, weights_only=False)
        if not isinstance(raw_checkpoint, dict):
            raise TypeError("LIP checkpoint must contain a mapping")
        checkpoint = cast(dict[str, Any], raw_checkpoint)
        if checkpoint.get("vnorm", "none") != "none":
            # PlannerNet consumes the raw critic value E; a vnorm-trained
            # state_dict has the same in_dim, so it would load and silently
            # deploy on an input scale it never saw.
            raise ValueError(
                f"checkpoint was trained with vnorm={checkpoint['vnorm']!r}, which PlannerNet does not support"
            )
        horizon = int(checkpoint["horizon"])
        self.actor = PlannerNet(
            horizon=horizon,
            action_dim=int(_checkpoint_value(checkpoint, "action_dim", "a_dim")),
            hidden_dim=int(_checkpoint_value(checkpoint, "hidden_dim", "hidden")),
            action_limit=float(_checkpoint_value(checkpoint, "action_limit", "amax")),
            head_scale=float(checkpoint["head_scale"]),
        ).to(self.device)
        state_dict = _checkpoint_value(checkpoint, "state_dict", "sd")
        if not isinstance(state_dict, dict):
            raise TypeError("planner checkpoint state_dict must contain a mapping")
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self._actor_horizon = horizon
        self.lip_iterations = int(_checkpoint_value(checkpoint, "iterations", "iters"))
        if self.iters_override is not None:
            logger.info(f"LIP iterations overridden at deploy: {self.lip_iterations} -> {self.iters_override}")
            self.lip_iterations = self.iters_override
        self.temporal_objective = str(checkpoint["temporal_objective"])
        if self.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
            raise ValueError(f"unsupported temporal objective: {self.temporal_objective}")
        value_reference = value_path or str(checkpoint["value"])
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
        # record `window_frames` in the planner checkpoint; checkpoints without
        # it fall back to the declared latent widths.
        value_dim = int(getattr(value_module, "latent_dim", 0))
        declared_frames = checkpoint.get("window_frames")
        if declared_frames is not None:
            self.vframes = int(declared_frames)
        elif "z_dim" in checkpoint and value_dim:
            self.vframes = max(value_dim // int(checkpoint["z_dim"]), 1)
        else:
            self.vframes = 1
        if self.vframes > 1:
            logger.info(f"LIP window value: {self.vframes} frames")
            if self.graphed:
                raise ValueError("graphed LIP inference does not support windowed values")

    def _band_projection(self) -> torch.Tensor:
        """P = B B^T onto the first M temporal DCT-II modes, cached."""
        if self._band_proj is None:
            h, m = self.horizon, int(self.band_limit or 0)
            t = torch.arange(h, dtype=torch.float32, device=self.device)
            k = torch.arange(m, dtype=torch.float32, device=self.device)
            basis = torch.cos(torch.pi * (t[:, None] + 0.5) * k[None, :] / h)  # (H, M)
            basis = basis / basis.norm(dim=0, keepdim=True)  # orthonormal columns
            self._band_proj = basis @ basis.T  # (H, H)
        return self._band_proj

    def _apply_update(self, a_prev: torch.Tensor, a_new: torch.Tensor) -> torch.Tensor:
        """Keep only the low-frequency part of the step, then re-clip."""
        if self.band_limit is None:
            return a_new
        delta = torch.einsum("ij,bja->bia", self._band_projection(), a_new - a_prev)
        amax = float(self.actor.amax)
        return (a_prev + delta).clamp(-amax, amax)

    @property
    def horizon(self) -> int:
        return self._actor_horizon

    def _base(self) -> torch.nn.Module:
        return unwrap_encoder(self.model)

    def _score(
        self,
        trajectory: torch.Tensor,
        goal: torch.Tensor,
        history: torch.Tensor,
        plan: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Trajectory score; stacks the last ``vframes`` imagined frames and
        duplicates the observed goal frame when the value is windowed. With a
        grounding term loaded, ``plan`` (the actions that produced ``trajectory``)
        adds the physics penalty -- the energy the actor was trained on.
        """
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
        action_limit = self.actor.action_limit
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
        cand = cand.clamp(-action_limit, action_limit)
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

    def _cem_init(self, info_dict: dict[str, Any], n_envs: int) -> torch.Tensor:
        """A(0) = CEM mean on the solver's planning cost (N(mean, var) samples,
        top-k elites, mean/std update), batched exactly like :meth:`solve`'s
        MPPI loop. Returns (B, H, adim)."""
        H, adim = self.horizon, self.action_dim
        mean = torch.zeros(n_envs, H, adim, device=self.device, dtype=self.dtype)
        var = self.var_scale * torch.ones_like(mean)
        with torch.no_grad():
            for start_idx in range(0, n_envs, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_envs)
                bs = end_idx - start_idx
                batch_mean, batch_var = mean[start_idx:end_idx], var[start_idx:end_idx]
                expanded: dict[str, Any] = {}
                for k, v in info_dict.items():
                    vb = v[start_idx:end_idx]
                    if torch.is_tensor(v):
                        td = self.dtype if vb.is_floating_point() else None
                        tail = vb.shape[1:]
                        vb = vb.to(device=self.device, dtype=td).unsqueeze(1).expand(bs, self.num_samples, *tail)
                    elif isinstance(v, np.ndarray):
                        vb = np.repeat(vb[:, None, ...], self.num_samples, axis=1)
                    expanded[k] = vb
                for _ in range(self.cem_init_steps):
                    cand = torch.randn(
                        bs, self.num_samples, H, adim, generator=self.torch_gen, device=self.device, dtype=self.dtype
                    )
                    cand = cand * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                    cand[:, 0] = batch_mean
                    costs = cast(EnvironmentCost, self.model).get_cost(expanded, cand)
                    _, top = torch.topk(costs, k=self.topk, dim=1, largest=False)
                    rows = torch.arange(bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                    elites = cand[rows, top]
                    batch_mean, batch_var = elites.mean(dim=1), elites.std(dim=1)
                mean[start_idx:end_idx] = batch_mean
        return mean.float()

    # ------------------------------------------------------------------ lip
    def _graphed_for(self, wm: Any, z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor) -> Any:
        """Lazily build the CUDA-graphed refinement and stage this decision.

        Kept out of ``__init__`` so solver construction never touches CUDA
        graphs; the first solve pays a one-time capture per batch size.
        """
        if self._graphed_refinement is None:
            from .graphed import GraphedRefinement

            self._graphed_refinement = GraphedRefinement(
                wm,
                self.lip_value,
                self.temporal_objective,
                self.horizon,
                self.action_dim,
                z_hist.shape[-1],
                self.device,
                warmup_iters=self.graph_warmup_iters,
            )
        self._graphed_refinement.bind(z_hist, a_hist, z_goal)
        return self._graphed_refinement

    def _proposal_lip(self, info_dict: dict[str, Any], n_envs: int) -> torch.Tensor:
        encode_start = time.time()
        wm = cast(EncoderWorldModel, self._base())
        with torch.no_grad():
            hist_key = "pixels_hist" if "pixels_hist" in info_dict else "pixels"
            px = info_dict[hist_key].to(self.device, dtype=self.dtype)
            enc_in = {"pixels": px}
            if getattr(wm, "wants_proprio", False):
                pro = info_dict.get("proprio")
                if pro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'proprio'")
                pro = torch.as_tensor(np.asarray(pro), dtype=torch.float32, device=self.device)
                enc_in["proprio"] = pro.reshape(px.shape[0], px.shape[1], -1)
            enc = wm.encode(enc_in)
            z_hist = enc["emb"][:, -3:].float()
            real_frames = int(z_hist.shape[1])
            if z_hist.shape[1] < 3:  # pad short history (episode start, or planning.history_len < 3)
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            act_regime = "real" if self.use_action_history and "action_hist" in info_dict else "zeros"
            if getattr(self, "_history_logged", None) != (hist_key, real_frames, act_regime):
                # logged whenever the regime changes: the first plan of an episode has no
                # past (the policy pads the frame stack and omits action_hist), later
                # replans carry real lagged frames and, if opted in, real actions
                logger.info(f"LIP history: {real_frames} frame(s) from '{hist_key}', action history {act_regime}")
                self._history_logged = (hist_key, real_frames, act_regime)
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
        # planning. Only the refinement is graph-captured, so the two are timed
        # separately.
        self._last_encode_seconds = time.time() - encode_start
        B = n_envs
        zh_r, zg_r = z_hist, zg
        z0_r = zh_r[:, -1]
        a_hist = torch.zeros(B, 2, self.action_dim, device=self.device)
        if self.use_action_history and "action_hist" in info_dict:
            # rlp.core.policy buffers the solver's OUTPUT actions before it inverse-transforms
            # them for the env, so `action_hist` (B, 2, a_dim) is already in the normalised
            # space the actor trained in (dataset action mean/std), oldest block first,
            # time-major within a block, zero-padded (= the mean action) at episode start.
            # Take it as-is: no stats to apply, and none exist on non-DINO checkpoints.
            a_hist = torch.as_tensor(info_dict["action_hist"], device=self.device).reshape(B, 2, -1).to(a_hist.dtype)
        self._ground_agent0 = self._ground_u_prev = None
        if self.grounding is not None:
            # the agent's real position (proprio = agent xy + velocity, raw px) anchors the
            # commanded path; the step before the plan comes from the executed history
            # when the policy publishes it, else the mean action (zero), as in training
            proprio = info_dict.get("proprio")
            if proprio is None:
                raise KeyError("grounding term needs 'proprio' (agent position) in the observation")
            pro = torch.as_tensor(np.asarray(proprio) if not torch.is_tensor(proprio) else proprio).to(self.device)
            pro = pro.float()
            if pro.dim() == 3:
                pro = pro[:, -1]
            self._ground_agent0 = pro[:, :2]
            u_prev = torch.zeros(B, self.grounding.act_dim, device=self.device)
            if self.use_action_history and "action_hist" in info_dict:
                last = a_hist[:, -1].float().reshape(B, -1, self.grounding.act_dim)[:, -1]
                u_prev = (last * self.grounding.astd + self.grounding.amu) * ACTION_SCALE
            self._ground_u_prev = u_prev
        A = torch.zeros(B, self.horizon, self.action_dim, device=self.device)
        if self.init_mode == "value":
            A = self._value_init(wm, z_hist, a_hist, zg)
        elif self.init_mode == "cem":
            A = self._cem_init(info_dict, B).clamp(-self.actor.action_limit, self.actor.action_limit)
        A_init = A.detach().clone()
        e_iters: list[torch.Tensor] = []  # E(A_k) before the k-th update, k = 0..K-1
        lat_iters: list[torch.Tensor] = []  # imagined terminal latent distance of A_k
        graphed_ref = self._graphed_for(wm, zh_r, a_hist, zg_r) if self.graphed else None
        buf: list[torch.Tensor] = []
        for iteration in range(self.lip_iterations):
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
                        f"Graphed verification iteration={iteration} "
                        f"gradient={float((gA - gA_ref).abs().max()):.3e} "
                        f"score={float((E_graphed - score_ref.detach()).abs().max()):.3e} "
                        f"trajectory={float((traj_f - traj_ref.detach()).abs().max()):.3e}"
                    )
            else:
                with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                    A_in = A.detach().requires_grad_(True)
                    traj = rollout_traj(wm, zh_r, a_hist, A_in)
                    score = self._score(traj, zg_r, zh_r, A_in)
                    (gA,) = torch.autograd.grad(score.sum(), A_in)
                traj_f = traj.detach()
            with torch.no_grad():
                E = E_graphed if graphed_ref is not None else score.detach()
                if self.probe_directory is not None:
                    e_iters.append(E.detach().float().clone())
                    lat_iters.append((traj_f[:, -1].float() - zg_r).norm(dim=-1).detach())
                A = self.actor(A, gA, E)
                buf.append(A.clone())
        with torch.no_grad():
            if not buf:  # iters_override=0: emit the initial plan untouched
                buf = [A.detach().clone()]
            cands = torch.stack(buf if self.lip_select == "buffer" else buf[-1:], dim=1)
            C = cands.shape[1]
            cf = cands.reshape(B * C, self.horizon, self.action_dim)
            zh_c = zh_r.repeat_interleave(C, dim=0)
            ah_c = a_hist.repeat_interleave(C, dim=0)
            zg_c = zg_r.repeat_interleave(C, dim=0)
            if graphed_ref is not None and C == 1:
                # selection unroll through the same captured graph (shape matches)
                Ef = graphed_ref.score(cf).view(B, C)
            else:
                final_trajectory = rollout_traj(wm, zh_c, ah_c, cf)
                Ef = self._score(final_trajectory, zg_c, zh_c, cf).view(B, C)
            best = Ef.argmin(dim=1)
            A = cands.view(B, C, self.horizon, self.action_dim)[torch.arange(B, device=self.device), best]
        if self.probe_directory is not None:
            global _LIP_PROBE_N
            with torch.no_grad():
                z_traj = rollout_traj(wm, z_hist, a_hist[:B], A)  # (B,H,D) imagined path
                z_imag = z_traj[:, -1]
                e_imag = self._score(z_traj, zg, z_hist, A)
                # candidate population under both objectives (probe-only rollout;
                # Ef is the critic energy the deployed argmin actually used)
                traj_c = rollout_traj(wm, zh_c, ah_c, cf)
                lat_pop = (traj_c[:, -1] - zg_c).norm(dim=-1).view(B, C)
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
                    "cands": cands.view(B, C, self.horizon, self.action_dim).detach().cpu(),
                    "A_init": A_init.cpu(),
                    "E_iters": torch.stack(e_iters).cpu() if e_iters else torch.zeros(0),
                    "lat_iters": torch.stack(lat_iters).cpu() if lat_iters else torch.zeros(0),
                    "init_mode": self.init_mode,
                    "iters": int(self.lip_iterations),
                },
                self.probe_directory / f"probe_{_LIP_PROBE_N:04d}.pt",
            )
            _LIP_PROBE_N += 1
        return A.detach().to(self.dtype)

    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action
        start_time = time.time()
        outputs: dict[str, Any] = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))

        mean = self._proposal_lip(info_dict, total_envs)
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
