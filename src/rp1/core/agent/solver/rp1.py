import time
from typing import Any, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rp1.core.agent.planner import PlannerNet
from rp1.core.agent.solver.base import EncoderWorldModel, EnvironmentCost, PlannerCheckpoint, unwrap_encoder
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, window_pair, windowed_trajectory_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_terminal, rollout_traj
from rp1.utils.logging import logger

__all__ = ["RP1Solver"]


class RP1Solver(CEMSolver):
    """Plan with a trained rp1 checkpoint; optionally refine with MPPI.

    With ``n_steps=0`` (the benchmarked configuration) the solver is fully
    deterministic: one pass of K learned refinement iterations from a single
    initialization, then execute the plan. ``n_steps > 0`` additionally runs MPPI
    (softmax-weighted, temperature ``lam``) seeded from the rp1 plan.

    The plan length (horizon) always comes from the actor checkpoint.
    """

    def __init__(
        self,
        *args: Any,
        checkpoint: PlannerCheckpoint,
        lam: float,
        rp1_select: str,
        init_mode: str,
        init_samples: int,
        init_scale: float,
        cem_init_steps: int,
        iters_override: int | None,
        graphed: bool,
        graph_warmup_iters: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.rp1_select = rp1_select
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
        # CUDA-graphed refinement (GraphedRefinement), captured on the first solve
        self.graphed = graphed
        self.graph_warmup_iters = graph_warmup_iters
        self._graphed_refinement: Any = None

        payload = checkpoint.payload
        horizon = int(payload["horizon"])
        self.actor = PlannerNet(
            horizon=horizon,
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            action_limit=float(payload["action_limit"]),
            head_scale=float(payload["head_scale"]),
        ).to(self.device)
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("planner payload state_dict must contain a mapping")
        self.actor.load_state_dict(state_dict)
        self.actor.eval()
        self._actor_horizon = horizon
        self.rp1_iterations = int(payload["iterations"])
        if self.iters_override is not None:
            logger.info(f"rp1 iterations overridden at deploy: {self.rp1_iterations} -> {self.iters_override}")
            self.rp1_iterations = self.iters_override
        self.temporal_objective = str(payload["temporal_objective"])
        if self.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
            raise ValueError(f"unsupported temporal objective: {self.temporal_objective}")
        value_module = checkpoint.value.to(self.device)
        value_module.eval()
        self.rp1_value = cast(ValueFunction, value_module)
        # window values score a stack of the last `vframes` imagined frames
        # against the goal frame repeated as often
        self.vframes = int(payload["window_frames"])
        if self.vframes > 1:
            logger.info(f"rp1 window value: {self.vframes} frames")
            if self.graphed:
                raise ValueError("graphed rp1 inference does not support windowed values")

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
                self.rp1_value, trajectory, goal, history, self.vframes, self.temporal_objective
            )
        return trajectory_value(self.rp1_value, trajectory, goal, history[:, -1], self.temporal_objective)

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
                E = self.rp1_value(*window_pair(trajectory.float(), zg_c, self.vframes)).view(B, C)
            else:
                term = rollout_terminal(
                    wm,
                    z_hist.repeat_interleave(C, dim=0),
                    a_hist.repeat_interleave(C, dim=0),
                    cand.reshape(B * C, H, adim),
                )
                E = self.rp1_value(term.float(), zg_c).view(B, C)
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

    # ------------------------------------------------------------------ rp1
    def _graphed_for(self, wm: Any, z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor) -> Any:
        """Lazily build the CUDA-graphed refinement and stage this decision.

        Kept out of ``__init__`` so solver construction never touches CUDA
        graphs; the first solve pays a one-time capture per batch size.
        """
        if self._graphed_refinement is None:
            from rp1.core.agent.solver.graphed import GraphedRefinement

            self._graphed_refinement = GraphedRefinement(
                wm,
                self.rp1_value,
                self.temporal_objective,
                self.horizon,
                self.action_dim,
                z_hist.shape[-1],
                self.device,
                warmup_iters=self.graph_warmup_iters,
            )
        self._graphed_refinement.bind(z_hist, a_hist, z_goal)
        return self._graphed_refinement

    def _proposal_rp1(self, info_dict: dict[str, Any], n_envs: int) -> torch.Tensor:
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
        graphed_ref = self._graphed_for(wm, zh_r, a_hist, zg_r) if self.graphed else None
        buf: list[torch.Tensor] = []
        for _ in range(self.rp1_iterations):
            if graphed_ref is not None:
                # one forward-graph and one backward-graph replay
                E_graphed, _, gA = graphed_ref.step(A)
            else:
                with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                    A_in = A.detach().requires_grad_(True)
                    traj = rollout_traj(wm, zh_r, a_hist, A_in)
                    score = self._score(traj, zg_r, zh_r, A_in)
                    (gA,) = torch.autograd.grad(score.sum(), A_in)
            with torch.no_grad():
                E = E_graphed if graphed_ref is not None else score.detach()
                A = self.actor(A, gA, E)
                buf.append(A.clone())
        with torch.no_grad():
            if not buf:  # iters_override=0: emit the initial plan untouched
                buf = [A.detach().clone()]
            cands = torch.stack(buf if self.rp1_select == "buffer" else buf[-1:], dim=1)
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
        return A.detach().to(self.dtype)

    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action
        start_time = time.time()
        outputs: dict[str, Any] = {"costs": [], "mean": [], "var": []}
        total_envs = len(next(iter(info_dict.values())))

        mean = self._proposal_rp1(info_dict, total_envs)
        var = self.var_scale * torch.ones_like(mean)

        # optional MPPI refinement around the rp1 plan (n_steps=0 -> pure rp1)
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
            f"rp1 solve completed in {total_seconds:.4f} seconds "
            f"(encode {encode_seconds:.4f}, plan {total_seconds - encode_seconds:.4f})"
        )
        return outputs
