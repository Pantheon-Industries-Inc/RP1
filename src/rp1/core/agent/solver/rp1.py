"""The rp1 solver: a trained planner network refines a plan along the value gradient."""

import time
from typing import Any, cast

import numpy as np
import torch
from stable_worldmodel.solver.cem import CEMSolver

from rp1.core.agent.planner import PlannerNet
from rp1.core.agent.solver.base import EncoderWorldModel, EnvironmentCost, PlannerCheckpoint, unwrap_encoder
from rp1.core.agent.value.grounding import ACTION_SCALE, GroundingPenalty
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, window_pair, windowed_trajectory_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_terminal, rollout_traj
from rp1.utils.logging import logger

__all__ = ["RP1Solver"]


class RP1Solver(CEMSolver):
    """Plan with a trained planner checkpoint, optionally followed by MPPI.

    With ``n_steps=0`` the solver is deterministic: K learned refinement
    iterations from one initial plan. ``n_steps > 0`` then runs MPPI
    (temperature ``lam``) around the refined plan. The horizon comes from the
    checkpoint.

    ``update_rule="gradient"`` swaps the learned update for a plain gradient step
    of size ``gd_lr`` on the same energy, inside the same loop, which tells an
    exploitable energy landscape apart from a rule that learned to exploit it.
    ``band_limit`` projects every update onto the first temporal DCT modes of the
    plan. A checkpoint trained with physics grounding adds the same penalty to
    the energy; ``ground_weight`` rescales it, and 0 deploys without it.
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
        update_rule: str,
        gd_lr: float,
        band_limit: int | None,
        use_action_history: bool,
        ground_weight: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if update_rule not in {"learned", "gradient"}:
            raise ValueError(f"unsupported update rule: {update_rule}")
        self.update_rule = update_rule
        self.gd_lr = float(gd_lr)
        self.band_limit = band_limit or None
        self._band_projection: torch.Tensor | None = None
        # the policy publishes the executed action blocks in the planner's normalized
        # space; the planner trained on zeros for replayed histories, so it is opt-in
        self.use_action_history = use_action_history
        self.lam = lam
        self.rp1_select = rp1_select
        self.init_mode = init_mode
        self.init_samples = int(init_samples)
        self.init_scale = float(init_scale)
        self.cem_init_steps = int(cem_init_steps)
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
            logger.info(f"Refinement iterations overridden: {self.rp1_iterations} -> {self.iters_override}")
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
            logger.info(f"Window value over {self.vframes} frames")
            if self.graphed:
                raise ValueError("graphed refinement does not support windowed values")
        self.grounding: GroundingPenalty | None = None
        exported = payload.get("grounding")
        if exported:
            grounding = GroundingPenalty.from_export(exported).to(self.device)
            if ground_weight is not None:
                grounding.weight = float(ground_weight)
            if grounding.weight == 0:
                logger.info("Grounding: trained with the term, deployed without it")
            elif self.graphed:
                raise ValueError("graphed refinement does not support the grounding term")
            else:
                self.grounding = grounding
                logger.info(
                    f"Grounding: weight {grounding.weight:g} margin {grounding.margin:.1f}px "
                    f"deadzone {grounding.deadzone:.1f}px tau {grounding.tau:g}"
                )
        elif ground_weight:
            raise ValueError("ground_weight is set, but the planner checkpoint carries no grounding term")
        self._grounding_anchors: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def horizon(self) -> int:
        return self._actor_horizon

    def _base(self) -> torch.nn.Module:
        return unwrap_encoder(self.model)

    def _score(
        self, trajectory: torch.Tensor, goal: torch.Tensor, history: torch.Tensor, plan: torch.Tensor
    ) -> torch.Tensor:
        """Energy of an imagined trajectory: its value, over windows of ``vframes`` frames when the
        value is windowed, plus the grounding penalty of the ``plan`` that produced it."""
        if self.vframes > 1:
            energy = windowed_trajectory_value(
                self.rp1_value, trajectory, goal, history, self.vframes, self.temporal_objective
            )
        else:
            energy = trajectory_value(self.rp1_value, trajectory, goal, history[:, -1], self.temporal_objective)
        if self.grounding is not None:
            if self._grounding_anchors is None:
                raise RuntimeError("grounding term scored before the agent position was read")
            copies = trajectory.shape[0] // self._grounding_anchors[0].shape[0]  # candidates per environment
            agent, command = (anchor.repeat_interleave(copies, dim=0) for anchor in self._grounding_anchors)
            penalty = self.grounding(history[:, -1].float(), trajectory.float(), plan.float(), agent, command)
            energy = energy + penalty.to(energy.dtype)
        return energy

    def _band_limited(self, previous: torch.Tensor, refined: torch.Tensor) -> torch.Tensor:
        """``refined`` with the update from ``previous`` kept to the first ``band_limit`` DCT-II modes."""
        if self.band_limit is None:
            return refined
        if self._band_projection is None:
            t = torch.arange(self.horizon, dtype=torch.float32, device=self.device)
            k = torch.arange(self.band_limit, dtype=torch.float32, device=self.device)
            basis = torch.cos(torch.pi * (t[:, None] + 0.5) * k[None, :] / self.horizon)
            basis = basis / basis.norm(dim=0, keepdim=True)
            self._band_projection = basis @ basis.T
        update = torch.einsum("ij,bja->bia", self._band_projection, refined - previous)
        return (previous + update).clamp(-self.actor.action_limit, self.actor.action_limit)

    def _grounding_anchors_of(self, info_dict: dict[str, Any], a_hist: torch.Tensor) -> None:
        """The agent position (px) and the preceding command (px) the grounding term starts from."""
        if self.grounding is None:
            self._grounding_anchors = None
            return
        proprio = info_dict.get("proprio")
        if proprio is None:
            raise KeyError("the grounding term needs 'proprio' (the agent position) in the observation")
        position = torch.as_tensor(np.asarray(proprio) if not torch.is_tensor(proprio) else proprio)
        position = position.to(self.device).float()
        if position.dim() == 3:
            position = position[:, -1]
        batch = a_hist.shape[0]
        # the last executed primitive action when the history is used, else the mean action
        command = torch.zeros(batch, self.grounding.act_dim, device=self.device)
        if self.use_action_history and "action_hist" in info_dict:
            last = a_hist[:, -1].float().reshape(batch, -1, self.grounding.act_dim)[:, -1]
            command = (last * self.grounding.astd + self.grounding.amu) * ACTION_SCALE
        self._grounding_anchors = (position[:, :2], command)

    def _value_init(
        self,
        wm: LatentWorldModel,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        zg: torch.Tensor,
    ) -> torch.Tensor:
        """The lowest-value plan among zero, iid-Gaussian and time-tiled Gaussian candidates.

        The tiled candidates repeat one action block across the horizon: sustained
        motions that refinement from zero cannot reach where the value gradient is
        flat at the zero plan.
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
        """The mean of a CEM run on the planning cost, ``(B, H, action_dim)``."""
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

    def _graphed_for(self, wm: Any, z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor) -> Any:
        """The CUDA-graphed refinement, built on the first solve, bound to this decision."""
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
            # lagged frames one action block apart when the policy keeps a history
            px = info_dict.get("pixels_hist", info_dict["pixels"]).to(self.device, dtype=self.dtype)
            enc_in = {"pixels": px}
            if getattr(wm, "wants_proprio", False):
                pro = info_dict.get("proprio")
                if pro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'proprio'")
                pro = torch.as_tensor(np.asarray(pro), dtype=torch.float32, device=self.device)
                enc_in["proprio"] = pro.reshape(px.shape[0], px.shape[1], -1)
            enc = wm.encode(enc_in)
            z_hist = enc["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:  # pad a short history: episode start, or planning.history_len < 3
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            genc_in = {"pixels": gx}
            if getattr(wm, "wants_proprio", False):
                gpro = info_dict.get("goal_state")
                if gpro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'goal_state'")
                gpro = torch.as_tensor(np.asarray(gpro), dtype=torch.float32, device=self.device)
                gpro = gpro.reshape(gx.shape[0], -1)[:, -2:]  # last frame's (x, y)
                genc_in["proprio"] = gpro.unsqueeze(1).expand(-1, gx.shape[1], -1)
            zg = wm.encode(genc_in)["emb"][:, -1].float()

        # encoding and planning are timed separately; only planning is graph-captured
        self._last_encode_seconds = time.time() - encode_start
        B = n_envs
        zh_r, zg_r = z_hist, zg
        a_hist = torch.zeros(B, 2, self.action_dim, device=self.device)
        if self.use_action_history and "action_hist" in info_dict:
            # oldest block first, zero (the mean action) where the episode has no past yet
            a_hist = torch.as_tensor(info_dict["action_hist"], device=self.device).reshape(B, 2, -1).to(a_hist.dtype)
        self._grounding_anchors_of(info_dict, a_hist)
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
                if self.update_rule == "gradient":
                    refined = (A - self.gd_lr * gA).clamp(-self.actor.action_limit, self.actor.action_limit)
                else:
                    refined = self.actor(A, gA, E)
                A = self._band_limited(A, refined)
                buf.append(A.clone())
        with torch.no_grad():
            if not buf:  # zero iterations: the initial plan
                buf = [A.detach().clone()]
            cands = torch.stack(buf if self.rp1_select == "buffer" else buf[-1:], dim=1)
            C = cands.shape[1]
            cf = cands.reshape(B * C, self.horizon, self.action_dim)
            zh_c = zh_r.repeat_interleave(C, dim=0)
            ah_c = a_hist.repeat_interleave(C, dim=0)
            zg_c = zg_r.repeat_interleave(C, dim=0)
            if graphed_ref is not None and C == 1:
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
            f"RP1 solve completed in {total_seconds:.4f} seconds "
            f"(encode {encode_seconds:.4f}, plan {total_seconds - encode_seconds:.4f})"
        )
        return outputs
