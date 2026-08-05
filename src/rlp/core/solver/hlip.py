"""Hierarchical solver (HWM x LIP) for pooled-latent LeWM world models.

Two-level receding-horizon planning per HWM (arXiv 2604.03208):
  1. HIGH level plans a sequence of latent macro-actions through the
     high-level WM (F2, stride K env steps per macro step) toward the goal
     latent; the FIRST predicted waypoint of the optimized plan becomes the
     subgoal (their ``pred[1]``).
  2. LOW level plans primitive action blocks toward that subgoal with the
     standard LeWM machinery (identical conventions to LIPSolver:
     3-frame latent history, zero action history, rollout_traj).

Each level is independently 'lip' (learned iterative planner: K refinement
steps through a PlannerNet with value gradient + energy, argmin-V selection)
or 'cem' (sampling with softmax refits, matching CEMSolver's update rule;
cost = terminal latent MSE — the LeWM-native criterion; NO learned value, so
hl=cem, ll=cem is the HWM-faithful baseline).

``hl_cutoff > 0`` ports their final-approach trick (flat planner for the last
steps near the goal): when the LL value estimates the goal within ``hl_cutoff``
env steps, the high level is bypassed and the low level receives the true
goal. With hl_cutoff <= 0 the hierarchy is always active.
"""

import time
from typing import Any, cast

import torch
from stable_worldmodel.solver.cem import CEMSolver

from rlp.logging import logger

from ..planner import PlannerNet
from ..rollout import rollout_terminal, rollout_traj
from ..world_model.protocols import LatentWorldModel
from .lip import EncoderWorldModel, LIPCheckpoint, ValueFunction


class HLIPSolver(CEMSolver):
    def __init__(
        self,
        *args: Any,
        ll_actor_path: str = "",
        hl_actor_path: str = "",
        hwm_path: str = "",
        hl: str = "lip",
        ll: str = "lip",
        hl_cutoff: float = -1.0,
        subgoal_index: int = 0,
        hl_horizon: int = 0,
        hl_samples: int = 1024,
        hl_iters: int = 8,
        hl_lam: float = 0.01,
        hl_amax: float = 3.0,
        hl_cost: str = "terminal",
        ll_samples: int = 300,
        ll_iters: int = 30,
        ll_lam: float = 0.01,
        ll_amax: float = 3.5,
        subgoal_project: bool = False,
        project_pool: str = "",
        subgoal_select: str = "index",
        reach_target: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        from rlp.core.value import load_metric

        from ..world_model.hwm import load_hwm

        assert hl in ("lip", "cem") and ll in ("lip", "cem")
        self.hl, self.ll = hl, ll
        self.hl_cutoff = float(hl_cutoff)
        self.subgoal_index = int(subgoal_index)
        assert subgoal_select in ("index", "reach")
        self.subgoal_select = subgoal_select
        self.reach_target = float(reach_target)
        self.hl_samples, self.hl_iters = int(hl_samples), int(hl_iters)
        self.hl_lam, self.hl_amax = float(hl_lam), float(hl_amax)
        self.hl_cost = hl_cost
        self.ll_samples, self.ll_iters = int(ll_samples), int(ll_iters)
        self.ll_lam, self.ll_amax = float(ll_lam), float(ll_amax)

        # ---------------- low level
        self.ll_actor = self.ll_value = None
        self._ll_horizon = None
        if ll == "lip" or hl_cutoff > 0:
            assert ll_actor_path, "ll=lip / hl_cutoff>0 need ll_actor_path"
            raw_checkpoint = torch.load(ll_actor_path, map_location=self.device, weights_only=False)
            if not isinstance(raw_checkpoint, dict):
                raise TypeError("low-level LIP checkpoint must contain a mapping")
            ck = cast(LIPCheckpoint, raw_checkpoint)
            assert ck.get("kind") in ("lip", "lip2", "lip4"), ck.get("kind")
            v4 = ck.get("kind") == "lip4"
            self.ll_actor = PlannerNet(
                ck["z_dim"],
                horizon=ck["horizon"],
                a_dim=ck.get("a_dim", 25),
                amax=ck.get("amax", 2.5),
                feed=ck.get("feed", "none") if ck.get("kind") == "lip2" else "none",
                use_zg=ck.get("use_zg", not v4),
                use_gate=ck.get("use_gate", not v4),
                use_z0=ck.get("use_z0", not v4),
                use_grad=ck.get("use_grad", True),
            ).to(self.device)
            self.ll_actor.load_state_dict(ck["sd"])
            self.ll_actor.eval()
            self._ll_horizon = ck["horizon"]
            self._ll_iters_lip = ck["iters"]
            ll_value_module = load_metric(ck["value"], device=self.device)
            ll_value_module.eval()
            self.ll_value = cast(ValueFunction, ll_value_module)

        # ---------------- high level
        self.hl_actor = self.hl_value = None
        self._hwm_path = hwm_path
        if hl == "lip":
            assert hl_actor_path, "hl=lip needs hl_actor_path"
            raw_checkpoint = torch.load(hl_actor_path, map_location=self.device, weights_only=False)
            if not isinstance(raw_checkpoint, dict):
                raise TypeError("high-level LIP checkpoint must contain a mapping")
            ck = cast(LIPCheckpoint, raw_checkpoint)
            assert ck.get("kind") == "lip4", f"hl actor must be lip4, got {ck.get('kind')}"
            self.hl_actor = PlannerNet(
                ck["z_dim"],
                horizon=ck["horizon"],
                a_dim=ck["a_dim"],
                amax=ck.get("amax", 3.0),
                use_zg=ck.get("use_zg", False),
                use_gate=ck.get("use_gate", False),
                use_z0=ck.get("use_z0", False),
                use_grad=ck.get("use_grad", True),
            ).to(self.device)
            self.hl_actor.load_state_dict(ck["sd"])
            self.hl_actor.eval()
            self._hl_horizon = ck["horizon"]
            self._hl_iters_lip = ck["iters"]
            hl_value_module = load_metric(ck["value"], device=self.device)
            hl_value_module.eval()
            self.hl_value = cast(ValueFunction, hl_value_module)
            self._hwm_path = self._hwm_path or ck.get("hwm", "")
        else:
            assert hl_horizon > 0, "hl=cem needs hl_horizon"
            self._hl_horizon = int(hl_horizon)
        assert self._hwm_path, "need hwm_path (or a hl actor ckpt storing it)"
        self.hwm, blob = load_hwm(self._hwm_path, device=self.device)
        self.macro_dim = self.hwm.macro_dim
        self.hl_stride = blob["cfg"]["stride"]

        # optional subgoal manifold projection: snap the imagined waypoint to
        # its nearest real latent (subsampled cache pool) before the LL sees it
        self._pool = None
        if subgoal_project:
            assert project_pool, "subgoal_project needs project_pool (LatentCache path)"
            from rlp.data import LatentCache

            self._pool = LatentCache.load(project_pool).z[::5].float().to(self.device)

    def _project(self, q: torch.Tensor, chunk: int = 200000) -> torch.Tensor:
        if self._pool is None:
            raise RuntimeError("subgoal projection pool is not configured")
        best_d = torch.full((len(q),), float("inf"), device=self.device)
        best_i = torch.zeros(len(q), dtype=torch.long, device=self.device)
        for i0 in range(0, self._pool.shape[0], chunk):
            d = torch.cdist(q, self._pool[i0 : i0 + chunk])
            m, ix = d.min(1)
            upd = m < best_d
            best_d[upd] = m[upd]
            best_i[upd] = ix[upd] + i0
        return self._pool[best_i]

    # plan length handed to the policy = LL plan length
    @property
    def horizon(self) -> int:
        return int(self._ll_horizon if self._ll_horizon else self._config.horizon)

    def _base(self) -> torch.nn.Module:
        candidate = getattr(self.model, "base", self.model)
        if not isinstance(candidate, torch.nn.Module):
            raise TypeError("solver model base must be torch.nn.Module")
        return candidate

    # ------------------------------------------------------------ shared
    def _encode(self, info_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        wm = cast(EncoderWorldModel, self._base())
        with torch.no_grad():
            px = info_dict["pixels"].to(self.device, dtype=self.dtype)
            enc = wm.encode({"pixels": px})
            z_hist = enc["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            zg = wm.encode({"pixels": gx})["emb"][:, -1].float()
        return z_hist, zg

    def _lip_refine(
        self,
        wm: LatentWorldModel,
        actor: PlannerNet,
        value: ValueFunction,
        iters: int,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        zg: torch.Tensor,
    ) -> torch.Tensor:
        """K learned refinement iterations (v4 contract), argmin-V last plans."""
        B = z_hist.shape[0]
        H, adim = actor.h, actor.a
        z0 = z_hist[:, -1]
        A = torch.zeros(B, H, adim, device=self.device)
        for k_it in range(iters):
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(wm, z_hist, a_hist, A_in)
                (gA,) = torch.autograd.grad(value(traj[:, -1], zg).sum(), A_in)
            with torch.no_grad():
                traj_f = rollout_traj(wm, z_hist, a_hist, A)
                E = value(traj_f[:, -1], zg)
                A = actor(A, gA, E, z0, zg, traj_f, k=k_it)
        return A.detach()

    def _cem_refine(
        self,
        wm: LatentWorldModel,
        iters: int,
        n_samples: int,
        lam: float,
        amax: float,
        horizon: int,
        adim: int,
        z_hist: torch.Tensor,
        a_hist: torch.Tensor,
        zg: torch.Tensor,
        cost_mode: str = "terminal",
    ) -> torch.Tensor:
        """Softmax-refit sampling (CEMSolver's update rule) over plans;
        cost = latent MSE to zg (terminal by default)."""
        B = z_hist.shape[0]
        mean = torch.zeros(B, horizon, adim, device=self.device)
        var = torch.ones_like(mean)
        with torch.no_grad():
            for _ in range(iters):
                cand = torch.randn(
                    B,
                    n_samples,
                    horizon,
                    adim,
                    device=self.device,
                    generator=self.torch_gen,
                ) * var.unsqueeze(1) + mean.unsqueeze(1)
                cand[:, 0] = mean
                cand = cand.clamp(-amax, amax)
                cf = cand.reshape(B * n_samples, horizon, adim)
                zh = z_hist.repeat_interleave(n_samples, dim=0)
                ah = a_hist.repeat_interleave(n_samples, dim=0)
                traj = rollout_traj(wm, zh, ah, cf)
                zg_r = zg.repeat_interleave(n_samples, dim=0)
                if cost_mode == "terminal":
                    cost = (traj[:, -1] - zg_r).pow(2).sum(-1)
                elif cost_mode == "last3":
                    n = min(3, horizon)
                    cost = (traj[:, -n:] - zg_r.unsqueeze(1)).pow(2).sum(-1).mean(1)
                else:  # all
                    cost = (traj - zg_r.unsqueeze(1)).pow(2).sum(-1).mean(1)
                cost = cost.view(B, n_samples)
                w = torch.softmax(-cost / lam, dim=1)
                mean = (w[..., None, None] * cand).sum(dim=1)
                spread = (cand - mean.unsqueeze(1)) ** 2
                var = ((w[..., None, None] * spread).sum(dim=1)).sqrt().clamp(min=0.05)
        return mean

    # ------------------------------------------------------------ solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action
        t0 = time.time()
        wm = cast(EncoderWorldModel, self._base())
        z_hist, zg = self._encode(info_dict)
        B = z_hist.shape[0]
        z0 = z_hist[:, -1]

        # ---- high level -> subgoal
        subgoal = zg.clone()
        need = torch.ones(B, dtype=torch.bool, device=self.device)
        if self.hl_cutoff > 0 and self.ll_value is not None:
            with torch.no_grad():
                need = self.ll_value(z0, zg) > self.hl_cutoff
        used_hl = int(need.sum())
        if used_hl:
            zh_hl = z0[need].unsqueeze(1).expand(-1, 3, -1)
            ah_hl = torch.zeros(used_hl, 2, self.macro_dim, device=self.device)
            if self.hl == "lip":
                if self.hl_actor is None or self.hl_value is None:
                    raise RuntimeError("high-level LIP components are unavailable")
                L = self._lip_refine(
                    self.hwm,
                    self.hl_actor,
                    self.hl_value,
                    self._hl_iters_lip,
                    zh_hl,
                    ah_hl,
                    zg[need],
                )
            else:
                L = self._cem_refine(
                    self.hwm,
                    self.hl_iters,
                    self.hl_samples,
                    self.hl_lam,
                    self.hl_amax,
                    self._hl_horizon,
                    self.macro_dim,
                    zh_hl,
                    ah_hl,
                    zg[need],
                    cost_mode=self.hl_cost,
                )
            with torch.no_grad():
                wp = rollout_traj(self.hwm, zh_hl, ah_hl, L)
            if self.subgoal_select == "reach":
                # pick the waypoint whose distance-from-here best matches the
                # LL's competence radius, among waypoints that make progress
                assert self.ll_value is not None, "subgoal_select=reach needs ll value"
                with torch.no_grad():
                    n, Hh, D = wp.shape
                    wf = wp.reshape(n * Hh, D).float()
                    z0n = z0[need].repeat_interleave(Hh, 0)
                    zgn = zg[need].repeat_interleave(Hh, 0)
                    d0w = self.ll_value(z0n, wf).view(n, Hh)
                    dwg = self.ll_value(wf, zgn).view(n, Hh)
                    d0g_n = self.ll_value(z0[need], zg[need]).unsqueeze(1)
                    score = (d0w - self.reach_target).abs() + 1e6 * (dwg >= d0g_n - 1.0).float()
                    best = score.argmin(dim=1)
                    sg = wp[torch.arange(n, device=self.device), best].float()
            else:
                sg = wp[:, min(self.subgoal_index, wp.shape[1] - 1)].float()
            if self._pool is not None:
                sg = self._project(sg)
            subgoal[need] = sg

        # ---- low level -> primitive plan toward subgoal
        a_hist = torch.zeros(B, 2, self.action_dim, device=self.device)
        if self.ll == "lip":
            if self.ll_actor is None or self.ll_value is None:
                raise RuntimeError("low-level LIP components are unavailable")
            A = self._lip_refine(
                wm,
                self.ll_actor,
                self.ll_value,
                self._ll_iters_lip,
                z_hist,
                a_hist,
                subgoal,
            )
        else:
            A = self._cem_refine(
                wm,
                self.ll_iters,
                self.ll_samples,
                self.ll_lam,
                self.ll_amax,
                self.horizon,
                self.action_dim,
                z_hist,
                a_hist,
                subgoal,
                cost_mode="terminal",
            )
        with torch.no_grad():
            zT = rollout_terminal(wm, z_hist, a_hist, A)
            final = (zT - subgoal).pow(2).sum(-1) if self.ll_value is None else self.ll_value(zT, subgoal)

        logger.info(f"HLIP solve completed in {time.time() - t0:.3f}s; high_level_used={used_hl}/{B}")
        return {
            "actions": A.detach().to(self.dtype).cpu(),
            "costs": final.detach().cpu().tolist(),
            "mean": [A.detach().cpu()],
            "var": [torch.zeros_like(A).cpu()],
        }


__all__ = ["HLIPSolver"]
