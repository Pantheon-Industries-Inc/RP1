"""Hierarchical CEM: macro-action search over a high-level WM, scored by value.

The configuration this exists to test has never been deployed. In the 2026-07
HWM campaign the high-level CEM scored candidates with raw terminal latent MSE
(`solver/hlip.py::_cem_search`), and the only learned high-level value in the
stack was an ENV-STEP critic trained on the dense fs1 cache — the actor lived in
macro space but nothing else did. Here the high level searches macro-actions
through :class:`rlp.core.world_model.hwm.HWM` and scores them with a
MACRO-LEVEL quasimetric (``scripts/hierarchy/train_hl_value.py``), whose
distances are in macro steps.

Why it matters at h200: cube episodes are 201 primitive steps, so at stride 25 a
200-step goal is a distance-8 query — present in-episode in every episode. The
env-step critic could only reach 200 by chaining n-step-50 backups, which
saturate, and RESULTS_hwm.md F7 read the resulting floor (every family <= 2.0)
as a data-support boundary. This solver is the instrument for separating "the
data cannot support h200" from "the value was in the wrong units".

Both levels use the softmax-refit update (CEMSolver's rule). The low level plans
primitive action blocks toward the first predicted waypoint; ``subgoal_index``
selects a later waypoint if wanted.
"""

from __future__ import annotations

import time
from typing import Any, cast

import torch

from rlp.core.rollout import rollout_traj
from rlp.core.solver.cem import CEMSolver
from rlp.core.solver.lip import unwrap_encoder
from rlp.core.value.io import load_metric
from rlp.core.world_model.hwm import load_hwm
from rlp.logging import logger

__all__ = ["HierarchicalCEMSolver"]


class HierarchicalCEMSolver(CEMSolver):
    """CEM at both levels; the high level is scored by a macro-level value.

    Args:
        hwm_path: HWM checkpoint (``scripts/hierarchy/train_hwm.py``).
        hl_value_path: macro-level metric directory; required for
            ``hl_cost="value"``.
        hl_cost: ``"value"`` (macro quasimetric) or ``"latent"`` (terminal MSE,
            i.e. the behaviour the old hierarchical CEM actually had).
        hl_horizon: macro steps per high-level plan. With stride 25, 8 spans the
            200 primitive steps an h200 cube task needs.
        subgoal_index: which predicted waypoint becomes the low level's target.
    """

    def __init__(
        self,
        *args: Any,
        hwm_path: str = "",
        bank_path: str = "",
        hl_dynamics: str = "f2",
        oracle_subgoal: bool = False,
        cache_path: str = "",
        hl_opt: str = "cem",
        hl_lr: float = 0.1,
        hl_adam_steps: int = 30,
        hl_value_path: str = "",
        hl_cost: str = "value",
        hl_horizon: int = 8,
        hl_samples: int = 512,
        hl_iters: int = 8,
        hl_lam: float = 0.01,
        hl_topk: int | None = None,
        hl_amax: float = 3.0,
        ll_samples: int | None = None,
        ll_iters: int | None = None,
        ll_lam: float = 0.01,
        ll_topk: int | None = None,
        ll_amax: float = 3.5,
        ll_clamp: bool = False,
        subgoal_index: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if hl_cost not in ("value", "latent"):
            raise ValueError(f"hl_cost must be 'value' or 'latent', got {hl_cost!r}")
        self.hl_cost = hl_cost
        self.hl_horizon = int(hl_horizon)
        self.hl_samples, self.hl_iters = int(hl_samples), int(hl_iters)
        self.hl_lam, self.hl_amax = float(hl_lam), float(hl_amax)
        self.hl_topk = int(hl_topk) if hl_topk else max(2, round(0.1 * self.hl_samples))
        self.ll_samples = int(ll_samples) if ll_samples else None
        self.ll_iters = int(ll_iters) if ll_iters else None
        self.ll_lam, self.ll_amax = float(ll_lam), float(ll_amax)
        self.ll_clamp = bool(ll_clamp)
        self.ll_topk = int(ll_topk) if ll_topk else max(2, round(0.1 * (ll_samples or 300)))
        self.subgoal_index = int(subgoal_index)

        if hl_dynamics not in ("f2", "compose"):
            raise ValueError(f"hl_dynamics must be 'f2' or 'compose', got {hl_dynamics!r}")
        self.hl_dynamics = hl_dynamics
        # Oracle-subgoal diagnostic (Hi-LeWM's test, at our horizon): replace the
        # high level with the TRUE future latent from the dataset, 1 macro ahead.
        # If the low level then succeeds, subgoal GENERATION is the bottleneck;
        # if it still fails, the problem is execution/compounding over 8 hops and
        # no generator could have helped.
        self.oracle_subgoal = bool(oracle_subgoal)
        self._oracle_cache = None
        self._oracle_rows: dict[int, Any] = {}
        self._episodes: Any = None
        self._starts: Any = None
        self._decision = 0
        self._prev_subgoal: torch.Tensor | None = None
        if self.oracle_subgoal:
            if not cache_path:
                raise ValueError("oracle_subgoal requires cache_path (the fs1 LatentCache)")
            from rlp.data import LatentCache

            cache = LatentCache.load(cache_path)
            self._oracle_cache = cache.z.to(self.device).float()
            self._oracle_rows = {int(e): r for e, r in cache.episodes().items()}
            logger.info(f"oracle subgoals from {cache_path} ({len(self._oracle_rows)} episodes)")
        if hl_opt not in ("cem", "adam"):
            raise ValueError(f"hl_opt must be 'cem' or 'adam', got {hl_opt!r}")
        self.hl_opt = hl_opt
        # defaults from the 2026-08-16 Adam sweep: (lr 0.1, 30 steps) was
        # near-optimal on 3 envs x 2 bases and lr >= 0.3 was harmful
        self.hl_lr, self.hl_adam_steps = float(hl_lr), int(hl_adam_steps)
        self.hwm = None
        self.bank = None
        if hl_dynamics == "f2":
            if not hwm_path:
                raise ValueError("hl_dynamics='f2' requires hwm_path")
            self.hwm, blob = load_hwm(hwm_path, device=self.device)
            self.macro_dim = self.hwm.macro_dim
            self.hl_stride = int(blob["cfg"]["stride"])
        else:
            # Compose the LOW-LEVEL WM over REAL action chunks instead of
            # predicting with F2. FIG6 measured F2 as worse than this at every
            # horizon (l1 0.42 vs 0.37 at 200 env steps) with the floor set by
            # the 125->8-D macro bottleneck, so the bottleneck is removed rather
            # than retrained.
            if not bank_path:
                raise ValueError("hl_dynamics='compose' requires bank_path")
            blob = torch.load(bank_path, map_location=self.device, weights_only=False)
            self.bank = blob["chunks"].to(self.device).float()
            self.hl_stride = int(blob["stride"])
            self.macro_dim = 0
        self.hl_value = None
        if hl_value_path:
            self.hl_value = load_metric(hl_value_path, device=self.device)
            self.hl_value.eval()
        if self.hl_cost == "value" and self.hl_value is None:
            raise ValueError("hl_cost='value' requires hl_value_path")
        logger.info(
            f"HierarchicalCEM: stride {self.hl_stride}, macro_dim {self.macro_dim}, "
            f"hl_horizon {self.hl_horizon} (= {self.hl_stride * self.hl_horizon} env steps), "
            f"hl_cost {self.hl_cost}"
        )

    # ------------------------------------------------------------------ utils
    def _refit(
        self,
        mean: torch.Tensor,
        amax: float,
        lam: float,
        iters: int,
        n_samples: int,
        cost_fn: Any,
        topk: int = 0,
        clamp: bool = True,
    ) -> torch.Tensor:
        """CEM refit. ``topk > 0`` uses the hard-elite rule swm's CEMSolver uses.

        The softmax branch (``topk <= 0``) is LIPSolver's rule and is kept only
        for ablation. It is NOT a drop-in here: LIP softmaxes a learned value of
        scale ~1 and seeds the mean from its actor, whereas both levels of this
        solver start from zeros and score raw quantities (a 192-d summed squared
        latent distance at the low level, macro-steps at the high level). At
        ``lam`` 0.01 the weights go one-hot on the first iteration, which makes
        the weighted variance collapse to the 0.05 floor and freezes the search
        around a single random draw -- measured 21.3 final cost against 6.7 for
        top-k at equal iterations on a surrogate of the low-level objective.
        Hard top-k is also the exact rule the flat CEM baseline runs, so using
        it keeps the optimizer out of the hierarchy comparison.
        """
        var = torch.ones_like(mean)
        b, horizon, adim = mean.shape
        k = min(int(topk), n_samples) if topk > 0 else 0
        with torch.no_grad():
            for _ in range(iters):
                cand = torch.randn(b, n_samples, horizon, adim, device=mean.device) * var.unsqueeze(
                    1
                ) + mean.unsqueeze(1)
                cand[:, 0] = mean  # keep the incumbent
                if clamp:
                    cand = cand.clamp(-amax, amax)
                cost = cost_fn(cand.reshape(b * n_samples, horizon, adim)).view(b, n_samples)
                if k > 0:
                    idx = torch.topk(cost, k, dim=1, largest=False).indices
                    rows = torch.arange(b, device=mean.device).unsqueeze(1).expand(-1, k)
                    elite = cand[rows, idx]
                    mean, var = elite.mean(dim=1), elite.std(dim=1).clamp(min=0.05)
                else:
                    w = torch.softmax(-cost / lam, dim=1)
                    mean = (w[..., None, None] * cand).sum(dim=1)
                    spread = (cand - mean.unsqueeze(1)) ** 2
                    var = ((w[..., None, None] * spread).sum(dim=1)).sqrt().clamp(min=0.05)
        return mean

    def _adam_macros(self, z0: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Optimise macro-actions by Adam through F2 and the value.

        The macro path is fully differentiable, and CEM covers a
        ``hl_horizon x macro_dim`` space (64-256 dims) with only ``hl_samples``
        draws. Gradient descent is the same move LIP makes at the low level.
        Clamping is applied inside the graph so the optimiser sees the bound.
        """
        macros = torch.zeros(
            z0.shape[0], self.hl_horizon, self.macro_dim, device=self.device, requires_grad=True
        )
        opt = torch.optim.Adam([macros], lr=self.hl_lr)
        with torch.enable_grad():
            for _ in range(self.hl_adam_steps):
                traj = self.hwm.rollout_from(z0, macros.clamp(-self.hl_amax, self.hl_amax))
                terminal = traj[:, -1]
                cost = (
                    self.hl_value(terminal, z_goal)
                    if self.hl_cost == "value"
                    else (terminal - z_goal).pow(2).sum(-1)
                ).sum()
                opt.zero_grad(set_to_none=True)
                cost.backward()
                opt.step()
        return macros.detach().clamp(-self.hl_amax, self.hl_amax)

    def set_task_context(self, episodes: Any, start_steps: Any) -> None:
        """Called by the eval driver with the per-env (episode, start) task list.

        Only the oracle arm needs it; everything else ignores it.
        """
        self._episodes = list(episodes)
        self._starts = list(start_steps)
        self._decision = 0
        self._prev_subgoal = None
        logger.info(f"task context set for {len(self._episodes)} envs (oracle={self.oracle_subgoal})")

    def _oracle_waypoint(self, b: int) -> torch.Tensor:
        """True latent one macro ahead of this decision, per env."""
        if self._episodes is None:
            raise RuntimeError("oracle_subgoal needs set_task_context from the eval driver")
        step_ahead = (self._decision + 1) * self.hl_stride
        rows = []
        for i in range(b):
            ep = int(self._episodes[i])
            ep_rows = self._oracle_rows[ep]
            idx = min(int(self._starts[i]) + step_ahead, len(ep_rows) - 1)
            rows.append(int(ep_rows[idx]))
        return self._oracle_cache[torch.as_tensor(rows, dtype=torch.long, device=self.device)]

    def _encode(self, info_dict: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z_hist (B,3,D), z_goal (B,D))`` from the policy's info dict."""
        # The eval driver hands the solver a COST STACK (MetricCost wrapping a
        # LatentGoalCost wrapping the WM), not the world model; peel it the same
        # way LIPSolver does or `.encode` is missing.
        base = unwrap_encoder(self.model)
        if getattr(base, "wants_proprio", False):
            raise NotImplementedError("proprio-variant WMs are not wired into the hierarchical solver")
        with torch.no_grad():
            hist_key = "pixels_hist" if "pixels_hist" in info_dict else "pixels"
            px = info_dict[hist_key].to(self.device, dtype=self.dtype)
            z_hist = base.encode({"pixels": px})["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:  # episode start: pad the frame stack
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            z_goal = base.encode({"pixels": gx})["emb"][:, -1].float()
        return z_hist, z_goal

    # ------------------------------------------------------------------ solve
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict[str, Any]:
        del init_action
        start = time.time()
        z_hist, z_goal = self._encode(info_dict)
        b = z_hist.shape[0]
        z0 = z_hist[:, -1]

        if self.oracle_subgoal:
            subgoal = self._oracle_waypoint(b)
            if self.hl_value is not None:
                # `residual` is the diagnostic that matters: the distance still
                # left to the waypoint the low level was aiming at during the
                # macro that just finished. Perfect tracking -> 0; a residual
                # that stays near the 1-macro reading means the low level did
                # not move the state at all.
                msg = (
                    f"ORACLE d{self._decision}: V(z0,goal) {self.hl_value(z0, z_goal).mean():.2f} "
                    f"V(z0,next-true-subgoal) {self.hl_value(z0, subgoal).mean():.2f}"
                )
                if self._prev_subgoal is not None and self._prev_subgoal.shape[0] == b:
                    resid = self.hl_value(z0, self._prev_subgoal).mean()
                    l2 = (z0 - self._prev_subgoal).pow(2).sum(-1).sqrt().mean()
                    msg += f" | residual-to-last-target {resid:.2f} macro-steps (latent l2 {l2:.2f})"
                logger.info(msg + " macro-steps")
            self._prev_subgoal = subgoal
            self._decision += 1
            return self._solve_low(z_hist, subgoal, b, start)

        if self.hl_dynamics == "compose":
            return self._solve_compose(z_hist, z_goal, start)

        # ---------------- high level: macro search through F2
        hl_n = self.hl_samples
        zg_hl = z_goal.repeat_interleave(hl_n, dim=0)
        z0_hl = z0.repeat_interleave(hl_n, dim=0)

        def hl_cost(macros: torch.Tensor) -> torch.Tensor:
            traj = self.hwm.rollout_from(z0_hl, macros)
            terminal = traj[:, -1]
            if self.hl_cost == "value":
                return cast(torch.Tensor, self.hl_value(terminal, zg_hl))
            return cast(torch.Tensor, (terminal - zg_hl).pow(2).sum(-1))

        macro_mean = torch.zeros(b, self.hl_horizon, self.macro_dim, device=self.device)
        if self.hl_opt == "adam":
            macro_mean = self._adam_macros(z0, z_goal)
        else:
            macro_mean = self._refit(
                macro_mean, self.hl_amax, self.hl_lam, self.hl_iters, hl_n, hl_cost, topk=self.hl_topk
            )

        with torch.no_grad():
            waypoints = self.hwm.rollout_from(z0, macro_mean)
            idx = min(self.subgoal_index, waypoints.shape[1] - 1)
            subgoal = waypoints[:, idx]
            if self.hl_value is not None:
                d_goal = self.hl_value(z0, z_goal).mean().item()
                d_sub = self.hl_value(z0, subgoal).mean().item()
                logger.info(
                    f"HL: V(z0,goal) {d_goal:.2f} macro-steps, V(z0,subgoal) {d_sub:.2f}"
                )

        return self._solve_low(z_hist, subgoal, b, start)

    def _solve_low(
        self, z_hist: torch.Tensor, subgoal: torch.Tensor, b: int, start: float
    ) -> dict[str, Any]:
        """Low level: CEM over primitive action blocks toward ``subgoal``."""
        ll_n = self.ll_samples or self.num_samples
        ll_iters = self.ll_iters or self.n_steps
        base = unwrap_encoder(self.model)
        zh_ll = z_hist.repeat_interleave(ll_n, dim=0)
        ah_ll = torch.zeros(b * ll_n, 2, self.action_dim, device=self.device)
        sub_ll = subgoal.repeat_interleave(ll_n, dim=0)

        def ll_cost(plan: torch.Tensor) -> torch.Tensor:
            traj = rollout_traj(base, zh_ll, ah_ll, plan)
            return cast(torch.Tensor, (traj[:, -1] - sub_ll).pow(2).sum(-1))

        plan_mean = torch.zeros(b, self.horizon, self.action_dim, device=self.device)
        plan_mean = self._refit(
            plan_mean,
            self.ll_amax,
            self.ll_lam,
            ll_iters,
            ll_n,
            ll_cost,
            topk=self.ll_topk,
            clamp=self.ll_clamp,  # flat CEM does not clamp; match it
        )

        logger.info(f"Hierarchical solve completed in {time.time() - start:.4f} seconds")
        return {"actions": plan_mean, "costs": [], "mean": [], "var": []}

    # ------------------------------------------------------- compose variant
    def _chunk_to_blocks(self, chunks: torch.Tensor) -> torch.Tensor:
        """``(B, S, K, act_dim)`` chunks -> ``(B, S*K/block, block*act_dim)`` plan."""
        b, s_len, k, adim = chunks.shape
        per_block = self.action_dim // adim  # primitive steps per action block
        return chunks.reshape(b, s_len * k // per_block, per_block * adim)

    def _solve_compose(
        self, z_hist: torch.Tensor, z_goal: torch.Tensor, start: float
    ) -> dict[str, Any]:
        """High level = sequences of REAL action chunks rolled through the LL WM.

        Search is over bank indices (no free macro vector, so nothing can drift
        off the action manifold), scoring the 200-step terminal latent with the
        macro-level value. The winning sequence's FIRST chunk is executed, so the
        actions deployed are ones the data actually contains.
        """
        import numpy as np

        base = unwrap_encoder(self.model)
        b = z_hist.shape[0]
        n = self.hl_samples
        n_bank = self.bank.shape[0]
        slots = self.hl_horizon

        zh = z_hist.repeat_interleave(n, dim=0)
        ah = torch.zeros(b * n, 2, self.action_dim, device=self.device)
        zg = z_goal.repeat_interleave(n, dim=0)

        # categorical CEM over bank indices, one distribution per slot
        logits = torch.zeros(b, slots, n_bank, device=self.device)
        best_idx = None
        with torch.no_grad():
            for _ in range(self.hl_iters):
                probs = torch.softmax(logits, dim=-1)
                idx = torch.multinomial(probs.reshape(-1, n_bank), n, replacement=True)
                idx = idx.reshape(b, slots, n).permute(0, 2, 1)  # (b, n, slots)
                chunks = self.bank[idx.reshape(-1, slots)]  # (b*n, slots, K, adim)
                plan = self._chunk_to_blocks(chunks)
                traj = rollout_traj(base, zh, ah, plan)
                terminal = traj[:, -1]
                cost = (
                    self.hl_value(terminal, zg)
                    if self.hl_cost == "value"
                    else (terminal - zg).pow(2).sum(-1)
                ).view(b, n)
                # refit: raise the logits of the elite sequences
                k_elite = max(1, n // 10)
                elite = cost.topk(k_elite, dim=1, largest=False).indices  # (b, k)
                new_logits = torch.full_like(logits, -1e4)
                for i in range(b):
                    sel = idx[i, elite[i]]  # (k, slots)
                    for s_i in range(slots):
                        vals, counts = torch.unique(sel[:, s_i], return_counts=True)
                        new_logits[i, s_i, vals] = counts.float().log()
                logits = new_logits
                best_idx = idx[torch.arange(b), elite[:, 0]]  # (b, slots)

            chunks = self.bank[best_idx]  # (b, slots, K, adim)
            if self.hl_value is not None:
                plan_full = self._chunk_to_blocks(chunks)
                traj = rollout_traj(base, z_hist, torch.zeros(b, 2, self.action_dim, device=self.device), plan_full)
                logger.info(
                    f"HL(compose): V(z0,goal) {self.hl_value(z_hist[:, -1], z_goal).mean():.2f} -> "
                    f"V(terminal,goal) {self.hl_value(traj[:, -1], z_goal).mean():.2f} macro-steps"
                )
            # execute the first chunk: real actions, no low-level re-planning
            first = self._chunk_to_blocks(chunks[:, :1])
            plan_mean = first[:, : self.horizon]
            if plan_mean.shape[1] < self.horizon:  # pad short chunks
                pad = plan_mean[:, -1:].expand(-1, self.horizon - plan_mean.shape[1], -1)
                plan_mean = torch.cat([plan_mean, pad], dim=1)

        logger.info(f"Hierarchical(compose) solve in {time.time() - start:.4f} seconds")
        return {"actions": plan_mean, "costs": [], "mean": [], "var": []}
