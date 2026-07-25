"""LIP — Learned Iterative Planner.

A planning *algorithm* learned purely by optimizing against a frozen value
function on top of a frozen world model. No behavior cloning, no policy
gradient: the planner is a learned update rule

    A_{k+1} = clip( A_k + gate * dA ),   [dA, gate] = f_theta(A_k, grad_A V, V_k, z0, zg)

applied for K iterations from A_0 = 0, trained by backpropagating
``V(z_T(A_K), z_g)`` through the differentiable frozen WM (pathwise, no RL).

At plan time one refinement pass costs ~2 WM rollouts per iteration
(~16-20 rollout-equivalents total at K=8) versus ~9000 for the repo's CEM
defaults (300 samples x 30 iterations) — a ~450x compute reduction at equal
or better success rate (see scripts/plan/README_lip.md for benchmarks).

Two world-model families are supported, selected by the ``kind`` field of the
actor checkpoint (written by the training scripts):

- ``kind='lip'``      pooled-latent WMs (e.g. LeWM): state = (B, 3, D) pooled
  history, rollout via :func:`rollout_terminal`.
- ``kind='lip4'``     LIPv4 (current default): same rollout as ``lip`` but the
  minimal-input, gate-free update rule — f_theta sees only [A, grad_A V, E];
  state and goal reach the planner exclusively through the value function.
- ``kind='lip_dino'`` patch-token WMs (PreJEPA/DINO): state = (B, 3, P, D)
  patch tokens, action embeddings injected per block, rollout via
  :func:`rollout_terminal_dino`. Plans are trained/rolled in normalized action
  units and denormalized (stats stored in the checkpoint) before execution.

Trainers: ``scripts/plan/train_lip.py`` and ``scripts/plan/train_lip_dino.py``.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn

from stable_worldmodel.solver.cem import CEMSolver


__all__ = ["PlannerNet", "PlannerNetV3", "rollout_terminal", "rollout_traj",
           "rollout_terminal_dino", "LIPSolver"]


class PlannerNet(nn.Module):
    """The learned update rule f_theta.

    Input:  current plan A (B,H,a), value gradient dV/dA (B,H,a), scalar value
            E (B,), current latent z0 (B,D), goal latent zg (B,D) — and, with
            ``feed='end'`` (LIP-v2), the imagined terminal latent, or with
            ``feed='traj'`` all H imagined latents.
    Output: refined plan, gated residual update, clipped to +-amax.

    v2 note: ``feed='end'`` does not raise peak success but tightens the
    training-draw distribution (5/5 paired draws >= v1 across two tasks; e.g.
    K8/lr3e-4/4k-steps single-cube draws {80,82,84} vs v1's {76,78,78}).
    """

    def __init__(self, z_dim, horizon=5, a_dim=25, hidden=512, amax=2.5,
                 feed="none", use_zg=True, use_gate=True, use_z0=True,
                 head_scale=1.0):
        super().__init__()
        self.h, self.a, self.amax, self.feed = horizon, a_dim, amax, feed
        self.use_zg, self.use_gate, self.use_z0 = use_zg, use_gate, use_z0
        extra = {"none": 0, "end": z_dim, "traj": horizon * z_dim}[feed]
        # use_zg=False ('vonly-ized' MLP): the raw goal embedding is dropped —
        # the goal reaches the actor only through the teacher's signals
        # (E and grad_A V). use_z0=False: raw current state dropped too; with
        # both off the actor sees only [A, grad V, E] — the purest
        # learned-optimizer form. use_gate=False: pure residual, A + dA.
        # LIPv4 (kind='lip4') = all three off; the trainers default to it.
        in_dim = (2 * horizon * a_dim + 1
                  + (int(use_z0) + int(use_zg)) * z_dim + extra)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * a_dim + (1 if use_gate else 0)),
        )
        if head_scale != 1.0:
            # small-init for the no-gate form: refinement starts near-identity
            # with live gradients (covers the gate's damping role at init;
            # train-time only — loading a state_dict overwrites this)
            with torch.no_grad():
                self.net[-1].weight.mul_(head_scale)
                self.net[-1].bias.mul_(head_scale)

    def forward(self, A, gradA, E, z0, zg, ztraj=None, k=0, vtraj=None):
        B = A.shape[0]
        parts = [A.reshape(B, -1), gradA.reshape(B, -1), E.reshape(B, 1)]
        if self.use_z0:
            parts.append(z0)
        if self.use_zg:
            parts.append(zg)
        if self.feed == "end":
            parts.append(ztraj[:, -1])
        elif self.feed == "traj":
            parts.append(ztraj.reshape(B, -1))
        out = self.net(torch.cat(parts, dim=-1))
        if self.use_gate:
            dA = out[:, :-1].view(B, self.h, self.a)
            gate = torch.sigmoid(out[:, -1:]).unsqueeze(-1)
            return (A + gate * dA).clamp(-self.amax, self.amax)
        return (A + out.view(B, self.h, self.a)).clamp(-self.amax, self.amax)


class PlannerNetV3(nn.Module):
    """Trajectory-aligned refiner (LIP-v3): a small transformer over the H plan
    blocks instead of one flat MLP over the whole plan.

    Token t pairs action block A_t with the imagined latent reached after it:
        [A_t, grad_{A_t}V, proj(z_t), proj(z_g - z_t), E, pos_t] + iter-emb(k)
    and each token gets its OWN update dA_t and gate — refinement can localize
    (careful step at the grasp block, bold step in free space) and condition on
    the refinement iteration k (big steps early, polish late). Fixes v1's
    single-scalar gate / no-temporal-structure limits and realigns the v2 feed
    (whole-traj flattened into an MLP hurt; per-token alignment is the natural
    correspondence). Same residual + clip contract as PlannerNet.
    """

    def __init__(self, z_dim, horizon=5, a_dim=25, width=256, layers=2,
                 amax=2.5, n_iters=8, zproj=64, goal_mode="diff", zero_init=False,
                 gd_init=0.0, feat_norm=False, vscale=25.0, iter_mode="emb",
                 head_mode="gate", p0=0.05, head_scale=1.0, cond_mode="token",
                 use_gate=True, pre_ln=False):
        super().__init__()
        self.h, self.a, self.amax = horizon, a_dim, amax
        self.goal_mode = goal_mode
        self.feat_norm = feat_norm
        self.vscale = vscale
        self.iter_mode = iter_mode
        self.head_mode = head_mode
        self.cond_mode = cond_mode
        self.use_gate = use_gate
        if goal_mode == "vonly":
            # token = [A_t, grad_t, RAW z_t (, V(z_t, zg))]: no projections, no zg —
            # the goal enters ONLY through the learned quasimetric (raw latent
            # geometry toward the goal is never exposed).
            # cond_mode='global' (v3.2): value/iteration leave the token concat and
            # enter as a learned global bias cond=W_c[E/vscale, k/(K-1)] added to
            # every token — a dedicated path so the two scalars aren't drowned by
            # the 192-dim latent in the input projection. Final value only.
            in_dim = 2 * a_dim + z_dim + (1 if cond_mode == "token" else 0)
        else:
            self.zp = nn.Linear(z_dim, zproj)
            self.gp = nn.Linear(z_dim, zproj)
            in_dim = 2 * a_dim + 2 * zproj + 1
        if iter_mode == "scalar" and cond_mode == "token":
            in_dim += 1                     # k/(K-1) appended per token, no embedding
        self.cond = nn.Linear(2, width) if cond_mode == "global" else None
        self.inp = nn.Linear(in_dim, width)
        self.pos = nn.Parameter(torch.zeros(1, horizon, width))
        self.it_emb = nn.Embedding(n_iters, width) if iter_mode == "emb" else None
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2 * width,
                                           dropout=0.0, batch_first=True,
                                           norm_first=pre_ln)
        self.enc = nn.TransformerEncoder(layer, layers)
        # head modes:
        #   'gate'    (legacy): out = [dA, gate-logit];  A' = A + sigmoid(gate)*dA
        #   'precond': out = [p, r];  dA = -softplus(p) ⊙ ghat + r  — the actor IS
        #     a learned per-entry preconditioner on normalized value-gradient
        #     descent plus a residual. Bias init makes step 0 ≈ GD with step p0;
        #     small-random weights keep every gradient path live from step 0.
        # head width: precond = [p, r]; gate = [dA, gate-logit]; no-gate = [dA]
        # (no-gate: A' = A + dA — the gate is redundant in expressiveness; its
        # highway-style Jacobian damping through the K-step unroll is instead
        # provided at init time by head_scale. Round-H ablation.)
        if head_mode == "precond":
            head_out = 2 * a_dim
        else:
            head_out = a_dim + (1 if use_gate else 0)
        self.head = nn.Linear(width, head_out)
        self.n_iters = n_iters
        # GD-init (legacy bolt-on, superseded by head_mode='precond'): update
        # includes -eta_k * ghat with learnable per-iteration step sizes.
        self.eta = nn.Parameter(torch.full((n_iters,), float(gd_init))) if gd_init > 0 else None
        self.in_ln = nn.LayerNorm(width) if feat_norm else None
        if head_mode == "precond":
            softplus_inv_p0 = math.log(math.expm1(float(p0)))
            nn.init.normal_(self.head.weight, std=1e-3)
            nn.init.zeros_(self.head.bias)
            with torch.no_grad():
                self.head.bias[:a_dim] = softplus_inv_p0
        elif head_scale != 1.0:
            # small-init: near-identity start with LIVE gradients everywhere —
            # no GD prior, no extra mechanisms; the init alone fixes both the
            # compounding-noise (default) and dead-gradient (exact-zero) issues
            with torch.no_grad():
                self.head.weight.mul_(float(head_scale))
                nn.init.zeros_(self.head.bias)
        if zero_init:
            # near-identity start: dA == 0 at init (refiner = "change nothing"),
            # embeddings small — avoids compounding random updates through the
            # K-step unrolled chain at the start of training
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            if self.it_emb is not None:
                with torch.no_grad():
                    self.it_emb.weight.mul_(0.02)

    def forward(self, A, gradA, E, z0, zg, ztraj=None, k=0, vtraj=None):
        B = A.shape[0]
        ghat = None
        if self.eta is not None or self.feat_norm or self.head_mode == "precond":
            ghat = gradA / (gradA.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8)
        kc = min(int(k), self.n_iters - 1)
        if self.goal_mode == "vonly":
            g_in = ghat if self.feat_norm else gradA
            parts = [A, g_in, ztraj]
            if self.cond_mode == "token":
                v_in = vtraj / self.vscale if self.feat_norm else vtraj
                parts.append(v_in.unsqueeze(-1))
        elif self.goal_mode == "sep":
            # goal fed raw (projected) + per-step teacher value V(z_t, zg)
            zt = self.zp(ztraj)
            zrel = self.gp(zg).unsqueeze(1).expand(-1, self.h, -1)
            parts = [A, gradA, zt, zrel, vtraj.unsqueeze(-1)]
        else:                                                  # 'diff' (v3.0)
            zt = self.zp(ztraj)
            zrel = self.gp(zg.unsqueeze(1) - ztraj)
            e = E.view(B, 1, 1).expand(-1, self.h, -1)
            parts = [A, gradA, zt, zrel, e]
        if self.iter_mode == "scalar" and self.cond_mode == "token":
            parts.append(A.new_full((B, self.h, 1), kc / max(self.n_iters - 1, 1)))
        x = self.inp(torch.cat(parts, dim=-1)) + self.pos
        if self.it_emb is not None:
            ki = torch.full((B, 1), kc, dtype=torch.long, device=A.device)
            x = x + self.it_emb(ki)
        if self.cond is not None:
            kn = A.new_full((B,), kc / max(self.n_iters - 1, 1))
            x = x + self.cond(torch.stack([E / self.vscale, kn], dim=-1)).unsqueeze(1)
        if self.in_ln is not None:
            x = self.in_ln(x)
        out = self.head(self.enc(x))
        if self.head_mode == "precond":
            P = torch.nn.functional.softplus(out[..., :self.a])
            r = out[..., self.a:]
            A_new = A - P * ghat + r
        elif not self.use_gate:
            A_new = A + out                     # pure residual: magnitude owned by the head
        else:
            dA = out[..., :-1]
            gate = torch.sigmoid(out[..., -1:])
            A_new = A + gate * dA
            if self.eta is not None:
                A_new = A_new - self.eta[min(int(k), self.n_iters - 1)] * ghat
        return A_new.clamp(-self.amax, self.amax)


def rollout_terminal(wm, z_hist, a_hist, plan):
    """Pooled-latent WM: autoregressive H-block unroll; returns terminal latent.

    z_hist: (B, 3, D) latent history, a_hist: (B, 2, a) action-block history,
    plan: (B, H, a). Differentiable w.r.t. ``plan``.
    """
    return rollout_traj(wm, z_hist, a_hist, plan)[:, -1]


def rollout_traj(wm, z_hist, a_hist, plan):
    """Like :func:`rollout_terminal` but returns all H imagined latents (B, H, D)."""
    embs = list(z_hist.unbind(dim=1))
    acts = list(a_hist.unbind(dim=1))
    outs = []
    for t in range(plan.shape[1]):
        acts.append(plan[:, t])
        win_e = torch.stack(embs[-3:], dim=1)
        win_a = torch.stack(acts[-3:], dim=1)
        nxt = wm.predict(win_e, wm.action_encoder(win_a))[:, -1]
        embs.append(nxt)
        outs.append(nxt)
    return torch.stack(outs, dim=1)


def rollout_terminal_dino(wm, toks_hist, plan, pix_dim=384, act_emb_dim=10):
    """Patch-token WM (PreJEPA): rollout in token space.

    toks_hist: (B, 3, P, D) tokens = pixel patches ++ tiled proprio-emb ++
    tiled action-emb. Each step injects the plan block's action embedding into
    the last frame's action slice, predicts the next frame's tokens, appends.
    Returns the terminal frame's mean-pooled pixel embedding (B, pix_dim) —
    the space the value function was trained on. Differentiable w.r.t. ``plan``
    (raw, denormalized action units).
    """
    toks = list(toks_hist.unbind(1))
    act_enc = wm.extra_encoders["action"]
    for t in range(plan.shape[1]):
        a_emb = act_enc(plan[:, t].unsqueeze(1))[:, 0]
        last = toks[-1]
        last = torch.cat([last[..., :-act_emb_dim],
                          a_emb.unsqueeze(1).expand(-1, last.shape[1], -1)], dim=-1)
        toks[-1] = last
        win = torch.stack(toks[-3:], dim=1)
        nxt = wm.predict(win)[:, -1]
        toks.append(nxt)
    return toks[-1][..., :pix_dim].mean(dim=1)


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

    def __init__(self, *args, actor_path: str = "", lam: float = 1.0,
                 restarts: int = 1, restart_noise: float = 0.5,
                 lip_select: str = "last", robust_m: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        from stable_worldmodel.trm import load_metric

        self.lam = lam
        self.restarts = int(restarts)
        self.restart_noise = restart_noise
        self.lip_select = lip_select
        self.robust_m = int(robust_m)

        ck = torch.load(actor_path, map_location=self.device, weights_only=False)
        self.kind = ck.get("kind")
        if self.kind not in ("lip", "lip2", "lip3", "lip4", "lip_dino"):
            raise ValueError(f"LIPSolver: unsupported checkpoint kind {self.kind!r}")
        self.feed = ck.get("feed", "none") if self.kind == "lip2" else "none"
        if self.kind == "lip3":
            self.actor = PlannerNetV3(ck["z_dim"], horizon=ck["horizon"],
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
                                      pre_ln=ck.get("pre_ln", False)).to(self.device)
        else:
            # lip4 checkpoints store the flags explicitly; the flipped ck.get
            # defaults only guard hand-rolled checkpoints of each kind
            v4 = self.kind == "lip4"
            self.actor = PlannerNet(ck["z_dim"], horizon=ck["horizon"],
                                    a_dim=ck.get("a_dim", 25),
                                    amax=ck.get("amax", 2.5),
                                    feed=self.feed,
                                    use_zg=ck.get("use_zg", not v4),
                                    use_gate=ck.get("use_gate", not v4),
                                    use_z0=ck.get("use_z0", not v4)).to(self.device)
        self.actor.load_state_dict(ck["sd"])
        self.actor.eval()
        self._actor_horizon = ck["horizon"]        # plan length must match the trained net
        self.lip_iters = ck["iters"]
        self.lip_value = load_metric(ck["value"], device=self.device)
        self.lip_value.eval()
        if self.kind == "lip_dino":
            self._amu = ck["amu5"].to(self.device).float()
            self._ast = ck["ast5"].to(self.device).float()

    @property
    def horizon(self) -> int:
        ah = getattr(self, "_actor_horizon", None)
        return ah if ah else self._config.horizon

    def _base(self):
        return getattr(self.model, "base", self.model)

    # ------------------------------------------------------------------ lip
    def _proposal_lip(self, info_dict, n_envs):
        wm = self._base()
        with torch.no_grad():
            px = info_dict["pixels"].to(self.device, dtype=self.dtype)
            enc = wm.encode({"pixels": px})
            z_hist = enc["emb"][:, -3:].float()
            if z_hist.shape[1] < 3:                # pad short history at episode start
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            zg = wm.encode({"pixels": gx})["emb"][:, -1].float()

        R = max(1, self.restarts)
        B = n_envs
        zh_r = z_hist.repeat_interleave(R, dim=0)
        zg_r = zg.repeat_interleave(R, dim=0)
        z0_r = zh_r[:, -1]
        a_hist = torch.zeros(B * R, 2, self.action_dim, device=self.device)
        A = self.restart_noise * torch.randn(B * R, self.horizon, self.action_dim,
                                             device=self.device, generator=self.torch_gen)
        A[::R] = 0.0                               # one zero-init restart per env
        buf = []
        for k_it in range(self.lip_iters):
            with torch.enable_grad():
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(wm, zh_r, a_hist, A_in)
                (gA,) = torch.autograd.grad(self.lip_value(traj[:, -1], zg_r).sum(), A_in)
            with torch.no_grad():
                traj_f = rollout_traj(wm, zh_r, a_hist, A)
                E = self.lip_value(traj_f[:, -1], zg_r)
                vtraj = None
                gm = getattr(self.actor, "goal_mode", None)
                if gm == "sep" or (gm == "vonly" and
                                   getattr(self.actor, "cond", None) is None):
                    Bh, H = traj_f.shape[0], traj_f.shape[1]
                    vtraj = self.lip_value(
                        traj_f.reshape(Bh * H, -1),
                        zg_r.repeat_interleave(H, dim=0)).view(Bh, H)
                A = self.actor(A, gA, E, z0_r, zg_r, traj_f, k=k_it, vtraj=vtraj)
                buf.append(A.clone())
        with torch.no_grad():
            cands = torch.stack(buf if self.lip_select == "buffer" else buf[-1:], dim=1)
            C = cands.shape[1]
            cf = cands.reshape(B * R * C, self.horizon, self.action_dim)
            zh_c = zh_r.repeat_interleave(C, dim=0)
            ah_c = a_hist.repeat_interleave(C, dim=0)
            zg_c = zg_r.repeat_interleave(C, dim=0)
            if self.robust_m > 0:                  # value of m perturbed copies (robust argmin)
                scores = 0
                amax = self.actor.amax
                for _ in range(self.robust_m):
                    pert = (cf + 0.1 * torch.randn_like(cf)).clamp(-amax, amax)
                    scores = scores + self.lip_value(
                        rollout_terminal(wm, zh_c, ah_c, pert), zg_c)
                Ef = (scores / self.robust_m).view(B, R * C)
            else:
                Ef = self.lip_value(
                    rollout_terminal(wm, zh_c, ah_c, cf), zg_c).view(B, R * C)
            best = Ef.argmin(dim=1)
            A = cands.view(B, R * C, self.horizon, self.action_dim)[
                torch.arange(B, device=self.device), best]
        return A.detach().to(self.dtype)

    # ------------------------------------------------------------- lip_dino
    def _proposal_lip_dino(self, info_dict, n_envs):
        wm = self._base()
        px = info_dict["pixels"].to(self.device).float()[:, -3:]
        B = px.shape[0]
        pro = info_dict.get("proprio")
        pro = (pro.to(self.device).float()[:, -3:] if pro is not None
               else torch.zeros(B, px.shape[1], 41, device=self.device))
        if px.shape[1] < 3:
            k = 3 - px.shape[1]
            px = torch.cat([px[:, :1].expand(-1, k, -1, -1, -1), px], dim=1)
            pro = torch.cat([pro[:, :1].expand(-1, k, -1), pro], dim=1)
        with torch.no_grad():
            info = {"pixels": px, "proprio": pro,
                    "action": torch.zeros(B, 3, self.action_dim, device=self.device)}
            toks = wm.encode(info)["emb"].float()                    # (B,3,P,D)
            gx = info_dict["goal"].to(self.device).float()
            genc = wm.encode({"pixels": gx}, emb_keys=[], target="gemb")
            zg = genc["pixels_gemb"][:, -1].mean(dim=1).float()
            z0 = toks[:, -1, :, :384].mean(dim=1)
        A = torch.zeros(B, self.horizon, self.action_dim, device=self.device)
        for _ in range(self.lip_iters):
            with torch.enable_grad():
                A_in = A.detach().requires_grad_(True)
                zT = rollout_terminal_dino(wm, toks, A_in * self._ast + self._amu)
                (gA,) = torch.autograd.grad(self.lip_value(zT, zg).sum(), A_in)
            with torch.no_grad():
                E = self.lip_value(
                    rollout_terminal_dino(wm, toks, A * self._ast + self._amu), zg)
                A = self.actor(A, gA, E, z0, zg)
        return (A * self._ast + self._amu).to(self.dtype)            # raw action units

    # ------------------------------------------------------------------ solve
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        outputs = {"costs": [], "mean": [], "var": []}
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

            expanded = {}
            for k, v in info_dict.items():
                vb = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    td = self.dtype if vb.is_floating_point() else None
                    vb = vb.to(device=self.device, dtype=td).unsqueeze(1).expand(
                        bs, self.num_samples, *vb.shape[1:])
                elif isinstance(v, np.ndarray):
                    vb = np.repeat(vb[:, None, ...], self.num_samples, axis=1)
                expanded[k] = vb

            final_cost = [0.0] * bs
            for _ in range(self.n_steps):
                cand = torch.randn(bs, self.num_samples, self.horizon, self.action_dim,
                                   generator=self.torch_gen, device=self.device,
                                   dtype=self.dtype)
                cand = cand * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                cand[:, 0] = batch_mean
                costs = self.model.get_cost(expanded, cand)
                w = torch.softmax(-costs.float() / self.lam, dim=1)
                batch_mean = (w[..., None, None] * cand.float()).sum(dim=1).to(self.dtype)
                spread = (cand.float() - batch_mean.unsqueeze(1).float()) ** 2
                batch_var = ((w[..., None, None] * spread).sum(dim=1).sqrt()
                             .clamp(min=0.05).to(self.dtype))
                final_cost = (w * costs.float()).sum(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(f"LIP solve time: {time.time() - start_time:.4f} seconds")
        return outputs
