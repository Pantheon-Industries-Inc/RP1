"""PlannerNet family — the learned update rule f_theta at the core of LIP.

A planning *algorithm* learned purely by optimizing against a frozen value
function on top of a frozen world model. No behavior cloning, no policy
gradient: the planner is a learned update rule

    A_{k+1} = clip( A_k + gate * dA ),   [dA, gate] = f_theta(A_k, grad_A V, V_k, z0, zg)

applied for K iterations from A_0 = 0, trained by backpropagating
``V(z_T(A_K), z_g)`` through the differentiable frozen WM (pathwise, no RL).

At plan time one refinement pass costs ~2 WM rollouts per iteration
(~16-20 rollout-equivalents total at K=8) versus ~9000 for the repo's CEM
defaults (300 samples x 30 iterations) — a ~450x compute reduction at equal
or better success rate (see docs/lip/README_lip.md for benchmarks).

Three architectures share the residual + clip contract: ``PlannerNet`` (the
flat-MLP form, v1/v2/v4), ``PlannerNetRec`` (v4 with a GRU-carried hidden
state across iterations), and ``PlannerNetV3`` (a per-plan-block transformer
with localized, iteration-conditioned updates). Selection between them is a
checkpoint property (``kind``), handled by :class:`rlp.core.solver.LIPSolver`.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "PlannerNet",
    "PlannerNetV3",
    "PlannerNetRec",
]


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

    def __init__(
        self,
        z_dim: int,
        horizon: int,
        a_dim: int,
        hidden: int,
        amax: float,
        feed: str,
        use_zg: bool,
        use_gate: bool,
        use_z0: bool,
        use_grad: bool,
        head_scale: float,
    ) -> None:
        super().__init__()
        self.h, self.a, self.amax, self.feed = horizon, a_dim, amax, feed
        self.use_zg, self.use_gate, self.use_z0 = use_zg, use_gate, use_z0
        self.use_grad = use_grad
        extra = {"none": 0, "end": z_dim, "traj": horizon * z_dim}[feed]
        # use_zg=False ('vonly-ized' MLP): the raw goal embedding is dropped —
        # the goal reaches the actor only through the teacher's signals
        # (E and grad_A V). use_z0=False: raw current state dropped too; with
        # both off the actor sees only [A, grad V, E] — the purest
        # learned-optimizer form. use_gate=False: pure residual, A + dA.
        # LIPv4 (kind='lip4') = all three off; the trainers default to it.
        # use_grad=False: drop grad_A V — ABLATION of the value-gradient input;
        # the actor must plan from [A, E, z0, zg] with no first-order signal
        # (only informative on a full-input actor that still has z0/zg).
        in_dim = horizon * a_dim + int(use_grad) * horizon * a_dim + 1 + (int(use_z0) + int(use_zg)) * z_dim + extra
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, horizon * a_dim + (1 if use_gate else 0)),
        )
        if head_scale != 1.0:
            # small-init for the no-gate form: refinement starts near-identity
            # with live gradients (covers the gate's damping role at init;
            # train-time only — loading a state_dict overwrites this)
            with torch.no_grad():
                self.net[-1].weight.mul_(head_scale)
                self.net[-1].bias.mul_(head_scale)

    def forward(
        self,
        A: torch.Tensor,
        gradA: torch.Tensor,
        E: torch.Tensor,
        z0: torch.Tensor,
        zg: torch.Tensor,
        ztraj: torch.Tensor | None = None,
        k: int = 0,
        vtraj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del k, vtraj
        B = A.shape[0]
        parts = [A.reshape(B, -1)]
        if self.use_grad:
            parts.append(gradA.reshape(B, -1))
        parts.append(E.reshape(B, 1))
        if self.use_z0:
            parts.append(z0)
        if self.use_zg:
            parts.append(zg)
        if self.feed == "end":
            if ztraj is None:
                raise ValueError("feed='end' requires ztraj")
            parts.append(ztraj[:, -1])
        elif self.feed == "traj":
            if ztraj is None:
                raise ValueError("feed='traj' requires ztraj")
            parts.append(ztraj.reshape(B, -1))
        out = torch.as_tensor(self.net(torch.cat(parts, dim=-1)))
        if self.use_gate:
            dA = out[:, :-1].view(B, self.h, self.a)
            gate = torch.sigmoid(out[:, -1:]).unsqueeze(-1)
            return (A + gate * dA).clamp(-self.amax, self.amax)
        return (A + out.view(B, self.h, self.a)).clamp(-self.amax, self.amax)


class PlannerNetRec(nn.Module):
    """Recurrent LIPv4 (kind='lip4r'): the learned refinement rule carries a
    hidden state across the K weight-tied iterations, instead of the memoryless
    v4 map (plan, grad, value) -> plan.

        x_k          = enc([A_k, grad_A V, V])          input embedding
        s_{k+1}      = GRUCell(x_k, s_k)                 recurrent update
        dA_k         = head(s_{k+1})                     residual plan update
        A_{k+1}      = clip_±amax(A_k + dA_k)

    Same input set as v4 min0 (only [A, grad V, E] carry task info — no raw
    z0/zg unless the flags are on), same pure-residual + clip contract. The
    only addition is the memory s: the optimizer can accumulate momentum /
    curvature-like statistics of its own refinement trajectory, which the
    memoryless MLP (whose only cross-iteration carrier is A itself) cannot.

    s0_mode='zero': s_0 = 0 (blank memory).
    s0_mode='z0':   s_0 = tanh(W z0) — memory seeded from the current latent,
                    so the very first update is state-conditioned. (When
                    s_dim == z_dim the projection can learn the identity, the
                    literal s_0 = z_0.)
    """

    def __init__(
        self,
        z_dim: int,
        horizon: int,
        a_dim: int,
        hidden: int,
        s_dim: int,
        amax: float,
        use_z0: bool,
        use_zg: bool,
        use_grad: bool,
        s0_mode: str,
        head_scale: float,
    ) -> None:
        super().__init__()
        self.h, self.a, self.amax = horizon, a_dim, amax
        self.use_z0, self.use_zg, self.use_grad = use_z0, use_zg, use_grad
        self.s_dim, self.s0_mode = s_dim, s0_mode
        # kept for checkpoint-flag parity with PlannerNet (no gate here)
        self.use_gate = False
        in_dim = horizon * a_dim + int(use_grad) * horizon * a_dim + 1 + (int(use_z0) + int(use_zg)) * z_dim
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.cell = nn.GRUCell(hidden, s_dim)
        self.head = nn.Linear(s_dim, horizon * a_dim)
        self.s0_proj = nn.Linear(z_dim, s_dim) if s0_mode == "z0" else None
        if head_scale != 1.0:
            # near-identity start (dA ~ 0) with live gradients — matches the v4
            # practice of covering the dropped gate's damping role at init,
            # important here because the residual is unrolled through K GRU steps
            with torch.no_grad():
                self.head.weight.mul_(head_scale)
                self.head.bias.mul_(head_scale)

    def init_state(self, B: int, z0: torch.Tensor) -> torch.Tensor:
        if self.s0_mode == "z0":
            if self.s0_proj is None:
                raise RuntimeError("z0 state initialization requires a projection")
            return torch.tanh(self.s0_proj(z0))
        return z0.new_zeros(B, self.s_dim)

    def forward(
        self,
        A: torch.Tensor,
        gradA: torch.Tensor,
        E: torch.Tensor,
        z0: torch.Tensor,
        zg: torch.Tensor,
        s: torch.Tensor,
        ztraj: torch.Tensor | None = None,
        k: int = 0,
        vtraj: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ztraj, k, vtraj
        B = A.shape[0]
        parts = [A.reshape(B, -1)]
        if self.use_grad:
            parts.append(gradA.reshape(B, -1))
        parts.append(E.reshape(B, 1))
        if self.use_z0:
            parts.append(z0)
        if self.use_zg:
            parts.append(zg)
        x = self.enc(torch.cat(parts, dim=-1))
        s_new = self.cell(x, s)
        dA = self.head(s_new).view(B, self.h, self.a)
        A_new = (A + dA).clamp(-self.amax, self.amax)
        return A_new, s_new


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

    def __init__(
        self,
        z_dim: int,
        horizon: int,
        a_dim: int,
        width: int,
        layers: int,
        amax: float,
        n_iters: int,
        zproj: int,
        goal_mode: str,
        zero_init: bool,
        gd_init: float,
        feat_norm: bool,
        vscale: float,
        iter_mode: str,
        head_mode: str,
        p0: float,
        head_scale: float,
        cond_mode: str,
        use_gate: bool,
        pre_ln: bool,
    ) -> None:
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
            in_dim += 1  # k/(K-1) appended per token, no embedding
        self.cond = nn.Linear(2, width) if cond_mode == "global" else None
        self.inp = nn.Linear(in_dim, width)
        self.pos = nn.Parameter(torch.zeros(1, horizon, width))
        self.it_emb = nn.Embedding(n_iters, width) if iter_mode == "emb" else None
        layer = nn.TransformerEncoderLayer(
            width,
            nhead=4,
            dim_feedforward=2 * width,
            dropout=0.0,
            batch_first=True,
            norm_first=pre_ln,
        )
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
        head_out = 2 * a_dim if head_mode == "precond" else a_dim + (1 if use_gate else 0)
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

    def forward(
        self,
        A: torch.Tensor,
        gradA: torch.Tensor,
        E: torch.Tensor,
        z0: torch.Tensor,
        zg: torch.Tensor,
        ztraj: torch.Tensor | None = None,
        k: int = 0,
        vtraj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del z0
        if ztraj is None:
            raise ValueError("PlannerNetV3 requires an imagined trajectory")
        B = A.shape[0]
        ghat = None
        if self.eta is not None or self.feat_norm or self.head_mode == "precond":
            ghat = gradA / (gradA.pow(2).mean(dim=(1, 2), keepdim=True).sqrt() + 1e-8)
        kc = min(int(k), self.n_iters - 1)
        if self.goal_mode == "vonly":
            g_in = ghat if self.feat_norm else gradA
            if g_in is None or vtraj is None:
                raise ValueError("vonly mode requires gradient and value trajectories")
            parts = [A, g_in, ztraj]
            if self.cond_mode == "token":
                v_in = vtraj / self.vscale if self.feat_norm else vtraj
                parts.append(v_in.unsqueeze(-1))
        elif self.goal_mode == "sep":
            # goal fed raw (projected) + per-step teacher value V(z_t, zg)
            zt = self.zp(ztraj)
            zrel = self.gp(zg).unsqueeze(1).expand(-1, self.h, -1)
            if vtraj is None:
                raise ValueError("sep mode requires a value trajectory")
            parts = [A, gradA, zt, zrel, vtraj.unsqueeze(-1)]
        else:  # 'diff' (v3.0)
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
        out = torch.as_tensor(self.head(self.enc(x)))
        if self.head_mode == "precond":
            if ghat is None:
                raise RuntimeError("preconditioned mode requires normalized gradients")
            P = torch.nn.functional.softplus(out[..., : self.a])
            r = out[..., self.a :]
            A_new = A - P * ghat + r
        elif not self.use_gate:
            A_new = A + out  # pure residual: magnitude owned by the head
        else:
            dA = out[..., :-1]
            gate = torch.sigmoid(out[..., -1:])
            A_new = A + gate * dA
            if self.eta is not None:
                if ghat is None:
                    raise RuntimeError("gradient-descent initialization requires normalized gradients")
                A_new = A_new - self.eta[min(int(k), self.n_iters - 1)] * ghat
        return torch.as_tensor(A_new.clamp(-self.amax, self.amax))
