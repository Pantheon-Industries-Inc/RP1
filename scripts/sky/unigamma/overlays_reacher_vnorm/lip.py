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
import os
import time

import numpy as np
import torch
import torch.nn as nn

from stable_worldmodel.solver.cem import CEMSolver



def window_pair(traj, z_goal, frames, z0=None):
    """(stacked_state, tiled_goal) for an m-frame window value.

    ``traj`` is (B, H, D) imagined latents, newest last. Fewer than ``frames``
    available => repeat the oldest (train_window.py's clamp). ``frames==1``
    returns the terminal frame unchanged, so single-frame values are untouched.
    """
    if frames <= 1:
        return traj[:, -1], z_goal
    cols = []
    for k in range(frames - 1, -1, -1):
        idx = traj.shape[1] - 1 - k
        if idx >= 0:
            cols.append(traj[:, idx])
        else:
            cols.append(traj[:, 0] if traj.shape[1] else z0)
    return torch.cat(cols, dim=-1), z_goal.repeat(*([1] * (z_goal.dim() - 1)), frames)
# mech-interp hook: if LIP_PROBE_DIR is set, _proposal_lip dumps per-replan
# (z0, imagined-terminal latent, its value, goal latent) so an open-loop eval
# yields imagined-vs-actually-reached (consecutive replans) for the A/B probe.
_LIP_PROBE_N = 0

# I3 readout (see LIPSolver._V). 'deadline' scores the chunk that lands on the
# graded step; 'min' scores the best still-reachable chunk. Both are inert
# without plan_config.deadline. Env var, not config, so the sampling path in
# eval_wm.py reads the same switch.
_I3_MODE = os.environ.get("I3_MODE", "deadline").lower()


__all__ = ["PlannerNet", "PlannerNetV3", "PlannerNetRec", "rollout_terminal",
           "rollout_traj", "rollout_terminal_dino", "LIPSolver"]


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
                 use_grad=True, head_scale=1.0, vnorm="none"):
        super().__init__()
        self.h, self.a, self.amax, self.feed = horizon, a_dim, amax, feed
        self.use_zg, self.use_gate, self.use_z0 = use_zg, use_gate, use_z0
        self.use_grad = use_grad
        # vnorm (ported 2026-08-18 from overlays/lip.py, the tworoom/cube
        # overlay): conditioning of the two value-derived inputs. 'none' = the
        # shipped behaviour (raw E, raw grad); 'log' = log1p(E); 'loggn' =
        # log1p(E) + RMS-normalised grad_A V. in_dim is unchanged, so
        # state_dicts stay interchangeable across vnorm settings.
        self.vnorm = str(vnorm)
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
        in_dim = (horizon * a_dim + int(use_grad) * horizon * a_dim + 1
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
        E_in = E.reshape(B, 1)
        if self.vnorm in ("log", "loggn"):
            E_in = torch.log1p(E_in.clamp_min(0.0))
        parts = [A.reshape(B, -1)]
        if self.use_grad:
            g = gradA.reshape(B, -1)
            if self.vnorm == "loggn":
                g = g / g.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            parts.append(g)
        parts.append(E_in)
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

    def __init__(self, z_dim, horizon=5, a_dim=25, hidden=512, s_dim=256,
                 amax=2.5, use_z0=False, use_zg=False, use_grad=True,
                 s0_mode="zero", head_scale=1.0):
        super().__init__()
        self.h, self.a, self.amax = horizon, a_dim, amax
        self.use_z0, self.use_zg, self.use_grad = use_z0, use_zg, use_grad
        self.s_dim, self.s0_mode = s_dim, s0_mode
        # kept for checkpoint-flag parity with PlannerNet (no gate here)
        self.use_gate = False
        in_dim = (horizon * a_dim + int(use_grad) * horizon * a_dim + 1
                  + (int(use_z0) + int(use_zg)) * z_dim)
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

    def init_state(self, B, z0):
        if self.s0_mode == "z0":
            return torch.tanh(self.s0_proj(z0))
        return z0.new_zeros(B, self.s_dim)

    def forward(self, A, gradA, E, z0, zg, s, ztraj=None, k=0, vtraj=None):
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


def rollout_terminal(wm, z_hist, a_hist, plan, compat=False, hs=3):
    """Pooled-latent WM: autoregressive H-block unroll; returns terminal latent.

    z_hist: (B, 3, D) latent history, a_hist: (B, 2, a) action-block history,
    plan: (B, H, a). Differentiable w.r.t. ``plan``.
    """
    return rollout_traj(wm, z_hist, a_hist, plan, compat=compat, hs=hs)[:, -1]


def rollout_traj(wm, z_hist, a_hist, plan, compat=False, hs=3):
    """Like :func:`rollout_terminal` but returns all H imagined latents (B, H, D).

    ``compat=True`` selects the CEM-identical windowing (see
    :func:`rollout_traj_cem`); ``z_hist`` must then be the REAL encoder frames
    only (no padding) and ``a_hist`` is ignored.
    """
    if compat:
        return rollout_traj_cem(wm, z_hist, plan, hs=hs)
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


def rollout_traj_cem(wm, z_real, plan, hs=3):
    """Unroll the WM exactly as ``LeWM.rollout`` does — the CEM/MPPI/Adam path.

    The two planners did not share an interface. ``LeWM.rollout`` (used by
    every sampling/gradient solver) splits the plan as
    ``act_0, act_future = split(action_sequence, [H, T - H])`` where
    ``H = info['pixels'].size(2)`` is the number of REAL frames the env pool
    hands the policy. ``EnvPool`` stacks infos as ``(num_envs, 1, ...)``, so at
    eval time H = 1: the whole plan is future actions, and the attention window
    *grows* 1, 2, 3, 3, ... frames, truncated to the last ``hs``. No action slot
    is ever fabricated.

    LIP's own :func:`rollout_traj` instead always builds a full ``hs``-frame
    window, padding it with copies of z0 and with ZERO action blocks for the
    first ``hs - 1`` steps. Same (state_i, action_i) -> state_{i+1} pairing and
    the same number of optimized dims, but different context for the first
    ``hs - 1`` imagined steps, and a different positional-embedding slice
    (``pos[:hs]`` from step 0 instead of ``pos[:1]``, ``pos[:2]``, ...).

    Args:
        z_real: (B, H, D) real encoder frames available at plan time.
        plan: (B, T, a) action blocks, T >= H.
        hs: predictor context length (``predictor.num_frames``).

    Returns:
        (B, T - H + 1, D) imagined latents — with H = 1, one per plan block,
        matching :func:`rollout_traj`'s output length.
    """
    H, T = z_real.shape[1], plan.shape[1]
    assert T >= H, f"plan length {T} < history frames {H}"
    embs = list(z_real.unbind(dim=1))
    acts = list(plan.unbind(dim=1))
    outs = []
    for t in range(T - H + 1):
        lo = max(0, H + t - hs)
        win_e = torch.stack(embs[lo : H + t], dim=1)
        win_a = torch.stack(acts[lo : H + t], dim=1)
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


    def _V_traj(self, x, zg):
        """Terminal-window score of an imagined trajectory (B, <=H, D).

        With a window value the last ``vframes`` imagined frames are stacked;
        fewer than that available (a short prefix, or the start of an episode)
        repeats the oldest, which is window_pair's -- and train_window.py's --
        clamp convention.
        """
        if self.vframes <= 1:
            return self.lip_value(x[:, -1], zg)
        return self.lip_value(*window_pair(x, zg, self.vframes))

    def _V(self, x, zg):
        """Score with the value, stacking an m-frame window when required.

        ``x`` is either the full imagined trajectory (B, H, D) or a terminal
        latent (B, D). With a window value a terminal-only input cannot be
        scored honestly -- repeating one frame would silently turn the window
        cost back into the terminal cost -- so that case raises.

        I3. When the policy publishes ``i3_chunks_remaining`` (a deadline is set
        and fewer than H plan chunks still fit before the graded step) the plan
        overshoots the deadline: its terminal is a state the episode never
        reaches, and optimising it grades the planner on step 70 while the
        metric reads step 50. Score the best STILL-REACHABLE chunk instead,
        ``min_j V(z_j, z_g)`` for ``j < min(H, chunks_remaining)``.

        This is a pure readout change -- the refinement loop, the actor and the
        value are untouched -- and it is inert whenever chunks_remaining >= H.
        At rh=5 with eval_budget 50 and plan_len 25 the count is 10, 5 at the
        two replans, so no reported number can move. Applies to the gradient,
        to the per-iteration E and to the final argmin alike, because all three
        go through this one entry point.
        """
        if x.dim() == 3:
            _cr = getattr(self, "i3_chunks_remaining", None)
            _H = x.shape[1]
            if _cr is not None and 0 < int(_cr) < _H:
                _n = int(_cr)
                # 'deadline': the one chunk that lands on the graded step.
                # 'min': the best still-reachable chunk (measured worse @0.05).
                _js = [_n - 1] if _I3_MODE == "deadline" else list(range(_n))
                if not getattr(self, "_i3_said", False):
                    print(f"[I3] mode={_I3_MODE} {_n}/{_H} chunks reachable -> "
                          f"chunks {_js} (vframes={self.vframes})", flush=True)
                    self._i3_said = True
                return torch.stack(
                    [self._V_traj(x[:, :j + 1], zg) for j in _js],
                    dim=0).min(dim=0).values
            return self._V_traj(x, zg)
        if self.vframes > 1:
            raise RuntimeError(
                f"window value ({self.vframes} frames) received a terminal-only "
                f"latent {tuple(x.shape)}; the caller must pass the trajectory")
        return self.lip_value(x, zg)
    def __init__(self, *args, actor_path: str = "", lam: float = 1.0,
                 use_warm_start: bool = False,
                 restarts: int = 1, restart_noise: float = 0.5,
                 lip_select: str = "last", robust_m: int = 0,
                 init_mode: str = "zero", init_samples: int = 64,
                 init_scale: float = 1.5, rollout_compat: bool = True,
                 plan_scale: float = 1.0, plan_clip: float | None = None,
                 use_frame_history: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        from stable_worldmodel.trm import load_metric

        self.lam = lam
        # see solve() for why plan_scale exists; 1.0 == identity == old behavior
        self.plan_scale = float(plan_scale)
        self.plan_clip = None if plan_clip is None else float(plan_clip)
        # use_frame_history: consume the real (pixels_hist, action_hist) that
        # WorldModelPolicy accumulates, instead of padding one observed frame
        # into 3 copies with 2 zero action blocks. The WM trained on 3 real
        # frameskip-5 frames plus real actions, so the pad is a train/deploy
        # mismatch: it puts the true action sequence's imagined terminal
        # 0.1216 rad from the goal where matched conditioning gives 0.0580,
        # against a 0.05 rad success tolerance.
        self.use_frame_history = bool(use_frame_history)
        # rollout_compat: roll the WM exactly as LeWM.rollout does for
        # CEM/MPPI/Adam (see rollout_traj_cem). False = the legacy LIP
        # convention (history padded to 3 copies of z0, 2 zero action blocks),
        # which is what every actor trained before 2026-07-27 saw.
        self.rollout_compat = bool(rollout_compat)
        # Default OFF. The warm start is a real behaviour change for a
        # trained refiner -- an actor trained only on A=0 collapses from
        # 100.0 to 1.0 @0.1 when handed a non-zero start -- so it must be
        # opted into explicitly (solver.use_warm_start=true) and can never
        # be inherited from plan_config.warm_start, which defaults True.
        self.use_warm_start = bool(use_warm_start)
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

        ck = torch.load(actor_path, map_location=self.device, weights_only=False)
        self.kind = ck.get("kind")
        if self.kind not in ("lip", "lip2", "lip3", "lip4", "lip4r", "lip_dino"):
            raise ValueError(f"LIPSolver: unsupported checkpoint kind {self.kind!r}")
        self.feed = ck.get("feed", "none") if self.kind == "lip2" else "none"
        self.temporal_objective = ck.get(
            "temporal_objective", ck.get("train_args", {}).get(
                "temporal_objective", "terminal"))
        if self.kind == "lip4r":
            self.actor = PlannerNetRec(ck["z_dim"], horizon=ck["horizon"],
                                       a_dim=ck.get("a_dim", 25),
                                       hidden=ck.get("hidden", 512),
                                       s_dim=ck.get("s_dim", 256),
                                       amax=ck.get("amax", 2.5),
                                       use_z0=ck.get("use_z0", False),
                                       use_zg=ck.get("use_zg", False),
                                       use_grad=ck.get("use_grad", True),
                                       s0_mode=ck.get("s0_mode", "zero")).to(self.device)
        elif self.kind == "lip3":
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
                                    use_z0=ck.get("use_z0", not v4),
                                    use_grad=ck.get("use_grad", True),
                                    vnorm=ck.get("vnorm", "none")).to(self.device)
        # A vnorm actor deployed with raw E is a guaranteed silent null, so say
        # out loud which conditioning this cell actually loaded.
        if getattr(self.actor, "vnorm", "none") != "none":
            print(f"[lip] actor vnorm={self.actor.vnorm}", flush=True)
        elif ck.get("vnorm", "none") != "none":
            raise ValueError(
                f"checkpoint trained with vnorm={ck['vnorm']!r} but the "
                f"reconstructed {self.kind!r} actor has no vnorm support")
        self.actor.load_state_dict(ck["sd"])
        self.actor.eval()
        self._actor_horizon = ck["horizon"]        # plan length must match the trained net
        self.lip_iters = ck["iters"]
        self.lip_value = load_metric(ck["value"], device=self.device)
        # m-frame window values declare window_frames in their arch (see
        # train_window.py). Read it from the blob rather than inferring widths.
        _vb = torch.load(ck["value"], map_location="cpu", weights_only=False)
        self.vframes = int((_vb.get("arch") or {}).get("window_frames", 1))
        if self.vframes > 1:
            print(f"[vframes] window value: {self.vframes} frames", flush=True)
        # m-frame window values declare latent_dim = m*D; every query below
        # must stack that many imagined frames and tile the goal
        if self.vframes > 1:
            print(f"[vframes] window value: {self.vframes} frames", flush=True)
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

    def _value_init(self, wm, z_hist, a_hist, zg, rollT=None):
        """A(0) = argmin-E over {zero, iid-Gaussian, time-tiled Gaussian} raw plans.

        The tiled half (one action block repeated across the horizon) covers
        sustained-motion plans — the long-transport family the zero-init
        refinement cannot reach when the value gradient is flat at A=0.
        """
        B, H, adim = z_hist.shape[0], self.horizon, self.action_dim
        amax = float(self.actor.amax)
        n_iid = self.init_samples // 2
        n_tile = self.init_samples - n_iid
        iid = self.init_scale * torch.randn(B, n_iid, H, adim,
                                            device=self.device, generator=self.torch_gen)
        tile = self.init_scale * torch.randn(B, n_tile, 1, adim,
                                             device=self.device, generator=self.torch_gen)
        cand = torch.cat([torch.zeros(B, 1, H, adim, device=self.device),
                          iid, tile.expand(B, n_tile, H, adim)], dim=1)
        cand = cand.clamp(-amax, amax)
        C = cand.shape[1]
        # full trajectory (rollout_terminal was only its [:, -1] slice), so window
        # values can stack the last m frames at every scoring site
        rollT = rollT or (lambda zh, ah, p: rollout_traj(wm, zh, ah, p))
        with torch.no_grad():
            term = rollT(z_hist.repeat_interleave(C, dim=0),
                         a_hist.repeat_interleave(C, dim=0),
                         cand.reshape(B * C, H, adim))
            E = self._V(term.float(), zg.repeat_interleave(C, dim=0)).view(B, C)
        return cand[torch.arange(B, device=self.device), E.argmin(dim=1)]

    # ------------------------------------------------------------------ lip
    def _proposal_lip(self, info_dict, n_envs, init_action=None):
        wm = self._base()
        with torch.no_grad():
            px_key = "pixels"
            if self.use_frame_history and info_dict.get("pixels_hist") is not None:
                # (B, <=3, C, H, W) of real frames at action_block spacing,
                # accumulated by WorldModelPolicy. Falls back to the single
                # observed frame for the first blocks of an episode, where a
                # history genuinely does not exist yet.
                px_key = "pixels_hist"
            px = info_dict[px_key].to(self.device, dtype=self.dtype)
            enc_in = {"pixels": px}
            if getattr(wm, "wants_proprio", False):
                pro_key = ("proprio_hist"
                           if px_key == "pixels_hist"
                           and info_dict.get("proprio_hist") is not None
                           else "proprio")
                pro = info_dict.get(pro_key)
                if pro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'proprio'")
                pro = torch.as_tensor(np.asarray(pro), dtype=torch.float32,
                                      device=self.device)
                enc_in["proprio"] = pro.reshape(px.shape[0], px.shape[1], -1)
            enc = wm.encode(enc_in)
            z_real = enc["emb"][:, -3:].float()    # REAL frames (1 unless history is on)
            z_hist = z_real
            if z_hist.shape[1] < 3:                # pad short history at episode start
                pad = z_hist[:, :1].expand(-1, 3 - z_hist.shape[1], -1)
                z_hist = torch.cat([pad, z_hist], dim=1)
            gx = info_dict["goal"].to(self.device, dtype=self.dtype)
            genc_in = {"pixels": gx}
            if getattr(wm, "wants_proprio", False):
                gpro = info_dict.get("goal_state")
                if gpro is None:
                    raise KeyError("proprio-variant WM: info_dict lacks 'goal_state' "
                                   "(goal agent position for the goal latent)")
                gpro = torch.as_tensor(np.asarray(gpro), dtype=torch.float32,
                                       device=self.device)
                gpro = gpro.reshape(gx.shape[0], -1)[:, -2:]   # last frame's (x, y)
                genc_in["proprio"] = gpro.unsqueeze(1).expand(-1, gx.shape[1], -1)
            zg = wm.encode(genc_in)["emb"][:, -1].float()

        R = max(1, self.restarts)
        B = n_envs
        # --- rollout interface (see rollout_traj_cem) ----------------------
        # compat: real frames + CEM's growing window, no fabricated actions.
        # legacy: 3-frame padded history + 2 zero action blocks.
        hs = int(getattr(getattr(wm, "predictor", None), "num_frames", 3))
        z_roll = z_real if self.rollout_compat else z_hist

        def roll(zh, ah, plan):
            return rollout_traj(wm, zh, ah, plan,
                               compat=self.rollout_compat, hs=hs)

        def rollT(zh, ah, plan):
            # full trajectory: window values stack the last m imagined frames,
            # and _V slices [:, -1] itself when the value is single-frame
            return roll(zh, ah, plan)

        zh_r = z_roll.repeat_interleave(R, dim=0)
        zg_r = zg.repeat_interleave(R, dim=0)
        z0_r = zh_r[:, -1]                         # last REAL frame either way
        a_hist = torch.zeros(B * R, 2, self.action_dim, device=self.device)
        if self.use_frame_history and info_dict.get("action_hist") is not None:
            # Real executed action blocks where the legacy path fed zeros. The
            # WM attends over (state, action) pairs, so zeros in the two most
            # recent history slots are inputs it never saw in training.
            ah = torch.as_tensor(np.asarray(info_dict["action_hist"]),
                                 dtype=torch.float32, device=self.device)
            ah = ah.reshape(B, -1, self.action_dim).to(self.dtype)
            k = min(ah.shape[1], a_hist.shape[1])
            if k > 0:
                a_hist[:, -k:] = ah[:, -k:].repeat_interleave(R, dim=0)
        A = self.restart_noise * torch.randn(B * R, self.horizon, self.action_dim,
                                             device=self.device, generator=self.torch_gen)
        A[::R] = 0.0                               # one zero-init restart per env
        # Warm start: policy.py carries the unexecuted plan tail forward and
        # passes it as init_action (cem.py consumes it; this path used to drop
        # it). Seed ONLY restart 0 so exploration across restarts is unchanged.
        #
        # MUST come AFTER the zero-init above. A is env-major (zh_r uses
        # repeat_interleave), so A[::R] is restart 0 of every env -- precisely
        # the row seeded here. Seeding first meant the zero-init erased the warm
        # start for any R, and at the shipped default restarts=1 A[::1] is the
        # WHOLE tensor, so the warm start had never taken effect in any lip
        # cell. The ablation caught it: on and off returned bit-identical cells.
        if init_action is not None and getattr(self, 'use_warm_start', True):
            _w = init_action.to(device=self.device, dtype=A.dtype)
            if _w.dim() == 3 and _w.shape[0] == B and _w.shape[-1] == self.action_dim:
                _seed = A.new_zeros(B, self.horizon, self.action_dim)
                _k = min(_w.shape[1], self.horizon)
                _seed[:, :_k] = _w[:, :_k]          # shift: tail becomes the prefix
                if _k < self.horizon:               # pad by repeating the last block
                    _seed[:, _k:] = _w[:, _k - 1:_k]
                _warm_seed = _seed
                A = A.view(B, R, self.horizon, self.action_dim)
                A[:, 0] = _warm_seed
                A = A.view(B * R, self.horizon, self.action_dim)
                if not getattr(self, "_ws_said", False):
                    print(f"[warm-start] seeding restart 0 with a "
                          f"{tuple(_w.shape)} plan tail", flush=True)
                    self._ws_said = True
        if self.init_mode == "value" and R == 1:
            A = self._value_init(wm, z_roll, a_hist, zg, rollT=rollT)
        s = self.actor.init_state(A.shape[0], z0_r) if self.kind == "lip4r" else None
        with torch.no_grad():
            e0 = self._V_traj(zh_r, zg_r).detach()
        def refine_score(tr):
            out = self._V(tr, zg_r)
            return out - e0 if self.temporal_objective == "tel-exact" else out
        buf = []
        for k_it in range(self.lip_iters):
            with torch.enable_grad():
                A_in = A.detach().requires_grad_(True)
                traj = roll(zh_r, a_hist, A_in)
                (gA,) = torch.autograd.grad(refine_score(traj).sum(), A_in)
            with torch.no_grad():
                traj_f = roll(zh_r, a_hist, A)
                E = refine_score(traj_f)
                vtraj = None
                gm = getattr(self.actor, "goal_mode", None)
                if gm == "sep" or (gm == "vonly" and
                                   getattr(self.actor, "cond", None) is None):
                    Bh, H = traj_f.shape[0], traj_f.shape[1]
                    vtraj = self.lip_value(
                        traj_f.reshape(Bh * H, -1),
                        zg_r.repeat_interleave(H, dim=0)).view(Bh, H)
                if self.kind == "lip4r":
                    A, s = self.actor(A, gA, E, z0_r, zg_r, s, traj_f, k=k_it, vtraj=vtraj)
                else:
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
                    scores = scores + self._V(rollT(zh_c, ah_c, pert), zg_c)
                Ef = (scores / self.robust_m).view(B, R * C)
            else:
                Ef = self._V(rollT(zh_c, ah_c, cf), zg_c).view(B, R * C)
            best = Ef.argmin(dim=1)
            A = cands.view(B, R * C, self.horizon, self.action_dim)[
                torch.arange(B, device=self.device), best]
        if os.environ.get("LIP_PROBE_DIR"):
            global _LIP_PROBE_N
            with torch.no_grad():
                z_traj = roll(z_roll, a_hist[:B], A)                # (B,H,D) imagined path
                z_imag = z_traj[:, -1]
                e_imag = self._V(z_imag, zg)
            torch.save(
                {"z0": z_hist[:, -1].detach().cpu(), "z_traj": z_traj.detach().cpu(),
                 "z_imag": z_imag.detach().cpu(), "A": A.detach().cpu(),
                 "E": e_imag.detach().cpu(), "zg": zg.detach().cpu()},
                f"{os.environ['LIP_PROBE_DIR']}/probe_{_LIP_PROBE_N:04d}.pt")
            _LIP_PROBE_N += 1
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
            proposal = self._proposal_lip(info_dict, total_envs,
                                          init_action=init_action)
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

        # Deploy-time plan rescale. Probe F measured LIP reaching a BETTER
        # terminal value than CEM (0.278 vs 0.314) and a better imagined
        # terminal (0.0397 vs 0.0422 rad) while still losing the card, with
        # plan rms 0.32 against CEM's 0.58 and the data's 1.00 -- the minimizer
        # set is broad and biased toward weak torques, so the actor picks a
        # valid-but-limp member of it. Scaling at deploy time tests that
        # directly without retraining. plan_scale=1.0 is the identity, so the
        # default reproduces every number measured before this existed.
        if self.plan_scale != 1.0:
            mean = mean * self.plan_scale
            if self.plan_clip is not None:
                # Past the physical box MuJoCo clips anyway; clipping here keeps
                # the plan we score identical to the one that gets executed.
                mean = mean.clamp(-self.plan_clip, self.plan_clip)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(f"LIP solve time: {time.time() - start_time:.4f} seconds")
        return outputs
