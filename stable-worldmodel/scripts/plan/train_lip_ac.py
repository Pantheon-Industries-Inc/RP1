"""LIP-AC: train the TD value (critic) and the LIP planner (actor) in tandem.

The sequential pipeline (train_metric.py -> select TD by CEM -> train_lip.py)
selects the value by a zeroth-order criterion — CEM only needs correct *ranking*
of sampled plans — but LIP consumes the value's *gradient field* through the WM,
so the CEM-best TD need not be the best LIP teacher (observed on PushT, where a
rugged plan landscape reversed the method ordering). Here the two train jointly,
DDPG/TD3-style, with the planner as a K-step learned-optimizer actor:

  critic   d_phi(z, z_g): n-step expectile TD on the dense (fs1) latent cache.
           This TD loss is the ONLY gradient that reaches phi — the actor loss
           is computed through the frozen-parameter teacher, so the critic can
           never learn to "make plans look good" (the classic collapse).
  teacher  d_bar = EMA(d_phi) (Polyak, --ema-tau): serves TD bootstrap targets
           AND the actor (gradient features, refinement loss, and what we save
           for deployment — so training matches LIPSolver's plan-time value).
  actor    f_theta: exactly train_lip.py's objective, against the teacher.

Schedule: --pretrain critic-only warmup (auto: 2000 if fresh, 0 if warm-started
via --init-value), then --critic-ratio critic steps per actor step, critic+EMA
frozen after --freeze-critic-frac of actor steps so the planner settles on a
stationary teacher (LIP is lr-sensitive; a moving target late in training makes
it chase noise).

Optional feedback loop (--expand-weight > 0): the actor's own imagined terminal
latents become extra TD backups  d(z0,zg) <- H*fs + d_bar(z_T,zg)  — value
expansion on planner-visited states, shaping d exactly where the planner
travels. Low expectile makes this a one-sided (optimistic) bound absorber:
good plans tighten d, bad plans barely raise it. Off by default: it lets the
pair co-exploit WM errors, so compare against expand-weight 0 before trusting.

Outputs: --out (actor checkpoint, train_lip.py format, kind='lip'/'lip2') and
--out-value (the teacher via save_metric). --out records --out-value as its
`value`, so eval_wm.py `solver=lip solver.actor_path=...` works unchanged, and
the same value file can be CEM-evaluated via `+metric=` for apples-to-apples.
"""
import argparse
import copy
import math

import h5py

try:
    import hdf5plugin  # noqa: F401  (registers HDF5 compression filters, e.g. cube h5)
except ImportError:
    pass
import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.solver.lip import (
    PlannerNet, PlannerNetV3, PlannerNetRec, rollout_terminal, rollout_traj)
from stable_worldmodel.trm import LatentCache, load_metric
from stable_worldmodel.trm.io import build_metric, save_metric
from stable_worldmodel.trm.learners.td import _expectile_loss
from stable_worldmodel.trm.samplers import NStepGoalSampler


def _cos(base, final, t, T):
    """Cosine decay base -> final over T steps; base when final is None."""
    if final is None or T <= 0:
        return base
    t = min(t, T)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t / T))


def main():
    p = argparse.ArgumentParser()
    # data / models
    p.add_argument("--cache", required=True, help="fs5 latent cache (actor rollout contexts)")
    p.add_argument("--cache-td", required=True, help="fs1 latent cache (critic TD pairs; dense stride)")
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--init-value", default="", help="warm-start critic from a train_metric.py TD checkpoint")
    p.add_argument("--out", required=True, help="actor checkpoint (train_lip.py format)")
    p.add_argument("--out-value", required=True, help="final teacher checkpoint (save_metric format)")
    # actor (train_lip.py parity)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--steps", type=int, default=8000, help="actor steps")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-delta", type=int, default=10)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--actor-lr-final", type=float, default=None,
                   help="cosine-decay actor lr to this over all steps (None = constant)")
    p.add_argument("--mean-weight", type=float, default=0.1)
    p.add_argument("--feed", choices=["none", "end", "traj"], default="none")
    p.add_argument("--arch", choices=["v4", "v4r", "mlp", "traj"], default="v4",
                   help="'v4' (default) = LIPv4: minimal-input gate-free MLP "
                        "[A, grad V, E] — no raw z0/zg, pure residual update "
                        "(kind='lip4'); 'v4r' = recurrent LIPv4 (kind='lip4r'): "
                        "same inputs, GRU hidden state carried across the K "
                        "iterations (see --s-dim/--s0-mode); 'mlp' = legacy "
                        "full-input gated PlannerNet (ablate via "
                        "--drop-z0/--drop-zg/--no-gate); 'traj' = "
                        "PlannerNetV3 transformer (kind='lip3')")
    p.add_argument("--s-dim", type=int, default=256,
                   help="v4r: recurrent hidden-state dimension")
    p.add_argument("--s0-mode", choices=["zero", "z0"], default="zero",
                   help="v4r: initial hidden state — 'zero' (s0=0) or 'z0' "
                        "(s0=tanh(W z0), memory seeded from the current latent)")
    p.add_argument("--rec-hidden", type=int, default=512,
                   help="v4r: input-embedding width feeding the GRU cell")
    p.add_argument("--width", type=int, default=256, help="v3 transformer width")
    p.add_argument("--layers", type=int, default=2, help="v3 transformer layers")
    p.add_argument("--zero-init", action="store_true",
                   help="v3: near-identity init (zero dA head, 0.02x embeddings)")
    p.add_argument("--gd-init", type=float, default=0.0,
                   help="v3 vonly: add -eta_k * rms-normalized-grad to the update, "
                        "eta learnable per iteration, init at this value; with "
                        "--zero-init the refiner starts as normalized GD (L2O warm start)")
    p.add_argument("--feat-norm", action="store_true",
                   help="v3 vonly: rms-normalize grad feature, V/25, LayerNorm tokens")
    p.add_argument("--iter-mode", choices=["emb", "scalar"], default="scalar",
                   help="v3 iteration conditioning: 'scalar' = k/(K-1) appended per "
                        "token (default); 'emb' = learned per-iteration embedding "
                        "(rounds V/Z legacy)")
    p.add_argument("--head-mode", choices=["gate", "precond"], default="gate",
                   help="v3 head: 'precond' = dA = -softplus(p) * ghat + r (learned "
                        "per-entry preconditioner on normalized GD + residual; starts "
                        "as GD via bias init, no gate, all gradient paths live); "
                        "'gate' = legacy dA*sigmoid(gate)")
    p.add_argument("--p0", type=float, default=0.05,
                   help="precond head: initial per-entry GD step size (bias init)")
    p.add_argument("--head-scale", type=float, default=1.0,
                   help="scale the update-head init by this (all arches; e.g. 0.01 = "
                        "near-identity start with live gradients; for gate-free "
                        "v4 this covers the gate's damping role at init)")
    p.add_argument("--pre-ln", action="store_true",
                   help="v3: pre-LN transformer blocks (norm_first) — stability variant")
    p.add_argument("--drop-zg", action="store_true",
                   help="mlp arch: drop the raw goal embedding from the input — the "
                        "goal reaches the actor only via the teacher (E, grad V)")
    p.add_argument("--drop-z0", action="store_true",
                   help="mlp arch: drop the raw current-state embedding; with "
                        "--drop-zg too, the actor sees only [A, grad V, E]")
    p.add_argument("--no-gate", action="store_true",
                   help="v3: drop the per-token gate — pure residual A' = A + dA "
                        "(gate is expressively redundant; small-init covers its "
                        "Jacobian-damping role at init)")
    p.add_argument("--drop-grad", action="store_true",
                   help="mlp arch: ABLATION — drop grad_A V from the actor input; "
                        "it plans from [A, E, z0, zg] with no value-gradient signal. "
                        "Only informative with z0/zg kept (a full-input actor).")
    p.add_argument("--cond-mode", choices=["token", "global"], default="token",
                   help="v3 vonly scalars: 'token' = V(z_t,zg) + k appended per token; "
                        "'global' (v3.2) = final value E and k enter via a learned "
                        "Linear(2->width) bias added to all tokens — dedicated path, "
                        "per-step values dropped")
    p.add_argument("--goal-mode", choices=["diff", "sep", "vonly"], default="diff",
                   help="v3 goal conditioning: 'diff' = proj(zg - z_t) (assumes linear "
                        "latent geometry); 'sep' = proj(zg) + per-step teacher value; "
                        "'vonly' = [A_t, grad_t, RAW z_t, V(z_t, zg)] — no projections, "
                        "goal only via the learned quasimetric")
    p.add_argument("--amax", type=float, default=2.5,
                   help="plan clamp in z-scored action units (expert tails reach ~3.5 on "
                        "cube). CUBE: use 1.6 -- swept 1.0-3.5 x 3 seeds x 3 draws, flat "
                        "optimum over 1.4-2.2 (86.4 3-seed) and 3.5 costs ~7 pts plus a "
                        "catastrophic-seed mode (spread 20.7 -> 4.7). Default left at 2.5 "
                        "because reacher/tworoom share this file. See "
                        "Dyna/RESULTS_amax_sweep.md")
    # critic (train_metric.py --learner td parity)
    p.add_argument("--head", choices=["mlp", "quasimetric"], default="quasimetric")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--expectile", type=float, default=0.1)
    p.add_argument("--expectile-final", type=float, default=None,
                   help="linearly anneal expectile to this over the live phase (None = constant); "
                        "e.g. 0.1 -> 0.03: smooth teacher early, sharp shortest-path late")
    p.add_argument("--n-step", type=int, default=25)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--td-batch", type=int, default=1024)
    p.add_argument("--td-p-cross", type=float, default=0.3)
    p.add_argument("--td-max-delta", type=int, default=None)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr-final", type=float, default=None,
                   help="cosine-decay critic lr to this over the live phase (None = constant); "
                        "converges the teacher instead of letting it wander the TD basin")
    p.add_argument("--critic-wd", type=float, default=1e-4)
    p.add_argument("--huber-beta", type=float, default=1.0)
    # tandem schedule
    p.add_argument("--pretrain", type=int, default=-1,
                   help="critic-only warmup steps (-1 = auto: 0 if --init-value else 2000)")
    p.add_argument("--critic-ratio", type=int, default=1, help="critic steps per actor step")
    p.add_argument("--ema-tau", type=float, default=0.005)
    p.add_argument("--freeze-critic-frac", type=float, default=0.8,
                   help="freeze critic+teacher after this fraction of actor steps (1.0 = never)")
    p.add_argument("--expand-weight", type=float, default=0.0,
                   help="weight of TD backups along planner rollouts (0 = off)")
    p.add_argument("--replay-prob", type=float, default=0.0,
                   help="replan curriculum: fraction of actor-step contexts replaced by "
                        "the previous step's final imagined window (last 3 imagined "
                        "latents, same goal, zero action history — the exact query the "
                        "deployed receding-horizon replan makes). Trains the refiner to "
                        "CONTINUE from its own mid-task states instead of only expert "
                        "cache states. Inputs only — the critic's TD targets stay "
                        "expert-anchored.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--action-stats-pin", default=None,
                   help="pin action z-score stats instead of computing from --h5. "
                        "'expert' = canonical lewm-cube expert stats (use when the "
                        "cache/h5 is a mixture but eval un-scales with expert stats). "
                        "A *.json path pins {'mean': [...], 'std': [...]} explicitly "
                        "(e.g. the campaign-wide play stats a WM was trained with — "
                        "required when the h5's own stats are degenerate, e.g. an "
                        "expert gripper column with std 0).")
    p.add_argument("--bc-weight", type=float, default=0.0,
                   help="trust-region: add lambda * ||A - a_ref||^2 to the actor loss, "
                        "a_ref = the data's next-H real action blocks from the query state. "
                        "Fences the actor inside the data-action distribution (where the WM "
                        "is accurate) so it can't optimize into WM hallucination holes. "
                        "0 = off (default, preserves stock behavior).")
    a = p.parse_args()
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)

    # ------------------------------------------------------------ actor data (fs5 + h5)
    c = LatentCache.load(a.cache)
    z = c.z.to(dev).float()
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    with h5py.File(a.h5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
    # nan-aware: some datasets (e.g. lewm-cube) pad episode-terminal steps with
    # NaN actions; those rows are never sampled (history blocks stop before the
    # terminal step) but must not poison the normalization stats
    amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    if a.action_stats_pin == "expert":
        # mixture actions z-scored with EXPERT stats so the actor operates in
        # the same action convention the eval un-scales with (eval-consistency).
        amu = np.array([0.010831, -0.003126, 0.002633, 0.000422, 0.15846])
        astd = np.array([0.2887, 0.392736, 0.641535, 0.391823, 0.249935])
        print(f"[action-pin] LIP using fixed expert stats mean={amu.tolist()} "
              f"std={astd.tolist()}", flush=True)
    elif a.action_stats_pin and a.action_stats_pin.endswith(".json"):
        import json

        with open(a.action_stats_pin) as f:
            _st = json.load(f)
        amu = np.asarray(_st["mean"], dtype=np.float64)
        astd = np.asarray(_st["std"], dtype=np.float64)
        assert amu.shape == astd.shape == (act.shape[-1],), \
            f"pinned stats dim {amu.shape} != action dim {act.shape[-1]}"
        assert (astd > 1e-5).all(), f"pinned std degenerate: {astd.tolist()}"
        print(f"[action-pin] LIP using stats from {a.action_stats_pin} "
              f"mean={amu.tolist()} std={astd.tolist()}", flush=True)
    elif a.action_stats_pin:
        raise ValueError(f"unknown --action-stats-pin {a.action_stats_pin!r} "
                         "(expected 'expert' or a *.json path)")
    act_n = ((act - amu) / astd).astype(np.float32)
    fs = 5                                     # primitive steps per action block
    a_dim = act.shape[-1] * fs

    def blocks(e, t):
        h0 = int(ep_off[e] + fs * t)
        return act_n[h0:h0 + fs].reshape(-1)

    def sample(B):
        zh, ah, zg, aref = [], [], [], []
        for _ in range(B):
            e = ep_ids[rng.integers(len(ep_ids))]
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
            # trust-region reference: the data's next-H real action blocks from t
            # (clamped at episode end) — where the WM knows the dynamics.
            # Clamp to L-2, the last FULL in-episode block: block L-1 starts at
            # the episode's final primitive step, so its 5-step read spills into
            # the next episode (silently wrong) and runs off the array end for
            # the dataset's last episode (ragged np.stack crash — hit on
            # puzzle 2026-07-23 with 1001-step episodes, e=999/t=197).
            aref.append(np.stack([blocks(e, min(t + k, L - 2)) for k in range(a.horizon)]))
            if rng.random() < a.p_cross:
                e2 = ep_ids[rng.integers(len(ep_ids))]
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t + d, L - 1)]])
        return (torch.stack(zh), torch.from_numpy(np.stack(ah)).to(dev), torch.stack(zg),
                torch.from_numpy(np.stack(aref)).to(dev).float())

    # ------------------------------------------------------------ critic (fs1 cache)
    c_td = LatentCache.load(a.cache_td)
    if a.init_value:
        blob = torch.load(a.init_value, map_location="cpu", weights_only=False)
        assert blob["latent_dim"] == c_td.latent_dim, "init-value latent dim mismatch"
        arch = blob["arch"]
        critic = load_metric(a.init_value, device=dev)
    else:
        arch = {"head": a.head, "hidden_dim": a.hidden_dim, "depth": a.depth,
                "embed_dim": a.embed_dim, "softplus": True, "symmetric": False}
        critic = build_metric("td", c_td.latent_dim, arch).to(dev)
    critic.train()
    teacher = copy.deepcopy(critic).to(dev)
    for prm in teacher.parameters():
        prm.requires_grad_(False)               # actor loss flows THROUGH, never INTO
    teacher.eval()

    td_sampler = NStepGoalSampler(c_td, n_step=a.n_step, p_cross=a.td_p_cross,
                                  n_buckets=10, balanced=True, seed=a.seed,
                                  max_delta=a.td_max_delta)
    c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)

    n_plan = a.horizon * fs                     # plan length in primitive steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan), 1.0
    else:
        plan_disc = a.gamma ** n_plan
        plan_cost = (1.0 - plan_disc) / (1.0 - a.gamma)

    def critic_step(expand=None, tau=None, lr=None):
        tau = a.expectile if tau is None else tau
        if lr is not None:
            for pg in c_opt.param_groups:
                pg["lr"] = lr
        b = td_sampler.sample(a.td_batch)
        z_t, z_tn, z_g = b["z_t"].to(dev), b["z_tn"].to(dev), b["z_g"].to(dev)
        ne, reached, dist = b["n_eff"].to(dev), b["reached"].to(dev), b["dist"].to(dev)
        with torch.no_grad():
            d_next = teacher(z_tn, z_g)
            if a.gamma >= 1.0:
                cost, disc = ne, torch.ones_like(ne)
            else:
                disc = a.gamma ** ne
                cost = (1.0 - disc) / (1.0 - a.gamma)
            tgt = reached * dist + (1.0 - reached) * (cost + disc * d_next)
        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)
        if expand is not None:                  # value expansion on planner rollouts
            z0e, zTe, zge = expand
            with torch.no_grad():
                tgt_e = plan_cost + plan_disc * teacher(zTe, zge)
            loss = loss + a.expand_weight * _expectile_loss(
                critic(z0e, zge) - tgt_e, tau, a.huber_beta)
        c_opt.zero_grad(set_to_none=True)
        loss.backward()
        c_opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), critic.parameters()):
                tp.mul_(1.0 - a.ema_tau).add_(a.ema_tau * sp)
        return loss.item()

    # ------------------------------------------------------------ actor
    if a.arch == "traj":
        if (a.gd_init > 0 or a.feat_norm or a.cond_mode == "global") and a.goal_mode != "vonly":
            raise ValueError("--gd-init/--feat-norm/--cond-mode global need --goal-mode vonly")
        net = PlannerNetV3(z.shape[-1], horizon=a.horizon, a_dim=a_dim,
                           width=a.width, layers=a.layers, amax=a.amax,
                           n_iters=a.iters, goal_mode=a.goal_mode,
                           zero_init=a.zero_init, gd_init=a.gd_init,
                           feat_norm=a.feat_norm, iter_mode=a.iter_mode,
                           head_mode=a.head_mode, p0=a.p0,
                           head_scale=a.head_scale, cond_mode=a.cond_mode,
                           use_gate=not a.no_gate, pre_ln=a.pre_ln).to(dev)
    elif a.arch == "v4r":
        net = PlannerNetRec(z.shape[-1], horizon=a.horizon, a_dim=a_dim,
                            hidden=a.rec_hidden, s_dim=a.s_dim, amax=a.amax,
                            use_z0=False, use_zg=False, use_grad=not a.drop_grad,
                            s0_mode=a.s0_mode, head_scale=a.head_scale).to(dev)
    elif a.arch == "v4":
        net = PlannerNet(z.shape[-1], horizon=a.horizon, a_dim=a_dim, feed=a.feed,
                         amax=a.amax, use_zg=False, use_gate=False, use_z0=False,
                         use_grad=not a.drop_grad,
                         head_scale=a.head_scale).to(dev)
    else:
        net = PlannerNet(z.shape[-1], horizon=a.horizon, a_dim=a_dim, feed=a.feed,
                         amax=a.amax, use_zg=not a.drop_zg,
                         use_gate=not a.no_gate, use_z0=not a.drop_z0,
                         use_grad=not a.drop_grad,
                         head_scale=a.head_scale).to(dev)
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=1e-5)

    replay_buf = {"zh": None, "zg": None}   # previous step's final imagined windows

    def actor_step():
        zh, ah, zg, aref = sample(a.batch)
        if a.replay_prob > 0 and replay_buf["zh"] is not None:
            pick = torch.rand(a.batch, device=dev) < a.replay_prob
            idx = torch.nonzero(pick).squeeze(1)
            if idx.numel():
                take = torch.randint(0, replay_buf["zh"].shape[0], (idx.numel(),), device=dev)
                zh, ah, zg = zh.clone(), ah.clone(), zg.clone()
                zh[idx] = replay_buf["zh"][take]
                zg[idx] = replay_buf["zg"][take]
                ah[idx] = 0.0        # deployed replans query with zero action history
        z0 = zh[:, -1]
        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)
        s = net.init_state(a.batch, z0) if a.arch == "v4r" else None
        e_path, zT, tr = [], None, None
        for k in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            with torch.enable_grad():
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(wm, zh, ah, A_in)
                (gA,) = torch.autograd.grad(teacher(traj[:, -1], zg).sum(), A_in)
            # A_in holds A's values, so the grad pass's trajectory IS the feature
            # trajectory — reuse it instead of a third WM rollout (value-identical)
            traj_f = traj.detach()
            E_feat = teacher(traj_f[:, -1], zg).detach()
            vtraj = None
            if a.goal_mode == "sep" or (a.goal_mode == "vonly" and a.cond_mode == "token"):
                vtraj = teacher(traj_f.reshape(-1, traj_f.shape[-1]),  # per-step V(z_t, zg)
                                zg.repeat_interleave(a.horizon, dim=0)
                                ).view(a.batch, a.horizon).detach()
            if a.arch == "v4r":
                A, s = net(A, gA.detach(), E_feat, z0, zg, s, traj_f, k=k, vtraj=vtraj)
            else:
                A = net(A, gA.detach(), E_feat, z0, zg, traj_f, k=k, vtraj=vtraj)
            tr = rollout_traj(wm, zh, ah, A)
            zT = tr[:, -1]
            e_path.append(teacher(zT, zg).mean())
        if a.replay_prob > 0:
            replay_buf["zh"] = tr[:, -3:].detach()
            replay_buf["zg"] = zg.detach()
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        if a.bc_weight > 0:                       # trust-region toward data actions
            loss = loss + a.bc_weight * ((A - aref) ** 2).mean()
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        expand = (z0.detach(), zT.detach(), zg.detach())
        return e_path[0].item(), e_path[-1].item(), expand

    # ------------------------------------------------------------ schedule
    pretrain = a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000)
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            print(f"pretrain {i}: td_loss {cl:.4f}", flush=True)

    freeze_at = int(a.freeze_critic_frac * a.steps)
    expand = None
    for step in range(a.steps):
        critic_live = step < freeze_at
        if step == freeze_at:
            print(f"step {step}: critic+teacher frozen for the remaining "
                  f"{a.steps - freeze_at} actor steps", flush=True)
        # during-run schedules: teacher converges (lr decay) and sharpens
        # (expectile anneal) over the live phase; actor lr decays over all steps
        tau_s = a.expectile if a.expectile_final is None else (
            a.expectile + (a.expectile_final - a.expectile)
            * min(step, freeze_at) / max(freeze_at, 1))
        clr_s = _cos(a.critic_lr, a.critic_lr_final, step, freeze_at)
        alr_s = _cos(a.actor_lr, a.actor_lr_final, step, a.steps)
        for pg in a_opt.param_groups:
            pg["lr"] = alr_s
        cl = float("nan")
        if critic_live:
            for _ in range(a.critic_ratio):
                cl = critic_step(expand if a.expand_weight > 0 else None,
                                 tau=tau_s, lr=clr_s)
                expand = None                   # consume each actor batch once
        e_first, e_final, expand = actor_step()
        if step % 500 == 0:
            print(f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                  f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}",
                  flush=True)

    # ------------------------------------------------------------ save (teacher first;
    # the actor checkpoint references it, and it is what the actor optimized against)
    save_metric(teacher.cpu(), "td", c_td.latent_dim, arch, a.out_value)
    print(f"saved teacher value -> {a.out_value}", flush=True)
    net.eval()
    if a.arch == "traj":
        kind = "lip3"
    elif a.arch == "v4r":
        kind = "lip4r"
    elif a.arch == "v4":
        kind = "lip4"
    else:
        kind = "lip" if a.feed == "none" else "lip2"
    torch.save({"kind": kind, "feed": a.feed, "sd": net.cpu().state_dict(),
                "z_dim": z.shape[-1], "horizon": a.horizon, "iters": a.iters,
                "a_dim": a_dim, "amax": a.amax, "width": a.width, "layers": a.layers,
                "s_dim": a.s_dim, "s0_mode": a.s0_mode, "hidden": a.rec_hidden,
                "goal_mode": a.goal_mode, "gd_init": a.gd_init,
                "feat_norm": a.feat_norm, "iter_mode": a.iter_mode,
                "head_mode": a.head_mode, "cond_mode": a.cond_mode,
                "use_gate": getattr(net, "use_gate", not a.no_gate),
                "use_zg": getattr(net, "use_zg", not a.drop_zg),
                "use_z0": getattr(net, "use_z0", not a.drop_z0),
                "use_grad": getattr(net, "use_grad", not a.drop_grad),
                "head_scale": a.head_scale, "pre_ln": a.pre_ln,
                "value": a.out_value}, a.out)
    print(f"saved learned planner -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
