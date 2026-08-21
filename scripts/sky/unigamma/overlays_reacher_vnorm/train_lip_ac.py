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
import gc
import math
import os
import shutil
import time

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
    p.add_argument("--recipe-profile", default="auto", choices=[
        "auto", "generic", "reacher-lewm", "reacher-pldm",
    ], help="established task/base recipe used for every option not explicitly supplied")
    # data / models
    p.add_argument("--cache", required=True, help="fs5 latent cache (actor rollout contexts)")
    p.add_argument("--cache-td", required=True, help="fs1 latent cache (critic TD pairs; dense stride)")
    p.add_argument("--cache-device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="actor-cache placement; auto keeps caches below 8 GiB on GPU")
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--init-value", default="", help="warm-start critic from a train_metric.py TD checkpoint")
    p.add_argument("--out", required=True, help="actor checkpoint (train_lip.py format)")
    p.add_argument("--out-value", required=True, help="final teacher checkpoint (save_metric format)")
    # actor (train_lip.py parity)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--steps", type=int, default=None, help="actor steps (profile default)")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="snapshot the actor every N steps to <out>.step{N}.pt so a "
                        "selection pass can early-stop on HELD-at-end rather than on "
                        "the training objective (which anti-predicts it, r=+0.58)")
    p.add_argument("--resume-state", default="",
                   help="resume an actor-only run from a full training-state checkpoint")
    p.add_argument("--save-train-state", default="",
                   help="atomically save full actor-only optimizer/RNG/replay state here")
    p.add_argument("--state-every", type=int, default=0,
                   help="save resumable actor-only state every N completed actor steps")
    p.add_argument("--stop-after", type=int, default=0,
                   help="stop after this absolute step while retaining --steps as the "
                        "LR-schedule horizon; 0 runs through --steps")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--reuse-refinement-rollouts", action="store_true",
                   help="reuse iteration k's scored rollout as iteration k+1's "
                        "gradient rollout (K=8: 16 -> 9 WM rollouts)")
    p.add_argument("--temporal-objective",
                   choices=["terminal", "tel-exact", "tel-stopprev"],
                   default="terminal",
                   help="actor rollout score: terminal V_H; exact telescoping "
                        "sum(V_{h+1}-V_h); or the same forward scalar with each "
                        "negative V_h detached so intermediate gradients do not cancel")
    p.add_argument("--pad-context", action=argparse.BooleanOptionalAction, default=None,
                   help="train under the 1-frame deployment conditioning (THE paper "
                        "config): context collapsed to [z0]*hs with zero action "
                        "history, instead of real multi-frame cache windows")
    p.add_argument("--max-delta", type=int, default=None)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--actor-lr", type=float, default=None)
    p.add_argument("--actor-lr-final", type=float, default=None,
                   help="cosine-decay actor lr to this over all steps (None = constant)")
    p.add_argument("--align-mode", default="terminal",
                   choices=["terminal", "dense", "prefix", "randhorizon"],
                   help="which rollout timesteps enter the loss; see the "
                        "patch docstring. terminal = historical behaviour.")
    p.add_argument("--align-weight", type=float, default=0.0,
                   help="weight on the alignment term (0 = off)")
    p.add_argument("--mean-weight", type=float, default=None)
    p.add_argument("--lambda-schedule", choices=["uniform", "geom-early", "geom-late"],
                   default="uniform",
                   help="how the per-iteration values v_k are weighted in the actor "
                        "loss. uniform = the historical mean(); geom-early weights the "
                        "FIRST refinement iterations (regularises against world-model "
                        "exploitation, which anti-correlates with success at r=+0.58); "
                        "geom-late weights the LAST ones (the falsification twin)")
    p.add_argument("--lambda-base", type=float, default=0.5,
                   help="geometric ratio for the non-uniform schedules")
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
    p.add_argument("--vnorm", choices=["none", "log", "loggn", "scale"], default="none",
                   help="v4: conditioning of the value-derived actor inputs "
                        "(ported 2026-08-18 from the tworoom/cube overlay). "
                        "'none' = shipped raw E and raw grad; 'log' = log1p(E); "
                        "'loggn' = log1p(E) + RMS-normalised grad_A V. Saved "
                        "into the checkpoint and reapplied by LIPSolver at "
                        "deploy.")
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
    p.add_argument("--amax", type=float, default=None,
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
    p.add_argument("--expectile", type=float, default=None)
    p.add_argument("--expectile-final", type=float, default=None,
                   help="linearly anneal expectile to this over the live phase (None = constant); "
                        "e.g. 0.1 -> 0.03: smooth teacher early, sharp shortest-path late")
    p.add_argument("--n-step", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--squash", choices=["hard", "tanh"], default="hard",
                   help="plan box enforcement: hard clamp (zero grad when "
                        "saturated) or amax*tanh(u/amax) (gradient flows "
                        "everywhere). Saved into the checkpoint.")
    p.add_argument("--boundary", choices=["legacy", "smooth", "disc"], default="legacy",
                   help="n-step seam at gamma<1: legacy = raw in-window label vs "
                        "discounted bootstrap (non-monotone at delta=n); smooth = "
                        "boot n_eff + g^n_eff*d_next (continuous, per-window "
                        "discounting); disc = discount the exact branch too")
    p.add_argument("--td-batch", type=int, default=1024)
    p.add_argument("--td-p-cross", type=float, default=0.3)
    p.add_argument("--v4-hidden", type=int, default=512,
                   help="v4 refiner MLP width (round-tripped via the checkpoint)")
    p.add_argument("--v4-layers", type=int, default=2,
                   help="v4 refiner hidden layers (2 = shipped net)")
    p.add_argument("--td-max-delta", type=int, default=None)
    p.add_argument("--critic-lr", type=float, default=None)
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
    p.add_argument("--freeze-critic-frac", type=float, default=None,
                   help="freeze critic+teacher after this fraction of actor steps (1.0 = never)")
    p.add_argument("--actor-only", action="store_true",
                   help="train against --init-value as a fixed teacher; do not load "
                        "the TD cache or allocate a critic/optimizer/sampler")
    p.add_argument("--expand-weight", type=float, default=None,
                   help="weight of TD backups along planner rollouts (0 = off)")
    p.add_argument("--replay-chunk", type=int, default=-1,
                   help="which imagined chunk the replay state comes from. -1 = "
                        "end of plan (an rh=5 replan, the historical default); "
                        "1 = one chunk in, which is what rh=1 actually queries.")
    p.add_argument("--warm-a0", type=float, default=0.0,
                   help="probability a REPLAYED sample starts refinement from "
                        "the previous plan's shifted tail instead of zeros -- "
                        "the deploy-time warm start, in training. 0 = off.")
    p.add_argument("--replay-prob", type=float, default=None,
                   help="replan curriculum: fraction of actor-step contexts replaced by "
                        "the previous step's final imagined window (last 3 imagined "
                        "latents, same goal, zero action history — the exact query the "
                        "deployed receding-horizon replan makes). Trains the refiner to "
                        "CONTINUE from its own mid-task states instead of only expert "
                        "cache states. Inputs only — the critic's TD targets stay "
                        "expert-anchored.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-run-name", default="",
                   help="enable W&B training-v1 telemetry under this run name")
    p.add_argument("--wandb-run-id", default="",
                   help="stable W&B run id used with resume=allow")
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
    profile = a.recipe_profile
    wm_name = str(a.wm).lower()
    if profile == "auto":
        if "reacher" in wm_name:
            profile = "reacher-pldm" if "pldm" in wm_name else "reacher-lewm"
        else:
            profile = "generic"
    recipes = {
        "generic": dict(steps=8000, batch=128, pad_context=False, max_delta=10,
                        actor_lr=3e-4, actor_lr_final=None, mean_weight=0.1,
                        amax=2.5, expectile=0.1, expectile_final=None, n_step=25,
                        gamma=1.0, critic_lr=1e-3, critic_lr_final=None,
                        freeze_critic_frac=0.8, expand_weight=0.0, replay_prob=0.0),
        "reacher-lewm": dict(steps=6000, batch=256, pad_context=True, max_delta=12,
                             actor_lr=1e-4, actor_lr_final=1e-5, mean_weight=0.3,
                             amax=2.4, expectile=0.1, expectile_final=0.03,
                             n_step=50, gamma=0.98, critic_lr=1e-3,
                             critic_lr_final=1e-4, freeze_critic_frac=0.5,
                             expand_weight=0.0, replay_prob=0.5),
        "reacher-pldm": dict(steps=6000, batch=256, pad_context=True, max_delta=12,
                             actor_lr=3e-4, actor_lr_final=3e-5, mean_weight=0.1,
                             amax=1.4, expectile=0.1, expectile_final=0.03,
                             n_step=50, gamma=0.98, critic_lr=1e-3,
                             critic_lr_final=1e-4, freeze_critic_frac=0.5,
                             expand_weight=1.0, replay_prob=0.5),
    }
    for key, value in recipes[profile].items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    a.recipe_profile = profile
    print("[recipe] " + profile + " " + " ".join(
        f"{k}={getattr(a, k)}" for k in recipes[profile]), flush=True)
    if a.actor_only:
        if not a.init_value:
            p.error("--actor-only requires --init-value")
        if a.expand_weight != 0:
            p.error("--actor-only requires --expand-weight 0")
    if (a.resume_state or a.save_train_state or a.stop_after) and not a.actor_only:
        p.error("training-state screen/resume is supported only with --actor-only")
    if a.state_every < 0:
        p.error("--state-every must be non-negative")
    if a.state_every and not a.save_train_state:
        p.error("--state-every requires --save-train-state")
    if a.stop_after and not 0 < a.stop_after <= a.steps:
        p.error("--stop-after must be in [1, --steps]")
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
    c = LatentCache.load(a.cache, mmap=True)
    _cache_bytes = c.z.numel() * c.z.element_size()
    _cache_dev = dev if a.cache_device == "cuda" else (
        "cpu" if a.cache_device == "cpu" else
        (dev if _cache_bytes < (8 << 30) else "cpu"))
    z = c.z.to(_cache_dev).float()
    print(f"[cache] actor {_cache_bytes / (1 << 30):.2f} GiB on {_cache_dev}",
          flush=True)
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

    # Preserve the historical RNG call order, but batch all latent indexing
    # into two gathers instead of issuing O(B) tiny GPU indexing kernels.
    def sample(B):
        hist_idx, ah, goal_idx, aref = [], [], [], []
        for _ in range(B):
            e = ep_ids[rng.integers(len(ep_ids))]
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            hist_idx.extend((rows[t - 2], rows[t - 1], rows[t]))
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
                goal_idx.append(r2[rng.integers(len(r2))])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                goal_idx.append(rows[min(t + d, L - 1)])
        idx_dev = z.device
        zh = z.index_select(0, torch.as_tensor(hist_idx, device=idx_dev)) \
              .reshape(B, 3, -1).to(dev, non_blocking=True)
        zg = z.index_select(0, torch.as_tensor(goal_idx, device=idx_dev)) \
              .to(dev, non_blocking=True)
        return (zh, torch.from_numpy(np.stack(ah)).to(dev), zg,
                torch.from_numpy(np.stack(aref)).to(dev).float())

    # ------------------------------------------------------------ critic (fs1 cache)
    # Actor-only mode intentionally never opens the dense TD cache and never
    # allocates the trainable critic or Adam states.
    blob = (torch.load(a.init_value, map_location="cpu", weights_only=False)
            if a.init_value else None)
    c_td = None if a.actor_only else LatentCache.load(a.cache_td, mmap=True)
    Ztd = st_td_t = td_sampler = c_opt = critic = None
    _base_dim = int(c.latent_dim if a.actor_only else c_td.latent_dim)
    if blob is not None:
        _ld, _cd = int(blob["latent_dim"]), _base_dim
        assert _cd and _ld % _cd == 0 and _ld // _cd in (1, 3), (
            f"init-value width {_ld} is neither 1x nor 3x the cache dim {_cd}")
        vframes = _ld // _cd
        if vframes == 3:
            _wl = int(blob["arch"].get("window_lag", 5))
            assert _wl == fs, (
                f"window lag {_wl} != action_block {fs}: consecutive imagined "
                "latents are action_block apart, so the deployed window would "
                "not match the trained one")
            print(f"[vframes] 3-frame window value, lag {_wl}", flush=True)
        arch = blob["arch"]
        value_latent_dim = _ld
        if a.actor_only:
            teacher = load_metric(a.init_value, device=dev)
        else:
            critic = load_metric(a.init_value, device=dev)
    else:
        arch = {"head": a.head, "hidden_dim": a.hidden_dim, "depth": a.depth,
                "embed_dim": a.embed_dim, "softplus": True, "symmetric": False}
        vframes = 1
        critic = build_metric("td", c_td.latent_dim, arch).to(dev)
        value_latent_dim = int(c_td.latent_dim)
    if not a.actor_only:
        Ztd, st_td_t = c_td.z, c_td.step_idx
        critic.train()
        teacher = copy.deepcopy(critic).to(dev)
    for prm in teacher.parameters():
        prm.requires_grad_(False)               # actor loss flows THROUGH, never INTO
    teacher.eval()

    # A quasimetric encodes its two endpoints independently.  The goal is
    # constant across all refinement iterations, so encode it once per batch.
    # Skip transparent compressor wrappers: their input transform must remain
    # part of forward(), and the current actor campaigns use the plain head.
    _cache_goal_enc = (hasattr(teacher, "enc")
                       and getattr(teacher, "inner", None) is None)

    def _goal_enc(zg):
        if not _cache_goal_enc:
            return None
        with torch.no_grad():
            return teacher.enc(zg)

    def _vscore(zi, zg, eg=None):
        if eg is None:
            return teacher(zi, zg)
        ei = teacher.enc(zi)
        s = teacher.sym_dim
        ds = (ei[..., :s] - eg[..., :s]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        da = torch.relu(eg[..., s:] - ei[..., s:]).max(dim=-1).values
        return ds + da

    if not a.actor_only:
        td_sampler = NStepGoalSampler(c_td, n_step=a.n_step, p_cross=a.td_p_cross,
                                      n_buckets=10, balanced=True, seed=a.seed,
                                      max_delta=a.td_max_delta)
        c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr,
                                  weight_decay=a.critic_wd)
    else:
        print(f"[actor-only] fixed initial teacher {a.init_value}", flush=True)

    n_plan = a.horizon * fs                     # plan length in primitive steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan), 1.0
    else:
        plan_disc = a.gamma ** n_plan
        plan_cost = (float(n_plan) if a.boundary == "smooth"
                     else (1.0 - plan_disc) / (1.0 - a.gamma))


    def _wstate(traj):
        """Stack the last `vframes` imagined frames."""
        if vframes <= 1:
            return traj[:, -1]
        cols = []
        for k in range(vframes - 1, -1, -1):
            i = traj.shape[1] - 1 - k
            cols.append(traj[:, i] if i >= 0 else traj[:, 0])
        return torch.cat(cols, dim=-1)

    def _wpair(traj, zg):
        """Stack the state window and tile the goal to match."""
        g = zg if vframes <= 1 else zg.repeat(*([1] * (zg.dim() - 1)), vframes)
        return _wstate(traj), g

    def _wsteps(zh_, tr_, zg_):
        """Per-timestep windowed cost over the imagined rollout -> (B, H).

        The window at step t is the last `vframes` frames of
        [history, imagined[:t+1]], matching how it fills at deploy.
        """
        S = torch.cat([zh_, tr_], dim=1)
        hs_, H_ = zh_.shape[1], tr_.shape[1]
        outs = []
        for t in range(H_):
            if vframes <= 1:
                w, g = S[:, hs_ + t], zg_
            else:
                cols = [S[:, hs_ + t - k] for k in range(vframes - 1, -1, -1)]
                w = torch.cat(cols, dim=-1)
                g = zg_.repeat(*([1] * (zg_.dim() - 1)), vframes)
            outs.append(teacher(w, g))
        return torch.stack(outs, dim=1)

    def _wrow(idx):
        """Window-stack TD-cache rows exactly as train_window.py builds them."""
        assert Ztd is not None and st_td_t is not None
        if vframes <= 1:
            return Ztd[idx]
        cols = []
        for k in range(vframes - 1, -1, -1):
            off = k * fs
            j = idx - off
            j = torch.where(st_td_t[idx] < off, idx - st_td_t[idx], j)
            cols.append(Ztd[j])
        return torch.cat(cols, dim=-1)

    def critic_step(expand=None, tau=None, lr=None):
        tau = a.expectile if tau is None else tau
        if lr is not None:
            for pg in c_opt.param_groups:
                pg["lr"] = lr
        b = td_sampler.sample(a.td_batch)
        z_t, z_tn, z_g = b["z_t"].to(dev), b["z_tn"].to(dev), b["z_g"].to(dev)
        ne, reached, dist = b["n_eff"].to(dev), b["reached"].to(dev), b["dist"].to(dev)
        with torch.no_grad():
            d_next = teacher(_wrow(b['tn_idx']).to(dev), _wrow(b['g_idx']).to(dev)) if vframes > 1 else teacher(z_tn, z_g)
            if a.gamma >= 1.0:
                cost, disc = ne, torch.ones_like(ne)
            else:
                disc = a.gamma ** ne
                cost = ne if a.boundary == "smooth" else (1.0 - disc) / (1.0 - a.gamma)
            dist_t = dist
            if a.gamma < 1.0 and a.boundary == "disc":
                dist_t = (1.0 - a.gamma ** dist) / (1.0 - a.gamma)
            tgt = reached * dist_t + (1.0 - reached) * (cost + disc * d_next)
        loss = _expectile_loss(
            (critic(_wrow(b['t_idx']).to(dev), _wrow(b['g_idx']).to(dev))
             if vframes > 1 else critic(z_t, z_g)) - tgt, tau, a.huber_beta)
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
                         use_grad=not a.drop_grad, hidden=a.v4_hidden,
                         n_layers=a.v4_layers,
                         head_scale=a.head_scale, vnorm=a.vnorm,
                         vnorm_k=(1.0 - a.gamma) if (a.gamma or 1.0) < 1.0 else 0.01,
                         squash=a.squash).to(dev)
        print(f"[vnorm] {a.vnorm}", flush=True)
    else:
        net = PlannerNet(z.shape[-1], horizon=a.horizon, a_dim=a_dim, feed=a.feed,
                         amax=a.amax, use_zg=not a.drop_zg,
                         use_gate=not a.no_gate, use_z0=not a.drop_z0,
                         use_grad=not a.drop_grad,
                         head_scale=a.head_scale).to(dev)
    if a.vnorm != "none" and getattr(net, "vnorm", "none") != a.vnorm:
        raise SystemExit(f"--vnorm {a.vnorm} requires --arch v4; "
                         f"{type(net).__name__} carries vnorm="
                         f"{getattr(net, 'vnorm', 'none')!r}")
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=1e-5)

    # "A" carries the plan that PRODUCED each stored state, so --warm-a0 can seed
    # refinement with its tail exactly as the deployed policy does.
    replay_buf = {"zh": None, "zg": None, "A": None}
    start_step = 0
    if a.resume_state:
        state = torch.load(a.resume_state, map_location="cpu", weights_only=False)
        expected = {"steps": a.steps, "seed": a.seed, "actor_lr": a.actor_lr,
                    "actor_lr_final": a.actor_lr_final, "batch": a.batch,
                    "replay_prob": a.replay_prob, "amax": a.amax,
                    "mean_weight": a.mean_weight}
        assert state.get("config") == expected, (state.get("config"), expected)
        start_step = int(state["next_step"])
        assert 0 < start_step < a.steps, start_step
        net.load_state_dict(state["net"])
        a_opt.load_state_dict(state["actor_optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for key, value in state.get("replay_buf", {}).items():
            replay_buf[key] = None if value is None else value.to(dev)
        print(f"[resume] restored full actor training state at step {start_step}",
              flush=True)

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
                _warm_idx, _warm_take = idx, take
        if a.pad_context:
            # 1-frame deployment interface (THE config): only z0 is real,
            # the solver pads the rest and zeroes the action history
            zh = zh[:, -1:].expand_as(zh).contiguous()
            ah = torch.zeros_like(ah)
        z0 = zh[:, -1]
        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)
        if (a.warm_a0 > 0 and replay_buf["A"] is not None
                and "_warm_idx" in dir() and _warm_idx.numel()):
            # Deploy-time construction: after executing c+1 chunks the unexecuted
            # tail A[c+1:] shifts to the front, padded by repeating its last
            # block. c is --replay-chunk, i.e. how far into the plan the stored
            # state sits.
            _c = (a.horizon - 1) if a.replay_chunk < 0 else min(a.replay_chunk,
                                                                a.horizon - 1)
            _keep = torch.rand(_warm_idx.numel(), device=dev) < a.warm_a0
            _sel = _warm_idx[_keep]
            if _sel.numel():
                _src = replay_buf["A"][_warm_take[_keep]]        # (n, H, adim)
                _tail = _src[:, _c + 1:]                          # unexecuted part
                _seed = torch.zeros(_sel.numel(), a.horizon, a_dim, device=dev)
                _k = min(_tail.shape[1], a.horizon)
                if _k > 0:
                    _seed[:, :_k] = _tail[:, :_k]
                    if _k < a.horizon:
                        _seed[:, _k:] = _tail[:, _k - 1:_k]
                    A[_sel] = _seed
        s = net.init_state(a.batch, z0) if a.arch == "v4r" else None
        e_path, zT, tr = [], None, None
        zg_metric = (zg if vframes <= 1 else
                     zg.repeat(*([1] * (zg.dim() - 1)), vframes))
        eg_metric = _goal_enc(zg_metric)

        def _traj_score(tr_):
            """Per-sample score, using deploy-matched value windows at every step."""
            if a.temporal_objective == "terminal":
                return _vscore(_wstate(tr_), zg_metric, eg_metric)
            v0 = _vscore(_wstate(zh), zg_metric, eg_metric)
            if a.temporal_objective == "tel-exact":
                return _vscore(_wstate(tr_), zg_metric, eg_metric) - v0
            vals = _wsteps(zh, tr_, zg)
            prev = torch.cat([v0.reshape(a.batch, 1), vals[:, :-1]], dim=1)
            if a.temporal_objective == "tel-stopprev":
                prev = prev.detach()
            return (vals - prev).sum(dim=1)

        if a.reuse_refinement_rollouts:
            A_roll = A.detach().requires_grad_(True)
            tr_roll = rollout_traj(wm, zh, ah, A_roll)
            score_roll = _traj_score(tr_roll)
        for k in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            if a.reuse_refinement_rollouts:
                (gA,) = torch.autograd.grad(score_roll.sum(), A_roll,
                                            retain_graph=k > 0)
                traj, score_feat = tr_roll, score_roll
            else:
                with torch.enable_grad():
                    A_in = A.detach().requires_grad_(True)
                    traj = rollout_traj(wm, zh, ah, A_in)
                    score_feat = _traj_score(traj)
                    (gA,) = torch.autograd.grad(score_feat.sum(), A_in)
            # A_in holds A's values, so the grad pass's trajectory IS the feature
            # trajectory — reuse it instead of a third WM rollout (value-identical)
            traj_f = traj.detach()
            E_feat = score_feat.detach()
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
            score = _traj_score(tr)
            e_path.append(score.mean())
            if a.reuse_refinement_rollouts:
                A_roll, tr_roll, score_roll = A, tr, score
        if a.replay_prob > 0:
            # --replay-chunk selects HOW FAR INTO THE PLAN the stored state
            # sits: -1 keeps the historical end-of-plan window (an rh=5 replan),
            # while 1 is the state an rh=1 policy actually replans from. Windows
            # clamp at the start of the imagined sequence, as at deploy.
            if a.replay_chunk < 0:
                replay_buf["zh"] = tr[:, -3:].detach()
            else:
                _c = min(a.replay_chunk, tr.shape[1] - 1)
                _lo = max(0, _c - 2)
                _w = tr[:, _lo:_c + 1]
                if _w.shape[1] < 3:                     # clamp by repeating oldest
                    _pad = _w[:, :1].expand(-1, 3 - _w.shape[1], -1)
                    _w = torch.cat([_pad, _w], dim=1)
                replay_buf["zh"] = _w.detach()
            replay_buf["zg"] = zg.detach()
            replay_buf["A"] = A.detach()
        _align = torch.zeros((), device=dev)
        if a.align_weight > 0 and a.align_mode != "terminal":
            _ps = _wsteps(zh, tr, zg)                      # (B, H) per-timestep
            _H = _ps.shape[1]
            if a.align_mode == "dense":
                _w = torch.tensor([a.gamma ** (t + 1) for t in range(_H)],
                                  device=dev, dtype=_ps.dtype)
                _align = (_ps * _w).sum(1).mean() / _w.sum()
            elif a.align_mode == "prefix":
                _align = _ps[:, 0].mean()                  # the block rh=1 executes
            else:                                          # randhorizon
                _h = torch.randint(0, _H, (_ps.shape[0],), device=dev)
                _align = _ps.gather(1, _h[:, None]).squeeze(1).mean()
            _align = a.align_weight * _align
        _stack = torch.stack(e_path)
        if a.lambda_schedule == "uniform":
            loss = e_path[-1] + a.mean_weight * _stack.mean() + _align
        else:
            _K = len(e_path)
            if a.lambda_schedule == "geom-early":
                _w = [a.lambda_base ** k for k in range(_K)]
            else:                                    # geom-late
                _w = [a.lambda_base ** (_K - 1 - k) for k in range(_K)]
            _n = sum(_w) or 1.0                      # normalise so mean_weight keeps its scale
            _wt = torch.tensor([wi / _n for wi in _w], device=_stack.device,
                               dtype=_stack.dtype)
            loss = e_path[-1] + a.mean_weight * (_wt * _stack).sum() + _align
        if a.bc_weight > 0:                       # trust-region toward data actions
            loss = loss + a.bc_weight * ((A - aref) ** 2).mean()
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        if vframes > 1:
            # stack the expansion tuple into the window value's input space:
            # start-state window (pad-context makes this [z0]*hs), terminal
            # window from the rollout, goal tiled to match
            _e0, _eg = _wpair(zh, zg)
            _eT, _ = _wpair(tr, zg)
            expand = (_e0.detach(), _eT.detach(), _eg.detach())
        else:
            expand = (z0.detach(), zT.detach(), zg.detach())
        return e_path[0].item(), e_path[-1].item(), expand

    # ------------------------------------------------------------ schedule
    wb = None
    if a.wandb_run_name:
        import wandb

        wb = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "RLP"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=a.wandb_run_name,
            id=a.wandb_run_id or None,
            resume="allow",
            job_type="actor_train",
            # Log the fully resolved profile, including values supplied by the
            # recipe rather than the CLI.  This is the audit record of what ran.
            config=vars(a),
        )

    pretrain = 0 if a.actor_only else (
        a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000))
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            print(f"pretrain {i}: td_loss {cl:.4f}", flush=True)

    freeze_at = 0 if a.actor_only else int(a.freeze_critic_frac * a.steps)
    expand = None
    stop_step = a.stop_after or a.steps
    assert start_step < stop_step <= a.steps, (start_step, stop_step, a.steps)
    last_heartbeat = 0.0
    last_step_end = time.monotonic()

    def save_train_state(next_step):
        cfg = {"steps": a.steps, "seed": a.seed, "actor_lr": a.actor_lr,
               "actor_lr_final": a.actor_lr_final, "batch": a.batch,
               "replay_prob": a.replay_prob, "amax": a.amax,
               "mean_weight": a.mean_weight}
        payload = {
            "version": 1,
            "next_step": next_step,
            "config": cfg,
            "net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "actor_optimizer": a_opt.state_dict(),
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "replay_buf": {k: (None if v is None else v.detach().cpu())
                           for k, v in replay_buf.items()},
        }
        tmp = f"{a.save_train_state}.tmp.{os.getpid()}"
        os.makedirs(os.path.dirname(os.path.abspath(a.save_train_state)), exist_ok=True)
        torch.save(payload, tmp)
        os.replace(tmp, a.save_train_state)
        print(f"[resume] saved full actor training state at step {next_step} -> "
              f"{a.save_train_state}", flush=True)

    for step in range(start_step, stop_step):
        critic_live = step < freeze_at
        if step == freeze_at:
            print(f"step {step}: critic+teacher frozen for the remaining "
                  f"{a.steps - freeze_at} actor steps", flush=True)
            if not a.actor_only:
                # The saved value is the EMA teacher.  Once frozen, the live
                # critic, Adam moments and dense TD cache have no consumers.
                critic = c_opt = td_sampler = c_td = Ztd = st_td_t = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[critic] released live critic, optimizer and TD cache", flush=True)
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
        step_end = time.monotonic()
        step_time = max(step_end - last_step_end, 1e-9)
        last_step_end = step_end
        if wb is not None and (step == start_step or step_end - last_heartbeat >= 10
                               or step + 1 == stop_step):
            import wandb

            wandb.log({"train/loss_actor": e_final,
                       "train/e_first": e_first,
                       "train/step_time_s": step_time}, step=step + 1)
            wb.summary.update({
                "pantheon_telemetry/contract_version": "training-v1",
                "pantheon_telemetry/heartbeat_at": time.time(),
                "pantheon_telemetry/total_steps": a.steps,
                "pantheon_telemetry/active_step": step + 1,
                "pantheon_telemetry/step_time_s": step_time,
                # This trainer has no reliable model-FLOP accounting. Zero is
                # the honest valid value; the end-to-end step time is real.
                "pantheon_telemetry/effective_mfu": 0.0,
                "pantheon_telemetry/wandb_url": wb.url,
            })
            last_heartbeat = step_end
        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 \
                and (step + 1) < a.steps:
            _snap = f"{a.out}.step{step + 1}.pt"
            kind = ("lip3" if a.arch == "traj" else
                    "lip4r" if a.arch == "v4r" else
                    "lip4" if a.arch == "v4" else
                    ("lip" if a.feed == "none" else "lip2"))
            torch.save({"kind": kind, "feed": a.feed, "sd": {k: v.detach().cpu() for k, v in net.state_dict().items()},
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
                        "vnorm": getattr(net, "vnorm", "none"),
                "vnorm_k": getattr(net, "vnorm_k", 1.0),
                "squash": getattr(net, "squash", "hard"),
                "v4_hidden": a.v4_hidden,
                "v4_layers": a.v4_layers,
                        "value": a.out_value, "train_args": vars(a)}, _snap)

        if (a.save_train_state and a.state_every
                and (step + 1) % a.state_every == 0
                and (step + 1) < stop_step):
            save_train_state(step + 1)

        if step % 500 == 0:
            print(f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                  f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}",
                  flush=True)

    if a.save_train_state and stop_step < a.steps:
        save_train_state(stop_step)
    if stop_step < a.steps:
        print(f"[screen] stopped at step {stop_step}; schedule horizon remains {a.steps}",
              flush=True)

    # ------------------------------------------------------------ save (teacher first;
    # the actor checkpoint references it, and it is what the actor optimized against)
    if a.actor_only:
        if os.path.abspath(a.init_value) != os.path.abspath(a.out_value):
            shutil.copy2(a.init_value, a.out_value)
    else:
        save_metric(teacher.cpu(), "td", value_latent_dim, arch, a.out_value)
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
                "vnorm": getattr(net, "vnorm", "none"),
                "vnorm_k": getattr(net, "vnorm_k", 1.0),
                "squash": getattr(net, "squash", "hard"),
                "v4_hidden": a.v4_hidden,
                "v4_layers": a.v4_layers,
                "temporal_objective": a.temporal_objective,
                "value": a.out_value, "train_args": vars(a)}, a.out)
    print(f"saved learned planner -> {a.out}", flush=True)
    if a.save_train_state and os.path.exists(a.save_train_state):
        os.remove(a.save_train_state)
    if wb is not None:
        wb.summary["train/completed"] = True
        wb.finish()


if __name__ == "__main__":
    main()
