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
import faulthandler
import signal as _hang_signal
import gc
import math
import os
import pathlib
import shutil
import time

# 2026-08-18 silent-hang forensics: six identical stalls (train seeds 0/1/2,
# six distinct nodes) stopped writing at ~step 500 with no error and no exit,
# so nothing ever reached the harness's failure accounting. SIGUSR1 now dumps
# every thread's Python stack to stderr (-> the cell's train log); the harness
# watchdog sends it before killing a stalled trainer, so the next stall leaves
# an autopsy instead of a mystery.
faulthandler.register(_hang_signal.SIGUSR1, all_threads=True)

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
        "auto", "generic", "tworoom-lewm", "tworoom-pldm",
        "ogbench-lewm", "ogbench-pldm",
    ], help="established task/base recipe used for every option not explicitly supplied")
    # data / models
    p.add_argument("--cache", required=True, help="fs5 latent cache (actor rollout contexts)")
    p.add_argument("--cache-td", required=True, help="fs1 latent cache (critic TD pairs; dense stride)")
    p.add_argument("--cache-device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="actor cache placement; auto keeps caches >=8 GiB mmap'd on CPU")
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--init-value", default="", help="warm-start critic from a train_metric.py TD checkpoint")
    p.add_argument("--out", required=True, help="actor checkpoint (train_lip.py format)")
    p.add_argument("--out-value", required=True, help="final teacher checkpoint (save_metric format)")
    # actor (train_lip.py parity)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--steps", type=int, default=None, help="actor steps (profile default)")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-delta", type=int, default=None)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--actor-lr", type=float, default=None)
    p.add_argument("--actor-lr-final", type=float, default=None,
                   help="cosine-decay actor lr to this over all steps (None = constant)")
    p.add_argument("--mean-weight", type=float, default=None)
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
    p.add_argument("--vnorm", choices=["none", "log", "loggn", "scale"], default="none",
                   help="v4: conditioning of the value-derived actor inputs. "
                        "'none' = shipped raw E (steps-to-go at gamma=1) and raw "
                        "grad; 'log' = log1p(E); 'loggn' = log1p(E) + "
                        "RMS-normalised grad_A V (full invariance to affine "
                        "rescaling of V). Saved into the checkpoint and "
                        "reapplied by LIPSolver at deploy.")
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
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="save a DEPLOYABLE snapshot (actor + co-critic teacher) "
                        "every N steps; 0 disables. Enables within-run early "
                        "stopping: select the best snapshot on val draws.")
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
                   help="freeze --init-value from step zero and train only the actor; "
                        "the TD cache, critic optimizer, and critic updates are skipped")
    p.add_argument("--reuse-refinement-rollouts", action="store_true",
                   help="reuse each post-update WM rollout as the next refinement's "
                        "gradient/feature rollout (K+1 rather than 2K rollouts)")
    p.add_argument("--temporal-objective",
                   choices=["terminal", "tel-exact", "tel-stopprev"],
                   default="terminal",
                   help="actor rollout score: terminal V_H; exact telescoping "
                        "sum(V_{h+1}-V_h); or the same forward scalar with each "
                        "negative V_h detached so intermediate gradients do not cancel")
    p.add_argument("--expand-weight", type=float, default=None,
                   help="weight of TD backups along planner rollouts (0 = off)")
    p.add_argument("--warm-state-frac", type=float, default=1.0,
                   help="fraction of the batch whose STATE is advanced one chunk into "
                        "the actor own plan when --train-warm is set. 1.0 (old default) "
                        "makes every query state self-generated, which collapses the "
                        "training distribution off the expert cache and costs accuracy on "
                        "COLD queries (the first solve of every episode, and every solve "
                        "at receding_horizon == horizon). 0.0 keeps the cache state and "
                        "warms only A0, separating the two effects.")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""),
                   help="wandb project; empty disables logging entirely")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    p.add_argument("--wandb-name", default="", help="run name; defaults to the --out stem")
    p.add_argument("--wandb-every", type=int, default=50)
    p.add_argument("--align-mode", default="terminal",
                   choices=["terminal", "dense", "prefix", "randhorizon"],
                   help="which rollout timesteps enter the loss. terminal = "
                        "historical behaviour; randhorizon teaches every prefix to "
                        "land, so no deadline readout is needed at deploy.")
    p.add_argument("--align-weight", type=float, default=0.0,
                   help="weight on the alignment term (0 = off)")
    p.add_argument("--align-from-value", action="store_true",
                   help="score the chunk the CRITIC expects arrival on: "
                        "r = clamp(ceil(V(z0,zg)/action_block), 1, H). No clock, "
                        "nothing learned about stopping.")
    p.add_argument("--train-align", action="store_true",
                   help="score the terminal value at min(horizon, goal_blocks)-1 "
                        "(train-time analogue of solver.align_deadline)")
    p.add_argument("--train-warm", action="store_true",
                   help="refine from the shifted previous plan at the next state, "
                        "matching what the policy hands a warm-started solver at rh<H")
    p.add_argument("--replay-prob", type=float, default=None,
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
    profile = a.recipe_profile
    wm_name = str(a.wm).lower()
    if profile == "auto":
        if "dinowm" in wm_name:
            profile = "generic"
        elif "tworoom" in wm_name:
            profile = "tworoom-pldm" if "pldm" in wm_name else "tworoom-lewm"
        elif "cube" in wm_name or "ogbench" in wm_name:
            profile = "ogbench-pldm" if "pldm" in wm_name else "ogbench-lewm"
        else:
            profile = "generic"
    recipes = {
        "generic": dict(steps=8000, batch=128, max_delta=10, actor_lr=3e-4,
                        actor_lr_final=None, mean_weight=0.1, amax=2.5,
                        expectile=0.1, expectile_final=None, n_step=25,
                        gamma=1.0, critic_lr=1e-3, critic_lr_final=None,
                        freeze_critic_frac=0.8, expand_weight=0.0, replay_prob=0.0),
        "tworoom-lewm": dict(steps=8000, batch=128, max_delta=12, actor_lr=3e-4,
                             actor_lr_final=3e-5, mean_weight=0.1, amax=2.2,
                             expectile=0.1, expectile_final=0.03, n_step=50,
                             gamma=1.0, critic_lr=1e-3, critic_lr_final=1e-4,
                             freeze_critic_frac=0.8, expand_weight=0.0, replay_prob=0.0),
        "tworoom-pldm": dict(steps=8000, batch=128, max_delta=12, actor_lr=3e-4,
                             actor_lr_final=3e-5, mean_weight=0.1, amax=2.8,
                             expectile=0.1, expectile_final=0.03, n_step=50,
                             gamma=1.0, critic_lr=1e-3, critic_lr_final=1e-4,
                             freeze_critic_frac=0.8, expand_weight=0.0, replay_prob=0.0),
        "ogbench-lewm": dict(steps=6000, batch=256, max_delta=10, actor_lr=3e-4,
                             actor_lr_final=3e-5, mean_weight=0.1, amax=1.6,
                             expectile=0.1, expectile_final=0.03, n_step=50,
                             gamma=0.98, critic_lr=1e-3, critic_lr_final=1e-4,
                             freeze_critic_frac=0.5, expand_weight=1.0, replay_prob=0.5),
        "ogbench-pldm": dict(steps=6000, batch=256, max_delta=10, actor_lr=3e-4,
                             actor_lr_final=3e-5, mean_weight=0.1, amax=4.5,
                             expectile=0.1, expectile_final=0.03, n_step=50,
                             gamma=0.98, critic_lr=1e-3, critic_lr_final=1e-4,
                             freeze_critic_frac=0.5, expand_weight=1.0, replay_prob=0.5),
    }
    for key, value in recipes[profile].items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    a.recipe_profile = profile
    print("[recipe] " + profile + " " + " ".join(
        f"{k}={getattr(a, k)}" for k in recipes[profile]), flush=True)
    if a.actor_only and not a.init_value:
        p.error("--actor-only requires --init-value")
    if a.actor_only and a.expand_weight != 0:
        p.error("--actor-only requires --expand-weight 0 (there is no live critic to expand)")
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    rng = np.random.default_rng(a.seed)
    _wb = None
    if a.wandb_project:
        try:
            import wandb as _wandb
            _wb = _wandb.init(
                project=a.wandb_project,
                entity=a.wandb_entity or None,
                name=a.wandb_name or pathlib.Path(a.out).stem,
                config=vars(a),
                reinit=True,
            )
        except Exception as _e:                     # never let logging kill a run
            print(f"[wandb] disabled: {_e}", flush=True)
            _wb = None
    torch.manual_seed(a.seed)

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)

    # ------------------------------------------------------------ actor data (fs5 + h5)
    c = LatentCache.load(a.cache, mmap=True)
    cache_gib = c.z.numel() * c.z.element_size() / 2**30
    cache_dev = a.cache_device
    if cache_dev == "auto":
        cache_dev = "cuda" if dev == "cuda" and cache_gib < 8.0 else "cpu"
    if cache_dev == "cuda" and dev != "cuda":
        cache_dev = "cpu"
    z = c.z.to(cache_dev).float()
    print(f"[cache] actor {cache_gib:.2f} GiB on {cache_dev}", flush=True)
    eps = c.episodes()
    # Sampling below clamps terminal goals to the episode end, so requiring an
    # episode to be longer than max_delta + 4 is both unnecessary and wrong for
    # horizon-matched TwoRoom h100: its episodes contain 20 fs5 blocks and a
    # max_delta of 20. We only need enough rows for the two-history query and
    # the [2, L-2) query range.
    keys = [k for k in eps if len(eps[k]) > 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    if not len(ep_ids):
        lens = [len(rows) for rows in eps.values()]
        raise ValueError(
            "actor cache has no episode with at least 5 fs5 rows; "
            f"episodes={len(lens)}, max_len={max(lens, default=0)}"
        )
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
        hist_rows, ah, goal_rows, aref, gblocks = [], [], [], [], []
        for _ in range(B):
            e = ep_ids[rng.integers(len(ep_ids))]
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            hist_rows.append([int(rows[t - 2]), int(rows[t - 1]), int(rows[t])])
            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
            # trust-region reference: the data's next-H real action blocks from t
            # (clamped at episode end) — where the WM knows the dynamics.
            # Clamp to L-2, the last FULL in-episode block: block L-1 starts at
            # the episode's final primitive step, so its 5-step read spills into
            # the next episode (silently wrong) and runs off the array end for
            # the dataset's last episode (ragged np.stack crash — hit on
            # puzzle 2026-07-23 with 1001-step episodes, e=999/t=197).
            aref.append(np.stack([blocks(e, min(t + k, L - 2)) for k in range(a.horizon)]))
            _cross = rng.random() < a.p_cross
            if _cross:
                e2 = ep_ids[rng.integers(len(ep_ids))]
                r2 = ep_rows[e2]
                goal_rows.append(int(r2[rng.integers(len(r2))]))
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                goal_rows.append(int(rows[min(t + d, L - 1)]))
            # goal distance in ACTION BLOCKS (the fs5 cache is one row per block).
            # Cross-episode goals have no meaningful deadline -> full horizon.
            gblocks.append(a.horizon if _cross else min(d, a.horizon))
        hi = torch.as_tensor(hist_rows, device=z.device, dtype=torch.long)
        gi = torch.as_tensor(goal_rows, device=z.device, dtype=torch.long)
        zh = z.index_select(0, hi.reshape(-1)).view(B, 3, -1).to(dev, non_blocking=True)
        zg = z.index_select(0, gi).to(dev, non_blocking=True)
        return (zh, torch.from_numpy(np.stack(ah)).to(dev), zg,
                torch.from_numpy(np.stack(aref)).to(dev).float(),
                torch.tensor(gblocks, device=dev, dtype=torch.long))

    # ------------------------------------------------------------ critic (fs1 cache)
    c_td = None
    critic = c_opt = td_sampler = None
    if a.actor_only:
        blob = torch.load(a.init_value, map_location="cpu", weights_only=False)
        assert blob["latent_dim"] == c.latent_dim, "init-value latent dim mismatch"
        arch = blob["arch"]
        value_latent_dim = blob["latent_dim"]
        teacher = load_metric(a.init_value, device=dev)
        print(f"[actor-only] fixed initial teacher: {a.init_value}", flush=True)
    else:
        c_td = LatentCache.load(a.cache_td, mmap=True)
        value_latent_dim = c_td.latent_dim
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
        td_sampler = NStepGoalSampler(c_td, n_step=a.n_step, p_cross=a.td_p_cross,
                                      n_buckets=10, balanced=True, seed=a.seed,
                                      max_delta=a.td_max_delta)
        c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr,
                                  weight_decay=a.critic_wd)
    for prm in teacher.parameters():
        prm.requires_grad_(False)               # actor loss flows THROUGH, never INTO
    teacher.eval()

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

    n_plan = a.horizon * fs                     # plan length in primitive steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan), 1.0
    else:
        plan_disc = a.gamma ** n_plan
        plan_cost = (float(n_plan) if a.boundary == "smooth"
                     else (1.0 - plan_disc) / (1.0 - a.gamma))

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
                cost = ne if a.boundary == "smooth" else (1.0 - disc) / (1.0 - a.gamma)
            dist_t = dist
            if a.gamma < 1.0 and a.boundary == "disc":
                dist_t = (1.0 - a.gamma ** dist) / (1.0 - a.gamma)
            tgt = reached * dist_t + (1.0 - reached) * (cost + disc * d_next)
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
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=1e-5)

    replay_buf = {"zh": None, "zg": None}   # previous step's final imagined windows

    def actor_step():
        zh, ah, zg, aref, gbl = sample(a.batch)
        if a.replay_prob > 0 and replay_buf["zh"] is not None:
            pick = torch.rand(a.batch, device=dev) < a.replay_prob
            idx = torch.nonzero(pick).squeeze(1)
            if idx.numel():
                take = torch.randint(0, replay_buf["zh"].shape[0], (idx.numel(),), device=dev)
                zh, ah, zg = zh.clone(), ah.clone(), zg.clone()
                zh[idx] = replay_buf["zh"][take]
                zg[idx] = replay_buf["zg"][take]
                ah[idx] = 0.0        # deployed replans query with zero action history
        # ---- deadline-aligned terminal index (else: always the last chunk)
        _ar = torch.arange(a.batch, device=dev)
        def _term(tr, idx):
            return tr[:, -1] if idx is None else tr[_ar, idx]
        term_idx = (gbl.clamp(1, a.horizon) - 1) if a.train_align else None
        if a.align_from_value:
            # detached: the actor never receives gradient through the choice of r
            with torch.no_grad():
                _v0 = teacher(zh[:, -1], zg).reshape(-1).float()
                _r = torch.ceil(_v0 / float(fs)).clamp(1, a.horizon).long()
            term_idx = _r - 1

        z0 = zh[:, -1]
        eg = _goal_enc(zg)

        def _traj_score(tr_):
            """Per-sample actor score under the selected temporal objective."""
            if a.temporal_objective == "terminal":
                return _vscore(_term(tr_, term_idx), zg, eg)
            v0 = _vscore(z0, zg, eg)
            if a.temporal_objective == "tel-exact":
                # Evaluate the algebraically reduced endpoints. This is exactly
                # the full telescoping sum, without materialising H value-head
                # graphs; it retains the start-state control variate in E.
                return _vscore(_term(tr_, term_idx), zg, eg) - v0
            H_ = tr_.shape[1]
            zg_steps = zg.repeat_interleave(H_, dim=0)
            eg_steps = None if eg is None else eg.repeat_interleave(H_, dim=0)
            vals = _vscore(tr_.reshape(-1, tr_.shape[-1]), zg_steps, eg_steps
                           ).view(a.batch, H_)
            prev = torch.cat([v0.reshape(a.batch, 1), vals[:, :-1]], dim=1)
            if a.temporal_objective == "tel-stopprev":
                prev = prev.detach()
            increments = vals - prev
            if term_idx is not None:
                keep = torch.arange(H_, device=dev)[None] <= term_idx[:, None]
                increments = increments * keep.to(increments.dtype)
            return increments.sum(dim=1)

        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)

        if a.train_warm:
            # Reproduce the deployed replan: solve from zero, commit block 0,
            # advance one block through the WM, then refine the shifted plan at
            # the new state with zero action history (what the policy supplies).
            with torch.no_grad():
                Aw = A
                sw = net.init_state(a.batch, z0) if a.arch == "v4r" else None
                for kw in range(a.iters):
                    with torch.enable_grad():
                        Aw_in = Aw.detach().requires_grad_(True)
                        trw = rollout_traj(wm, zh, ah, Aw_in)
                        (gAw,) = torch.autograd.grad(_traj_score(trw).sum(), Aw_in)
                    trw_f = rollout_traj(wm, zh, ah, Aw)
                    Ew = _traj_score(trw_f)
                    if a.arch == "v4r":
                        Aw, sw = net(Aw, gAw.detach(), Ew, z0, zg, sw, trw_f, k=kw)
                    else:
                        Aw = net(Aw, gAw.detach(), Ew, z0, zg, trw_f, k=kw)
                z_next = rollout_traj(wm, zh, ah, Aw[:, :1])[:, -1]
                # advance the state only for a fraction of the batch; the rest
                # keep their expert-cache state and receive only the warm A0
                _f = float(a.warm_state_frac)
                if _f >= 1.0:
                    _adv = torch.ones(a.batch, dtype=torch.bool, device=dev)
                elif _f <= 0.0:
                    _adv = torch.zeros(a.batch, dtype=torch.bool, device=dev)
                else:
                    _adv = torch.rand(a.batch, device=dev) < _f
                if _adv.any():
                    zh_adv = torch.cat([zh[:, 1:], z_next.unsqueeze(1)], dim=1)
                    zh = torch.where(_adv.view(-1, 1, 1), zh_adv, zh)
                    ah = torch.where(_adv.view(-1, 1, 1), torch.zeros_like(ah), ah)
                A = torch.cat([Aw[:, 1:],
                               torch.zeros(a.batch, 1, a_dim, device=dev)], dim=1).detach()
                z0 = zh[:, -1]
                if term_idx is not None:
                    term_idx = (gbl - 1).clamp(1, a.horizon) - 1

        s = net.init_state(a.batch, z0) if a.arch == "v4r" else None
        e_path, zT, tr = [], None, None
        if a.reuse_refinement_rollouts:
            A_roll = A.detach().requires_grad_(True)
            tr_roll = rollout_traj(wm, zh, ah, A_roll)
            score_roll = _traj_score(tr_roll)
        for k in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            if a.reuse_refinement_rollouts:
                (gA,) = torch.autograd.grad(
                    score_roll.sum(), A_roll, retain_graph=k > 0)
                traj_f = tr_roll.detach()
                E_feat = score_roll.detach()
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
            zT = _term(tr, term_idx)
            score = _traj_score(tr)
            e_path.append(score.mean())
            if a.reuse_refinement_rollouts and k + 1 < a.iters:
                A_roll, tr_roll, score_roll = A, tr, score
        if a.replay_prob > 0:
            replay_buf["zh"] = tr[:, -3:].detach()
            replay_buf["zg"] = zg.detach()
        _align = torch.zeros((), device=dev)
        if a.align_weight > 0 and a.align_mode != "terminal":
            # tr is the final refined rollout: (B, H, D). Single-frame critic, so
            # a per-step cost is a direct read; no window stacking needed.
            _H = tr.shape[1]
            _ps = torch.stack([teacher(tr[:, t], zg) for t in range(_H)], dim=1)
            if a.align_mode == "dense":
                _w = torch.tensor([a.gamma ** (t + 1) for t in range(_H)],
                                  device=dev, dtype=_ps.dtype)
                _align = (_ps * _w).sum(1).mean() / _w.sum()
            elif a.align_mode == "prefix":
                _align = _ps[:, 0].mean()          # the block rh=1 executes
            else:                                  # randhorizon
                _h = torch.randint(0, _H, (_ps.shape[0],), device=dev)
                _align = _ps.gather(1, _h[:, None]).squeeze(1).mean()
            _align = a.align_weight * _align
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean() + _align
        if a.bc_weight > 0:                       # trust-region toward data actions
            loss = loss + a.bc_weight * ((A - aref) ** 2).mean()
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        expand = (z0.detach(), zT.detach(), zg.detach())
        return e_path[0].item(), e_path[-1].item(), expand

    # ------------------------------------------------------------ schedule
    pretrain = 0 if a.actor_only else (
        a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000))
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            print(f"pretrain {i}: td_loss {cl:.4f}", flush=True)

    freeze_at = 0 if a.actor_only else int(a.freeze_critic_frac * a.steps)
    expand = None
    _last_heartbeat = 0.0
    _last_step_end = time.monotonic()
    def _snapshot(step):
        # Deployable mid-training checkpoint: same dict as the final save, but
        # CPU copies so training tensors stay on device. Teacher (EMA co-critic)
        # first — the actor references it.
        import copy as _copy
        root, ext = os.path.splitext(a.out)
        snap_out = f"{root}_step{step}{ext}"
        vroot, vext = os.path.splitext(a.out_value)
        snap_val = f"{vroot}_step{step}{vext}"
        if a.actor_only:
            snap_val = a.init_value
        else:
            save_metric(_copy.deepcopy(teacher).cpu(), "td", value_latent_dim, arch, snap_val)
        _kind = {"traj": "lip3", "v4r": "lip4r", "v4": "lip4"}.get(
            a.arch, "lip" if a.feed == "none" else "lip2")
        sd_cpu = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        torch.save({"kind": _kind, "feed": a.feed, "sd": sd_cpu,
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
                    "vnorm": getattr(net, "vnorm", "none"),
                    "vnorm_k": getattr(net, "vnorm_k", 1.0),
                    "squash": getattr(net, "squash", "hard"),
                    "head_scale": a.head_scale, "pre_ln": a.pre_ln,
                    "temporal_objective": a.temporal_objective,
                    "value": snap_val,
                    "wandb_id": (_wb.id if _wb is not None else None),
                    "wandb_project": a.wandb_project or None,
                    "wandb_entity": a.wandb_entity or None,
                    "train_args": vars(a)}, snap_out)
        print(f"[snapshot] step {step} -> {snap_out}", flush=True)

    for step in range(a.steps):
        critic_live = step < freeze_at
        if step == freeze_at:
            print(f"step {step}: critic+teacher frozen for the remaining "
                  f"{a.steps - freeze_at} actor steps", flush=True)
            if not a.actor_only:
                critic = c_opt = td_sampler = c_td = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
        _step_end = time.monotonic()
        _step_time = max(_step_end - _last_step_end, 1e-9)
        _last_step_end = _step_end
        if (_wb is not None and
                (step == 0 or _step_end - _last_heartbeat >= 10 or step + 1 == a.steps)):
            _wb.log({"train/loss_actor": e_final, "train/E_first": e_first,
                     "train/E_gain": e_first - e_final, "train/td_loss": cl,
                     "train/tau": tau_s, "train/critic_lr": clr_s,
                     "train/actor_lr": alr_s, "train/step_time_s": _step_time},
                    step=step + 1)
            _wb.summary.update({
                "pantheon_telemetry/contract_version": "training-v1",
                "pantheon_telemetry/heartbeat_at": time.time(),
                "pantheon_telemetry/total_steps": a.steps,
                "pantheon_telemetry/active_step": step + 1,
                "pantheon_telemetry/step_time_s": _step_time,
                "pantheon_telemetry/effective_mfu": 0.0,
                "pantheon_telemetry/wandb_url": _wb.get_url(),
            })
            _last_heartbeat = _step_end
        if step % 500 == 0:
            print(f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                  f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}",
                  flush=True)
        if a.ckpt_every and step > 0 and step % a.ckpt_every == 0:
            _snapshot(step)

    # ------------------------------------------------------------ save (teacher first;
    # the actor checkpoint references it, and it is what the actor optimized against)
    if _wb is not None:
        _wb.summary["final_E_final"] = e_final
        _wb.summary["final_E_first"] = e_first
    if a.actor_only:
        if os.path.abspath(a.init_value) != os.path.abspath(a.out_value):
            shutil.copy2(a.init_value, a.out_value)
        print(f"copied fixed teacher value -> {a.out_value}", flush=True)
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
                "vnorm": getattr(net, "vnorm", "none"),
                "vnorm_k": getattr(net, "vnorm_k", 1.0),
                "squash": getattr(net, "squash", "hard"),
                "head_scale": a.head_scale, "pre_ln": a.pre_ln,
                "temporal_objective": a.temporal_objective,
                "value": a.out_value,
                "wandb_id": (_wb.id if _wb is not None else None),
                "wandb_project": a.wandb_project or None,
                "wandb_entity": a.wandb_entity or None,
                "train_args": vars(a)}, a.out)
    print(f"saved learned planner -> {a.out}", flush=True)
    # Managed-job clusters are torn down on completion and are not reachable via
    # `sky status`, so anything left on node-local disk is lost. Persist the actor
    # and its tandem critic (the actor is inert without the paired value) as a
    # wandb artifact. ~1.3 MB + 0.6 MB on twins.
    if _wb is not None:
        try:
            import wandb
            art = wandb.Artifact(pathlib.Path(a.out).stem, type="lip_actor")
            for f in (a.out, a.out_value):
                if f and os.path.exists(f):
                    art.add_file(f)
            _wb.log_artifact(art)
            _wb.finish()
            print(f"[wandb] artifact uploaded: {pathlib.Path(a.out).stem}", flush=True)
        except Exception as e:
            print(f"[wandb] artifact upload FAILED: {e}", flush=True)
            try:
                _wb.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
