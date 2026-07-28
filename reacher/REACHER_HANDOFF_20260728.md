# REACHER — LIP / value-function campaign · handoff 2026-07-28

**Scope: the DM-Control Reacher project only** (pod `87.120.211.204:19342`).
Not to be confused with `Dyna/HANDOFF_20260728.md`, which is the OGBench
cube-single campaign on a different pod (`157.66.254.11:15788`), or with
`tworoom_matrix_20260727/`. Same date, different projects.

Read this before touching any reacher number. **Every reacher result produced
before 2026-07-28 is invalid or inflated**, for reasons in §2. The short version:
a day spent testing nine mechanistic hypotheses about the *planner* found that
the planner was never the problem — four harness/config defects were.

---

## 1. Pod and layout

`ssh root@87.120.211.204 -p 19342 -i ~/.ssh/id_ed25519`
2× H100 80GB · 64 CPUs · **187G quota, ~32G free — disk is a live constraint**

| path | what |
|---|---|
| `/workspace/swm_cem` | **current code, all fixes.** Use as `PYTHONPATH`. |
| `/workspace/swm_levers`, `swm_domain`, `swm_probe` | older frozen copies from parallel experiments |
| `/workspace/stable-worldmodel` | original; stale, do not use |
| `/workspace/datasets_canon/lewm-reacher/reacher.h5` | canonical authors' dataset, 93G — **do not delete** |
| `/workspace/caches/canon_<base>_fs{1,5}.pt` | latent caches, carry ground-truth `qpos` |
| `/workspace/metrics/`, `/workspace/actors/` | values / LIP actors |
| `/workspace/results/summary_*.csv` | every card, one row per cell — all drivers are idempotent off these |

Bases: `lejepa` (= the paper's LeWM) and `pldm`, both authors' released epoch-10
checkpoints, unmodified. `dinowmnp` is a documented dead end (frozen-DINOv2
flat-token; CEM works, LIP does not; see memory).

---

## 2. THE PROTOCOL — settled, and what changed

### Fixed on 2026-07-28 (all four had been silently distorting results)

**a) `MUJOCO_GL=egl`, never osmesa.** osmesa's software rasteriser differs from
the dataset's renders by 2.58/255 over 81% of pixels; EGL matches to 0.030/255
over 0.8%. That displaced the lejepa latent by 6.41 (~⅔ the distance between
states 5 steps apart) and pushed it 26× off the training manifold. Cost: **+7.3
on TD+CEM, +13.7 on LIP** (lejepa). PLDM barely moved (0.36) — its encoder is
robust, lejepa's is not. osmesa is purged from every script on the pod; each
driver now verifies the backend at startup and aborts rather than falling back.
Pin `MUJOCO_EGL_DEVICE_ID` and run **one EGL-rendering process per GPU** — two
concurrent ones crash with `OpenGL.raw.EGL._errors.EGLError`.

**b) 3-frame conditioning.** `plan_config.history_len` existed but nothing
populated it, so `EnvPool` handed the policy ONE frame which the solver padded
to 3 identical copies with zero action history — while the WM trained on 3 real
frameskip-5 frames. Three identical frames encode a *stationary* arm, so the
model was told the reacher was frozen at every replan. Worth **+12 to +17**.
Confirmed correct by DINO-WM Table 11 (Reacher H=3, frameskip 5; Wall/TwoRoom
H=1, which is why LeWM's appendix says "1 for TwoRoom" and omits reacher).
Now: `+plan_config.history_len=3`; the policy accumulates real frames at
`action_block` spacing and real executed action blocks. LIP additionally needs
`solver.use_frame_history=true`; CEM/MPPI/Adam pick it up automatically via
`get_cost`. Alignment: with H frames and P planned blocks the action tensor is
**T = H + P − 1** (H−1 real executed blocks, then the plan); at H=1 that reduces
to the old behaviour. Logs print `[hist-inject] H=3 T=7` — check for it.

**c) Strict success.** `ReacherQPosMatchTask.get_termination` (OUR addition —
canonical dm_control reacher has *no* termination) ended the episode at first
contact, and `world.py` OR-ed successes across steps. So an arm that **swung
through** the ball scored. Fixed: no early termination, env publishes per-step
`qpos_in_ball` / `qpos_maxdiff`, success = **entire arm (all joints, worst-joint)
inside 0.05 rad AT THE FINAL STEP**. Measured inflation from latching: **+6 to
+24, mean ≈ +17**. Logs print `[success-convention]` and `[threshold-sweep]`.

**d) CEM at 10 iterations, not 30.** Paper: *"300 candidate action sequences …
30 iterations in PushT and 10 iterations in the other environments."* Our
`cem.yaml` shipped `n_steps: 30`, so every CEM baseline had 3× the paper's budget.

### Standing protocol

- eval draws from episodes **8000:10000**; Dyna collection from **0:8000** (`+eval.ep_range`)
- card = {h25 = offset 25 / budget 50, h50 = offset 50 / budget 100} × seeds {42,43,44}, n=50
- **h25 alone is the paper's reacher protocol**; h50 is ours and runs 5–8 higher. Report separately.
- horizon 5, receding 5, action_block 5 — matches the paper; **do not shorten the horizon** (user's explicit call)

### Still unresolved

The **0.05 rad tolerance is ours alone** and its provenance is suspect: it equals
dm_control's `_BIG_TARGET = .05`, which is a geom radius in **metres** for a
finger-to-target test, not an angular tolerance — likely a units copy. Neither
LeWM nor DINO-WM states any tolerance. It also sits on the steepest part of the
sensitivity curve: full-arm held-at-end gives 10–20% at 0.015 rad, 26–42% at
0.025, **70–88% at 0.05**, 94–100% at 0.10. Median worst-joint error is
0.026–0.035 rad. Decide this deliberately.

---

## 3. Current results — STRICT metric, EGL, 3 frames, CEM 10 iters, held-out

| lejepa | h25 | h50 | 6-cell |
|---|---|---|---|
| **Latent + CEM** (the paper's own method) | **79.3** | 79.3 | **79.3** |
| LIP v4 | 70.7 | **80.7** | 75.7 |
| TD + CEM (MRN quasimetric) | 69.3 | 67.3 | 68.3 |
| dwell γ0.98 / γ0.95 | 58.0 / 56.0 | 55.3 / 56.0 | 56.7 / 56.0 |

| pldm | h25 | h50 | 6-cell |
|---|---|---|---|
| **Latent + CEM** | **71.3** | 70.7 | **71.0** |
| LIP v4 | 68.7 | 67.3 | 68.0 |
| TD + CEM | 68.0 | 67.3 | 67.7 |
| dwell γ0.98 / γ0.95 | 54.7 / 53.3 | 42.7 / 46.0 | 48.7 / 49.7 |

Paper reports LeWM 86 / PLDM 78 / DINO-WM 79 (h25).

**Three things to carry forward:**

1. **Fixing the metric REORDERED the planners.** Under latching TD+CEM led
   (96.0/96.0) and Latent+CEM trailed; under held-at-end Latent+CEM wins and
   TD+CEM is last. Mechanism: a quasimetric scores *steps-to-go*, so it rewards
   fast arrivals that overshoot — ideal under latching, wrong under holding.
   **TD+CEM's entire advantage in this campaign was a metric artifact.**
2. **LIP tracks its critic** — within 1.4 of TD+CEM on both bases. So the critic
   is the lever, not the planner. LIP is also the best arm at h50 on lejepa (80.7).
3. **Do not compare to 86/78 directly.** Our *latched* Latent+CEM h25 hit exactly
   **86.0** vs the paper's 86, so the published numbers plausibly use the generous
   convention, making our strict figures stricter than theirs. Separately LeWM
   reports DINO-WM at 79 while DINO-WM's own paper reports 0.92 — the two papers
   aren't on a common scale either. **The relative gap does reproduce: ours
   79.3−71.3 = 8.0 vs paper 86−78 = 8.**

### 3a. Value-function attempts — all four dead, with mechanisms

| value | lejepa 6-cell | pldm 6-cell | why it died |
|---|---|---|---|
| MRN quasimetric (steps-to-go) | 68.3 | 67.7 | hitting time: indifferent to arrival state, rewards overshoot |
| discounted dwell, r=1{in ball} | 56.7 | 48.7 | reward sparsity flattened the range signal (3% dynamic range vs the quasimetric's 20x) |
| at-rest quasimetric, `[z_t, z_t−z_{t−5}]` | **30.7** | **24.3** | motion signal too weak at the deployable lag; adds a noisy 4–5-step offset in the endgame; hackable by zero-action plans |
| *(none — Latent+CEM, terminal MSE)* | **79.3** | **71.0** | **still the best arm** |

**The single most useful measurement of the value line:** a plain 1-frame
quasimetric separates (at-goal-MOVING) from (at-goal-STOPPED) at **exactly
chance — AUC 0.496, gap −0.001**. `LeWM.encode` embeds each frame independently
(one CLS per frame), so **a single-frame latent carries no velocity**; motion
exists only in the predictor's attention. Therefore *no* value on single-frame
latents can express settling, and the quasimetric's indifference to arrival
state is a **representational** limit, not an objective-design flaw. The at-rest
lag-1 control confirms the encoding idea works (AUC 0.89) — but lag 1 is not
deployable (blocks are 5 primitive steps), and at lag 5 the delta is a large
displacement (‖Δz‖≈8.7 vs ‖z‖≈13.8), not a velocity (0.34 sd, AUC 0.61).
**Any future value work needs finer-grained frames or proprio, not another head.**

Also: at-rest has the **highest ever-in-ball of any arm (88–100)** and the lowest
held-at-end — passing-through inflation **+52 to +82** vs TD's +8/+34. The value
built to reward settling produced the worst settling.

---

## 4. In flight

| # | what | notes |
|---|---|---|
| ~~Stage A~~ | **DONE — at-rest value is DEAD, worst tried** | lejepa 36.7/24.7/**30.7**, pldm 26.0/22.7/**24.3** vs quasimetric 68.3/67.7 and Latent+CEM 79.3/71.0. See §3a. |
| **Stage B** | LIPv4 trained against the at-rest value | 6 jobs (3 seeds × 2 bases), 3 per GPU. Queued at user's request regardless of Stage A. Owns `summary_lipatrest_*`, `actors/lip4_atrest_*`. |

**At-rest design** (user's idea, and the one still alive): keep steps-to-go and
the `QuasimetricHead` — so the triangle inequality and long-range stitching
survive — but redefine *arrival* as "at the goal configuration AND at rest".
Value input per side `[z_t, z_{t−lag}]` (2×192), goal side `[z_g, z_g]` — a
duplicated frame implies zero motion. Prerequisite: `LeWM.encode` embeds each
frame independently (one CLS per frame), so a **single-frame latent carries no
velocity** — motion only exists in the predictor's attention. Hence the stacking;
without it "at rest" is inexpressible and the quasimetric's indifference to
arrival state is a *representational* limit, not an objective-design flaw.
`action_block=5` means consecutive imagined latents are already 5 apart, so
`[z_k, z_{k−1}]` from `rollout_traj` *is* the lag-5 stack.

**Stage B's critical failure mode:** LIP backprops *through* the value for
`∇_A V`. If the stacking severs that graph the actor trains on a zero gradient
and produces a plausible checkpoint that means nothing. Verify `requires_grad`
survives before trusting any Stage B number.

---

## 5. Parked, in priority order

1. **3 LIP training seeds on the existing quasimetric.** The 8.6-point gap that
   motivated the whole value redesign is **single-seed**, against a measured
   ~10-point LIP seed spread (76.7 / 83.0 / 86.3). Cheapest way to find out if
   the premise is real. **Do this first.**
2. **Dyna round 2.** 50/50 mix built and valid (3,972,908 rows, effective
   on-policy frac 0.497, from the fresh EGL/3-frame collection). 80/20 mix needs
   rebuilding — its build died mid-write on ENOSPC. Driver `/workspace/run_dyna2.sh`,
   all five stages patched for EGL + 3 frames + canonical + disjoint pools.
   **Disk forces the arms to run sequentially** (32G 50/50 + 18G 80/20 on top of a
   93G h5 in 187G does not fit) — delete each mix after its fine-tune.
3. **Latent+CEM under EGL at the paper's h25** as the true anchor vs 86.
4. **Oracle terminal cost** (true qpos, no learning) on the same protocol — tells
   you whether the WM fidelity ceiling binds before investing further in value
   design. The WM's 25-step open-loop error is **0.110 rad against a 0.05 rad
   tolerance** (only 21% of open-loop predictions land inside the ball), so no
   value may be able to rank plans by whether they'll hold.

---

## 6. Retracted — do not resurrect

- **All divergence numbers, every campaign.** `dyna_harness/ab_divergence.py`
  pairs probe rounds **by position**, but round *t+1* holds only surviving envs,
  so nearly every row compared imagined vs a *different episode's* reached value
  (mismatched 38/38 rows in one case). Joined on the goal latent: **+1.27 lejepa,
  +3.01 pldm** — almost no divergence. The same tool produced the OGBench
  campaign's 20.9 → 36.1; those need re-deriving.
- **The exploitation/optimizer's-curse framing.** Divergence does rise as
  imagined V falls (r=−0.99) but **reality improves in lockstep** (r=+0.96).
  Fabrication in radians is *flat* in search strength (0.096–0.107); imagined
  terminal latents do not go off-manifold (and the *true*-action rollout is the
  farthest off-manifold of all). The causal step is absent.
- **"Success is non-monotonic in optimization strength."** Monotone-then-flat;
  CEM 10/30/100 → 77.3/78.0/76.0. The Adam inverted-U was a single-cell artifact
  and died at 3 seeds.
- **"Reacher is saturated / no longer discriminates planners."** That was the
  latched metric pinning cells at 90–98; strict cells spread 68–79.
- **"LIP beats TD+CEM (94.7 vs 91.3)."** LIP had 3-frame conditioning and CEM did not.
- **"A cross-base optimizer contrast" (V_lip 1.119 lejepa vs 0.264 pldm).**
  Compared PRE-CANON actors. On canonical actors LIP beats CEM's achieved V on
  both bases. And: pre-canon V 0.917 → 76.9 success vs canon V 0.278 → 77.4,
  i.e. **a 3.3× better objective bought 0.5 points** — minimising this value
  harder is nearly irrelevant to succeeding.
- **The rollout-interface asymmetry**, the action-box violation, `plan_scale`
  (monotonically harmful, 86.0→44.0 over ×1.0→×2.8), H2 (uninformative gradient
  — matched across bases to 1–2%), value miscalibration, and training data
  (canonical retrain reproduced everything within noise). All measured, all dead.
- **"Anchor repro 88 ≈ 86"** — h50 compared against the paper's h25.

---

## 7. Bugs found in shared code (fixed; check before writing new heads/hooks)

- **`eval_wm.py` metric hook, frame-stack width inference.** It read
  `next(m.in_features ...)` and set `_m = in_features // D`. Valid only for
  `QuasimetricHead` (`Linear(latent_dim, hidden)`); `PairwiseMetricHead` uses
  `Linear(4*latent_dim, hidden)`, so 768 for a 192-d latent **exactly aliases a
  4-frame stack**. Every `--head mlp` metric would have hit it. Now reads the
  declared contract (`full_dim` / `latent_dim`). Verified `_m=1` under both old
  and new inference for all existing quasimetric blobs ⇒ no past number changed.
  **Lesson: have heads declare their width; never infer a contract from layer shapes.**
- **The same hook called `m(pred, goal)` instead of `m.cost(...)`.** For distance
  heads `cost == forward` so it never mattered — but any higher-is-better value
  would have been **minimised**, i.e. planned away from the goal, silently.
- **`@torch.no_grad()` on `cost()`** in `head.py` and `ContrastiveCritic` — killed
  every TD+Adam run. Removed; `cost()` must stay gradient-safe (GradientSolver
  and LIP both backprop through it).
- **`eval.ep_range` (my addition) initially broke** by narrowing `ep_indices`
  globally, which also builds the row→episode map → `KeyError`. Fixed by masking
  out-of-range rows instead.
- **`solver/gd.py`** moved actions to device only inside the pad branch → every
  second replan died on a cuda/cpu mismatch.
- **`config/solver/mppi.yaml` never existed** — `MPPISolver` shipped unrunnable.
- **`nn.Module` does not delegate attribute lookup to submodules** — a wrapper
  head must set `self.latent_dim` itself.

---

## 8. Ops gotchas that cost real time

- **`pkill -f <pattern>` over ssh kills your own session** if the pattern appears
  anywhere in your command line — *including inside a heredoc*. Hit three times.
  Use a split literal: `P=run_dyn"a2.sh"; pkill -f "$P"`.
- **Overwriting a running bash script corrupts it.** Kill first, then scp.
- **`test -d X || cp -r ...` silently preserves whatever a dead agent left behind** —
  that's how a stale code copy produced a phantom bug.
- **The Dyna collector caches per-call on old logs** (`grep -q "kept="`), so
  round-1 logs made a fresh collection "finish" in 4 seconds and reuse invalid
  data. Move old logs aside.
- **Round-1 Dyna checkpoints share the exact names round 2 writes** — renamed
  `*_R1OLD`. A discovery glob would have carded round-1 WMs as round-2 results.
- **MooseFS delete lag**: `df` still shows 100% after a large `rm`; space returns
  asynchronously (EDQUOT can fire mid-print during reclaim).
- **zsh does not word-split unquoted variables** — `$SSH_CMD` as a command fails.
- Poll loops ≥10 min with `ControlMaster`; hammering sshd once locked us out of
  every pod (`MaxStartups`).

---

## 9. Memory

Durable state lives in `~/.claude/projects/.../memory/reacher-lip-campaign.md`
(plus `egl-always.md`, `no-train-eval-split.md`, `dyna-handoff.md`). The campaign
file leads with the corrections; read those blocks before the older body, which
contains superseded numbers.

**Cross-campaign warning:** the osmesa domain gap and the latched-success metric
are structural, not reacher-specific. Any campaign evaluating on an authors'
dataset while rendering with osmesa — TwoRoom (`tworoom.h5`), cube/Dyna (HF
expert h5) — has the same exposure, and their published-vs-measured gaps may be
these bugs rather than anything scientific.
