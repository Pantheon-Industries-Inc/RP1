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

## 0. CORRECTIONS 2026-07-28 (later the same day) — metric provenance, and a fifth harness defect

**§2c and §2-"Still unresolved" are factually wrong about ownership.**
`ReacherQPosMatchTask`, its `get_termination`, and `_DEFAULT_QPOS_THRESHOLD = 0.05`
are **the authors' own released code** — added by LeWM co-author Quentin Le Lidec,
commit `2096f6a1` (PR #158, 2026-03-11, galilai-group/stable-worldmodel),
byte-identical at the paper-time commit `44c45bd1`. Neither the task nor the
0.05 rad tolerance is ours. What IS ours is the no-early-termination stub +
held-at-end scoring (the §2c "fix"). The units-copy theory (0.05 =
`_BIG_TARGET` metres) may still explain *why they picked it*, but it is their
number; if 0.05 rad needs a citation, cite their code, not their paper (neither
paper states any tolerance).

**The authors' published success convention IS latched.** Their `world.py` does
`episode_successes |= terminateds` every step, and their task terminates the
episode at first all-joints match (checked *inside* the `action_repeat=2` loop).
So "swing-through scores" is not a bug we found in our harness — it is the
published metric. §3's "the published numbers plausibly use the generous
convention" is upgraded from plausible to **verified in code**. Our latched
Latent+CEM h25 hitting 86.0 vs their 86 is the expected agreement.

**DECISION (user, 2026-07-28): we keep HELD-AT-END as the reported metric** —
a deliberate, documented departure that is stricter than the published
convention (it measures arrive-AND-stay, which latching cannot see). For paper
comparisons use the latched number, which every run already prints as
`ever-in-ball` in the `[success-convention]` line. Crosswalk on the current
cards (6-cell, strict protocol otherwise unchanged):

| arm | lejepa latched (held) | pldm latched (held) |
|---|---|---|
| Latent + CEM | 93.7 (79.3) | 89.0 (71.0) |
| TD + CEM | 94.0 (68.3) | 92.7 (67.7) |
| LIP v4, 3 actor-seed pool | 93.7 (75.3) | 92.6 (68.0) |
| ORACLE terminal cost | 95.0 (73.0) | 92.7 (71.7) |
| at-rest lag5 | 96.3 (30.7) | 95.7 (24.3) |

Under the authors' metric everything sits at 89–96 (vs published 86/78) and the
planner/value ordering disappears entirely — the at-rest value, catastrophic
under held-at-end, is nominally *best*. Every value-design conclusion in this
campaign is a statement about held-at-end only.

**FIFTH HARNESS DEFECT (found + fixed 07-28): the eval was nondeterministic.**
`_evaluate_from_dataset` reset with `seed=init_state.get('seed')`; the h5 has no
`seed` column ⇒ `seed=None` ⇒ each sub-env drew its **target-ball position**
(salient red geom, mean 0.20 m from the dataset episode's ball, arm reach
0.24 m) from an unseeded RNG on every invocation. The goal image meanwhile shows
the *dataset's* ball. Measured: the same command at the same cfg.seed returned
held-at-end **68 / 70 / 76 / 80 / 82** across five runs (task draw proven
identical; outcomes flipped). Note the authors' harness has the same hole, so
their 86 carries the same per-run jitter. Fixed in `world.py`: when the dataset
provides no seed, derive one per env from the drawn `(episode, start)` pair —
scene randomness becomes a deterministic property of the *task*, distribution
unchanged. Verified: identical numbers across repeated invocations. Consequence:
**all pre-fix cells carry ±≈6 invisible run-to-run jitter on top of task-draw
noise** — do not interpret ≤6-point single-card deltas from before this fix.
Backup of the pre-fix file: `/workspace/_bak_world_predetseed.py`.

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

**c) Strict success.** ~~`ReacherQPosMatchTask.get_termination` (OUR addition)~~
**CORRECTED 07-28 (later): the termination task, the 0.05 rad threshold, AND the
latched convention are the AUTHORS' OWN.** `ReacherQPosMatchTask` with
`_DEFAULT_QPOS_THRESHOLD = 0.05` was added to stable-worldmodel by a LeWM
co-author (commit `2096f6a1`, PR #158), is byte-identical at the paper-time
commit `44c45bd1`, and their `world.py` scores `episode_successes |= terminateds`
per step — i.e. **the published 86/78/79 mean "all joints within 0.05 rad at ANY
step in the budget"**. Neither paper states a threshold; the citation for 0.05
rad is their code. Our held-at-end scoring is therefore a **deliberate metric
deviation**, not a bug fix — keep it (it measures settling, which latching
cannot), but never compare it to the paper. The env publishes per-step
`qpos_in_ball` / `qpos_maxdiff`; every run reports BOTH conventions
(`[success-convention]`: held = reported `success_rate`, ever-in-ball = the
authors' latched metric) plus `[threshold-sweep]` at 5 radii. Latching inflates
+6 to +24 (mean ≈ +17) under 3-frame, up to ~+40 under 1-frame.

**e) Deterministic resets.** The eval was **nondeterministic**: `world.py` reset
envs with `seed=None` (the h5 has no seed column), so the reacher's **visible
red target ball** was re-drawn from an unseeded RNG on every invocation (mean
0.20 m from the dataset episode's ball, arm reach 0.24 m). The same command at
the same cfg.seed returned held 68–82 / latched 88–96 over five runs, task draw
proven identical (5/50 episodes flipped). Fixed by `patch_detseed.py`: reset
seed = f(episode, start), so the scene is a deterministic property of the TASK,
distribution unchanged. Verified 74.0 ×3 bit-repeatable. **Single-cell deltas
< ~14 pts from before this fix are within rerun noise — re-read old
single-cell claims accordingly.**

**d) CEM at 10 iterations, not 30.** Paper: *"300 candidate action sequences …
30 iterations in PushT and 10 iterations in the other environments."* Our
`cem.yaml` shipped `n_steps: 30`, so every CEM baseline had 3× the paper's budget.

### ⛔ THE BENCHMARK CONFIG — user directive 2026-07-28, emphatic. Nothing else counts.

**Every headline number from here on is the PAPER-EXACT config:**

- **1-frame conditioning** — no `+plan_config.history_len`, no `solver.use_frame_history`, anywhere. A `[hist-inject]` line in an eval log disqualifies the cell.
- **CEM 300 × 10 iterations** (`solver.n_steps=10`)
- **latched @0.05 rad** (`ever-in-ball` — the authors' metric) as the score; held-at-end recorded as diagnostic only
- **EGL · h25 only** (offset 25 / budget 50) · **eval seeds 42/43/44**, n=50 · `+eval.ep_range=8000:10000` · deterministic resets (detseed)
- horizon 5, receding 5, action_block 5 — do not shorten (user's explicit call)

Reference points at THE config (`summary_anchor10_*`): Latent+CEM **86.7 / 77.3 / 65.3** (lejepa/pldm/dinowmnp); paper Fig. 6: 86 / 78 / 79.

**Every 3-frame number in this document and in `summary_paper_* / summary_claim_* /
fair-matrix` CSVs is a protocol deviation — diagnostic-only, never a headline,
never comparable to the paper.** For LIP this cuts the other way too: actors must
be *trained* for the 1-frame interface (`train_lip_ac.py --pad-context`, added
07-28 — training contexts collapsed to `[z0]×3` + zero action history, exactly
what the solver pads at deploy); the canonical 3-frame-window actors run under a
train/deploy mismatch at THE config.

### RESOLVED 07-28 (later): the tolerance and the convention

~~"The 0.05 rad tolerance is ours alone"~~ — **wrong; it is the authors'**
(`_DEFAULT_QPOS_THRESHOLD = 0.05` in their `custom_tasks/reacher.py`, see §2c).
The units-copy suspicion may still explain *their* choice (it equals dm_control's
`_BIG_TARGET = .05`, a geom radius in metres — and their own task table sets the
*rendered* target to `_SMALL_TARGET = 0.015` m while scoring 0.05 in radians),
but it is their number to defend, not our artifact. Also: the canonical h5's
`success`/`reward` columns are NaN/sentinel garbage (define nothing);
`observation` = `[qpos ‖ target−finger ‖ qvel]` (dm_control stock, verified to
1e-8); a finger-space tolerance equivalent in strictness to 0.05 rad is
**0.0047 m** vs dm_control's 0.015/0.05 m radii, i.e. our joint-space test is
~10× stricter than the env's own physical target.

**CONVENTION (user calls, 07-28): anything compared to the paper uses the
authors' latched 0.05 rad metric; the campaign's own reported `success_rate`
stays held-at-end (the settling diagnostic latching cannot measure).** Both
come out of every run (`[success-convention]` line). The
sensitivity curve still applies to held-at-end: 10–20% @0.015, 26–42% @0.025,
70–88% @0.05, 94–100% @0.10 (3-frame numbers; the metric is steepest exactly at
0.05). Under 1-frame conditioning, held@0.1 ≈ latched@0.05 for the LeWM-family
bases — a useful "holds still, generous radius" reading.

### Paper anchor, paper-exact protocol (07-28: 1 frame, CEM 300×10, latched, egl,
held-out eps, deterministic resets) — `summary_anchor10_*.csv`

| base | latched @0.05 (paper metric) | paper Fig. 6 | held @0.05 | held @0.1 |
|---|---|---|---|---|
| lejepa (= LeWM) | **86.7** (90/84/86) | **86** | 46.7 | 82.7 |
| pldm | **77.3** (82/72/78) | **78** | 34.7 | 78.0 |
| dinowmnp | 65.3 (62/62/72) | 79 | 22.7 | 59.3 |

**LeWM and PLDM reproduce the paper to within a point** under the faithful
protocol — the campaign's harness, conversions, and data are validated
end-to-end. dinowmnp misses by ~14: ours is the **pixels-only** conversion,
while the paper's DINO-WM number is their own re-implementation (LeWM re-trained
it; DINO-WM's own paper says 0.92 on a different dataset/protocol with proprio —
their reacher env is not even in the public DINO-WM repo, so the exact variant
is unrecoverable). The earlier `summary_anchor_*` (86.0 lejepa / 86.0 pldm s42)
ran CEM at 30 iters — 3× the paper's budget; anchor10 supersedes it.

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
