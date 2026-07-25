# TwoRoom × min0: minimal-input learned planner (LIP-AC tandem) — perfect card

**Date:** 2026-07-14/15 · **Pod:** 4×H100 (31.24.80.32:15419) · **Status:** COMPLETE.

**Goal.** Bring the *min0* actor — the minimal-input learned iterative planner
`[A, ∇_A V, E]` confirmed champion-equivalent on OGBench cube — to TwoRoom, and
reproduce the perfect 12-cell card (std+hard surface × h25+h50 × eval seeds
42/43/44, n=50/cell) that the full-input sequential LIP achieved on 2026-07-12.
One tandem run per arm replaces the old two-stage pipeline (TD sweep → CEM
selection → LIP sweep → teacher hyper round).

**Headline.** **PERFECT CARDS: `trm_min0_sched` seeds 0 AND 1 score 100.0 on
all 12 cells (600/600 episodes each); seed 2 = 1198/1200.** A tandem-trained
min0 actor, one training run per seed, zero teacher search, cube-champion
schedules unchanged. Family over 3 training seeds: 1200/1200/1198 (worst seed
99.83%). Runner-up min0 arm `ng` = 1196/1200. The full-input tandem control
screens *below* every promoted min0 arm (296 vs 300/300) — the cube
input-sweep finding (dropping raw z0/z_g helps robustness) transfers to
TwoRoom and strengthens from tie to win. min0 tandem is confirmed as the
canonical implementation going forward (user directive 2026-07-14).

---

## 1. Setting

### 1.1 Environment

`swm/TwoRoom-v1` (stable_worldmodel.envs.two_room): 224×224 RGB, two rooms
separated by a vertical wall at x=112 (width 10px), connected by up to 3 doors
(door positions randomized per episode; door_size = half-extent in px). Agent =
Gaussian dot (σ=7px), target = second dot. Action = 2-D velocity direction in
[−1,1]², scaled by MAX_SPEED = 10.5 px/step. Proprio state = agent(2) +
target(2) + door centers (3×2). Episodes ≤ 100 primitive steps. Torch-based
CPU rendering/collision.

### 1.2 World model (frozen throughout)

LeWM, ViT-tiny encoder (192-d, patch 14, 224×224, from scratch) + predictor
over **frameskip-5 action blocks**: one WM step consumes 5 primitive actions
(a_block ∈ R^10 = 2×5) and 3 pooled history frames (num_frames 3).
Checkpoint: `lewm_tworoom` = `weights_epoch_16_partial.pt` (72 MB), identical
file to the 2026-07-10 campaign.

### 1.3 Eval protocol (identical to tworoom_lip_20260710)

Replay evaluation from `tworoom_play.lance`: an episode's frame at row t is the
start (env `_set_state`), the frame at t+offset is the goal (`_set_goal_state`);
success = reach within 16 px. Two horizons: **h25** (offset 25, budget 50) and
**h50** (offset 50, budget 100). Two surfaces: **std** (any start/goal) and
**hard** (`+eval.cross_wall=true`: start and goal on opposite sides of the
wall — requires stitching through a door). Eval seeds 42/43/44 drive the task
draw; n=50 episodes/cell; `solver.batch_size=10`; replan every executed block
(receding_horizon 5 primitive steps). **Card** = all 12 cells = 600 episodes.

LIP evals: `solver=lip solver.actor_path=<ckpt>` — restarts=1, robust_m=0,
n_steps=0 (pure learned planner, no sampling); K and amax deploy from the
checkpoint.

### 1.4 Data (regenerated; draws re-rolled)

Old pod died; dataset regenerated on pod a with the identical recipe:
`collect_play.py --episodes 1000 --num-envs 32 --seed 7` — ExpertPolicy
(action_noise=2.0, action_repeat_prob=0.05), default variations. 1000 episodes
→ 61,349 primitive frames. Caches: **fs1** = per-frame latents (61,349×192,
stride 1 — dense so the critic's n-step targets see every step), **fs5** = one
latent per action block (12,504×192) for actor rollout contexts. Collection →
caches: ~6 min total.

**The regenerated dataset is statistically equivalent but not byte-identical**
(CPU float nondeterminism compounds over trajectories), so the per-cell eval
task draws re-rolled. Anchor evidence (§4.1): the old perfect actor scores
10×100 + 2×98 on the new draws (both 98s deterministic on rerun — real, newly
drawn episodes it genuinely fails, not eval noise); latent+CEM anchors land
within the old cross-seed spread (62.0 vs old {68,54,62}; 70.0 vs old
{56,58,70}). Same difficulty family, different individual tasks — old-vs-new
per-cell values are not 1:1 comparable, but the 12-cell/600-episode bar is
equally (marginally more) demanding.

---

## 2. Method

### 2.1 min0 actor architecture

`PlannerNet` (stable_worldmodel/solver/lip.py), MLP form, with **both raw
latent inputs dropped** (`--drop-z0 --drop-zg`):

```
input  x_k = [ A_k (H·a=50), ∇_A V(z_T(A_k), z_g) (50), E_k = V(z_T(A_k), z_g) (1) ] ∈ R^101
net    Linear(101→512) → ReLU → Linear(512→512) → ReLU → Linear(512→51)
update A_{k+1} = clip( A_k + σ(gate)·dA , ±amax )      [gated; --no-gate: A_k + dA]
```

341,043 parameters. Applied K=8 iterations from A_0 = 0. H=5 action blocks ×
a_dim 10 (= 25 primitive steps lookahead). The state and goal reach the actor
**only** through the teacher's scalar E and gradient field ∇_A V (autograd
through the frozen WM rollout; detached — an input feature, not a training
path). This is the purest learned-optimizer form: the actor learns *how to
descend* the value landscape, not *where the goal is*. On cube this tied the
full-input champion (87.8 vs 88.0, 4 seeds × 3 draws) with lower seed variance;
here it *wins* (§4.2). Input dim 101 vs 485 for full-input.

### 2.2 LIP-AC tandem training (train_lip_ac.py)

DDPG-style actor-critic on top of the frozen WM:

- **Critic** (MRN quasimetric head, §2.3): n-step expectile TD on the **fs1**
  cache. Targets `tgt = reached·dist + (1−reached)·(n_eff + d_teacher(z_{t+n}, z_g))`
  (γ=1: cost = steps-to-go), balanced 10-bucket goal sampler, p_cross 0.3,
  n-step 50 (primitive steps). Expectile loss (asymmetric Huber, β=1), **low τ
  → optimistic toward the min**. AdamW lr 1e-3, wd 1e-4, batch 1024.
- **Teacher** = EMA(critic), τ_ema 0.005. Serves BOTH the critic's bootstrap
  targets AND the actor's E/∇V features — gradients flow *through* it, never
  *into* it.
- **Actor**: unroll K=8 refinements through the frozen WM (rollout on fs5
  history contexts + h5 primitive actions, batch 128, HER-style goals: p_cross
  0.3 cross-episode, else within-episode ≤ max_delta 10 blocks ahead), loss =
  `E_K + 0.1·mean_k(E_k)`, pathwise backprop through the differentiable WM.
  AdamW lr 3e-4, wd 1e-5, grad-clip 10.
- **Warm start**: critic initialized from the old sequential TD winner
  `td_e0.1_n50.pt` (τ0.1/n50 — the teacher that fixed h50 in the old
  campaign) → pretrain auto-skipped (0 steps).
- **Schedule**: critic interleaves 1:1 with actor steps until
  freeze_critic_frac 0.8 (step 6400/8000), then critic+teacher freeze.
  Winner arm adds during-run schedules (§3).

Actor checkpoint is self-contained for eval (use_* flags, amax, K, teacher
path; LIPSolver deploys ck["iters"] and computes ∇V at plan time).

### 2.3 Critic head

MRN quasimetric (Metric Residual Network, Liu et al. 2022):
`d(z_i→z_j) = ||u_i − u_j||₂ + max_k ReLU(v_j − v_i)_k`, shared encoder
192 → 256 (SiLU) → 256 → 128-d embed split 64 sym + 64 asym, softplus scale,
~149k params. Directed, non-negative, triangle inequality — the inductive bias
that lets TD stitch long distances from short transitions. Arch inherited from
the warm-start blob.

### 2.4 What the tandem replaces

Old sequential pipeline: TD sweep (9 configs, CEM-evaluated) → winner
confirmation → LIP sweep (4 configs) → h50 hyper round (teacher τ0.03→τ0.1
fix) → seed retrains ≈ ~15 trainings + ~50 selection evals over two days.
Tandem: **one training run per arm**, no CEM selection anywhere, teacher never
separately swept. The winner here used cube-champion schedules unchanged —
zero TwoRoom-specific teacher tuning.

---

## 3. Campaign design

All arms share the base recipe: `--horizon 5 --iters 8 --steps 8000
--n-step 50 --expectile 0.1 --critic-lr 1e-3 --actor-lr 3e-4
--init-value td_e0.1_n50.pt` + defaults (batch 128, td-batch 1024, max-delta
10, p-cross 0.3, mean-weight 0.1, freeze 0.8, ema-tau 0.005, amax 2.5, γ=1,
train seed 0).

| arm | flags on top of base | tests |
|---|---|---|
| base | --drop-z0 --drop-zg | min0, τ0.1 const (TwoRoom-canonical) |
| ng | + --no-gate | gate-free min0 |
| **sched** | + --critic-lr-final 1e-4 --expectile-final 0.03 --actor-lr-final 3e-5 | cube-champion schedules (τ anneal 0.1→0.03 over live phase, cosine lrs) |
| md12 | + --max-delta 12 | longer HER goals |
| ax35 | + --amax 3.5 | wider action clamp (cube champion value) |
| fullin | (no drop flags) | full-input tandem control |

Protocol: anchors (gate) → screen seed-0 arms on hard_h25_s42 + hard_h50_s42 +
std_h50_s42 → arms at 300/300 get full cards → best **min0** arm (fullin
excluded by directive) → seeds 1,2 → cards.

**Scheduler note (the big operational lesson):** evals are CPU-bound (torch
env rendering). With default torch threading, 4 concurrent evals ran at 20–32
cores *each* with catastrophic small-op oversubscription: 29–55 min/eval. Work
queue with `OMP_NUM_THREADS=18` + 12 concurrent evals: **37 s–3 min/eval —
a ~40× wall-clock speedup**, numerically identical results (killed-run cells
reproduced exactly). Trainings run concurrently on the GPUs (2 packed/GPU
costs only ~10%).

---

## 4. Results

### 4.1 Anchors (harness + data validation)

Old perfect actor `lv1_e01n50_seed1` on regenerated data, all 12 cells:

| | std h25 | std h50 | hard h25 | hard h50 |
|---|---|---|---|---|
| s42 | 100 | 100 | 100 | 100 |
| s43 | **98** | 100 | 100 | 100 |
| s44 | 100 | **98** | 100 | 100 |

Both 98s deterministic on rerun (real episodes in the re-rolled draws, both
later solved by min0 actors → solvable, actor-idiosyncratic). Latent+CEM:
std_h50_s42 = 62.0, hard_h50_s42 = 70.0 — inside the old cross-seed spread.
Gate passed.

### 4.2 Screen (seed 0; hard_h25_s42 / hard_h50_s42 / std_h50_s42)

| arm | hard h25 | hard h50 | std h50 | sum | E_final |
|---|---|---|---|---|---|
| **ng** | 100 | 100 | 100 | **300** | 1.018 |
| **sched** | 100 | 100 | 100 | **300** | 0.816 |
| base | 98 | 100 | 100 | 298 | 1.289 |
| fullin (control) | 96 | 100 | 100 | 296 | 1.087 |
| md12 | 96 | 100 | 98 | 294 | 0.962 |
| ax35 | 96 | 90 | 98 | 284 | 1.339 |

- **Both perfect screens are min0 arms; the full-input control is not** —
  the input-minimalism advantage, a tie on cube, is a win on TwoRoom.
- τ-anneal 0.1→0.03 (`sched`) does NOT hurt h50 in the tandem — unlike the
  frozen-τ0.03 teacher in the sequential pipeline (97.3 hard h50). Annealing a
  *live* critic ≠ training against a frozen sharp one.
- amax 3.5 (cube's champion value) clearly hurts here (−16 vs base at
  hard h50): action-clamp scale does not transfer across envs; keep per-env
  default 2.5.
- E_final remains an indicator only: ng has the best mid-train E (0.655@4k)
  but the imperfect card; ax35's 1.339 does flag the worst arm.

### 4.3 Cards (n=50/cell; every cell listed)

**`trm_min0_sched_s0` — PERFECT CARD 1200/1200:**

| | std h25 | std h50 | hard h25 | hard h50 |
|---|---|---|---|---|
| s42 | 100 | 100 | 100 | 100 |
| s43 | 100 | 100 | 100 | 100 |
| s44 | 100 | 100 | 100 | 100 |

`trm_min0_ng_s0` — 1196/1200: all cells 100 except hard_h25_s44 = 96 (2
episodes; same-cell 100 for sched-s0 and the lv1 anchor → solvable,
idiosyncratic).

**Seed replicas of sched (independent training seeds, same recipe):**

| training seed | card | blemish | E_final |
|---|---|---|---|
| 0 | **1200/1200 PERFECT** | — | 0.816 |
| 1 | **1200/1200 PERFECT** | — | 0.482 |
| 2 | 1198/1200 | hard_h25_s44 = 98 | 0.817 |

Family: 3598/3600 episodes (99.94%); worst seed 99.83%. The only blemish cell
across the whole campaign is hard_h25_s44, and the failing episodes differ per
actor (sched-s2: ep 17; ng-s0: eps 12+26; sched-s0/s1 and the lv1 anchor solve
all 50) — a few borderline episodes in that draw, solvable, actor-idiosyncratic;
no systematically unsolvable task anywhere. Compare v1 sequential seed spread on
the old draws: seed1 1200, seed2 1194 (98s on 3 cells) — the tandem min0 family
is at least as tight.

### 4.4 Comparison to prior lines

Old-campaign numbers (means over s42/43/44) on the *old* draws; this campaign
on re-rolled draws (§1.4) — family-level comparison only:

| planner | std h25 | std h50 | hard h25 | hard h50 | selection cost |
|---|---|---|---|---|---|
| random | ~28 | ~11 | ~13 | ~2 | — |
| latent+CEM (300×30) | 87.3 | 61.3 | 80.7 | 61.3 | — |
| TD+CEM (td_e0.03_n50) | 100 | 100 | 99.3 | 100 | 9-config sweep + CEM evals |
| LIP v1 seed0 (τ0.03 teacher) | 100 | 98.7 | 100 | 97.3 | + 4-config LIP sweep |
| LIP v1 (τ0.1 teacher, best seeds) | 100 | 100 | 100 | 100 | + teacher hyper round + seed retrains |
| **min0 tandem sched s0/s1 (this)** | **100** | **100** | **100** | **100** | **one run, cube recipe as-is** |

### 4.5 Cross-checks

Screen rows vs card rows on the 3 overlapping cells: identical for both
promoted arms (100=100 ×6) — eval deterministic under the queue/thread-cap
scheduler.

---

## 5. Compute

- Trainings (8k actor steps + 1:1 interleaved critic to 6.4k, batch 128/1024):
  ~95 min solo on one H100; 2-packed/GPU ~105 min (+10%). 6 screen arms
  trained concurrently on 4 GPUs in 105 min wall.
- Evals after the thread-cap fix: h25 ≈ 40–60 s, h50 ≈ 1–3 min, latent+CEM
  h50 ≈ 3 min (12 concurrent, OMP_NUM_THREADS=18, 224-core box).
- Full campaign wall-clock (data regen → first perfect card): **~2h50m**
  (22:53 collect start → 01:42 perfect card), of which 105 min = trainings;
  → all three seed cards done at 03:25 (**4h32m total**, including a ~45-min
  scheduler detour before the thread-cap fix).
- One perfect-card LIP eval costs ~16 WM-rollout-equivalents per replan vs
  ~9000 for CEM 300×30 — the deployed planner remains ~450× cheaper than the
  CEM teacher it replaces.

## 6. Lessons

1. **min0 transfers and wins.** On cube, dropping raw z0/z_g tied the champion;
   on TwoRoom both min0 screen arms beat the full-input tandem control. The
   mechanism (removing raw-latent shortcuts forces the actor onto the teacher's
   robust signal) is not cube-specific.
2. **Tandem + schedules beats sequential + teacher surgery.** The exact
   pathology the old campaign hand-fixed (frozen τ0.03 teacher misleads LIP at
   h50) does not arise when the critic anneals τ *while training with* the
   actor. The cube-champion schedule family transferred unmodified.
3. **amax is env-specific** — 3.5 was cube's win, −16 here. Keep 2.5 default
   per env; sweep only if expert action tails demand it.
4. **Gate: keep it on TwoRoom.** ng (no-gate) was the only promoted arm to
   drop episodes at seed 0 (hard_h25_s44 96), and the gated sched family went
   1200/1200/1198. Weak evidence (1 cell), but the gate is free — default on.
5. **OMP thread cap for CPU-env evals: ~40× wall-clock.** torch defaults to
   nthreads=ncores per process; small per-step tensor ops (224×224 grids)
   collapse under 4×224-thread oversubscription. Cap threads and run wide.
   Always verify with a re-run that fast ≠ wrong (cells reproduced exactly).
6. **E_final still never selects** (ng best-E, imperfect card) — consistent
   with cube S1/S7.

## 7. LIPv4: the gate-free minimal planner as the code default (2026-07-15)

User directive after the min0 campaign: make the **no-gate** minimal-input
variant the default and name it **LIPv4**; find the clipping/init settings
that make it hit 100%.

### 7.1 Code

- `PlannerNet` (solver/lip.py) gains `head_scale` (scales the update-head init;
  train-time only). New checkpoint kind **`lip4`** accepted by `LIPSolver`.
- **LIPv4** = `PlannerNet(use_z0=False, use_zg=False, use_gate=False)`:
  input `[A, ∇_A V, E]` (101-d on TwoRoom), pure residual update
  `A_{k+1} = clip(A_k + dA, ±amax)`, head outputs 50 (no gate slot); 340,530
  params.
- `train_lip_ac.py` and `train_lip.py`: `--arch v4` is the **new default**
  (kind `lip4`); legacy full-input gated MLP via `--arch mlp`
  (+ `--drop-z0/--drop-zg/--no-gate` ablations), transformer via `--arch traj`.
  Old checkpoints load unchanged (stored flags win over kind defaults).
- Verified end-to-end on pod a: 30-step smoke ckpt trains/saves/evals through
  the `lip4` solver path; **null control** (untrained near-zero actor) scores
  14.0 ≈ stand-still floor on std_h25_s42, so the path is honest — and the
  30-step actor already scores 100.0 there (with a saturated WM+value,
  TwoRoom's easy cells need very little update-rule learning).
- Local repo synced (solver/lip.py, scripts/plan/train_lip{,_ac}.py).

### 7.2 Sweep rounds: what the gate-free form needs

All rounds: base = the §4.3 winner recipe (tandem warm + schedules) with
`--arch v4`; full 12-cell cards, **plain deploy** (restarts=1). A deploy-time
restart-selection detour (restarts=8 variants) was explored and **rejected by
directive** — at default noise it traded borderline episodes both ways
(fixed s2, flipped a different single episode for s0; robust_m=4 moved the
flip to yet another cell; gentle noise 0.25 did reconcile all seeds, but the
requirement is the pure learned planner). Those rows remain in the CSV
(card_v4a20r8-\*, \*rm4-\*, \*n025-\*) for the record.

Card sums per training seed (plain deploy):

| round | arm | s0 | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|
| A | amax 2.0 | **1200** | **1200** | 1196 | | | |
| A | amax 2.5 | 1198 | 1194 | **1200** | | | |
| A | amax 3.0 | 1188 | — | — | | | |
| A | amax 2.5 + head-scale .01 | 1154 | 1188 | 1196 | | | |
| B | amax 1.8 | **1200** | 1194 | 1196 | | | |
| B | amax 2.2 | **1200** | 1198 | **1200** | **1200** | 1196 | 1196 |
| B | amax 2.0 + mean-weight .3 | 1198 | 1196 | 1196 | | | |
| B | amax 2.0 + 6k steps | 1194 | **1200** | 1198 | | | |
| C | amax 2.2 + p-cross .5 | **1200** | **1200** | 1198 | | | |
| **C** | **amax 2.2 + max-delta 12** | **1200** | **1200** | **1200** | | | |
| C | amax 2.2 + K12 | *(timed out at 3h, packed 3/GPU; moot)* | | | | | |

Findings:

1. **Clipping substitutes for the gate — up to a point.** amax monotone at
   seed 0 in round A (2.0 → 1200, 2.5 → 1198, 3.0 → 1188); round-B
   bracketing put the family sweet spot at 2.2. Without the gate's learned
   damping, a tighter action clamp bounds each refinement step instead.
   Env-specific (cube wanted 3.5 *with* a gate).
2. **Small-init does NOT transfer to the MLP.** head-scale 0.01 — the fix
   that made no-gate work on the v3 transformer — is the worst arm
   (1154/1188/1196, cells down to 88). E_finals look fine (0.39–0.71):
   another train-E/eval dissociation.
3. **The residual failures were wall-traps, not near-misses.** Video
   forensics on the 98-cells: the agent descends the value gradient straight
   into the wall at the goal's mirror position and stays pinned, never
   detouring through the door. Actor-specific (sibling seeds solve the same
   episodes); per-seed perfect-card rate ~40–60% across amax-only variants —
   plain triple-perfect was a coin flip until the mechanism was addressed.
4. **max-delta 12 fixes the wall-traps systematically.** HER goals to 60
   primitive steps put substantially more cross-door routing pairs in the
   actor's training distribution: the family goes 3/3 perfect. Echoes the
   old sequential campaign, where max-delta 12 was an alternate fix for the
   h50 gap. p-cross 0.5 (more cross-episode goals) nearly does it too
   (1200/1200/1198) — same mechanism, weaker dose.

### 7.3 Canonical LIPv4 recipe (TwoRoom) — TRIPLE-PERFECT, plain deploy

```
train_lip_ac.py --arch v4 --amax 2.2 --max-delta 12 \
  --iters 8 --horizon 5 --steps 8000 --n-step 50 \
  --expectile 0.1 --expectile-final 0.03 \
  --critic-lr 1e-3 --critic-lr-final 1e-4 \
  --actor-lr 3e-4 --actor-lr-final 3e-5 \
  --init-value <TD τ0.1/n50 warm start>
```

head-scale 1.0 (default), no gate, no raw latents; deploy = single pass of
the learned planner, restarts=1, no sampling. **All three training seeds card
1200/1200 — 3600/3600 episodes at 100.0** (actors `trm_v4c_md12_s{0,1,2}.pt`).
Code defaults keep amax 2.5 / max-delta 10 (env-neutral); the values above
are the TwoRoom-canonical recipe. Beats the gated sched family
(1200/1200/1198) — the gate is fully dispensable. Latent+CEM and random-floor
full cards on the same protocol: 73.2 and 11.5 overall vs LIPv4's 100.0.
Every per-cell number: `RESULTS_tworoom_scores.md`.

## 8. Artifacts

- **Deliverable actors:** `actors/trm_min0_sched_s0.pt` and `_s1.pt` (both
  perfect cards; + teachers `metrics/trm_min0_sched_s{0,1}_value.pt`) — min0,
  gated, K8, amax 2.5, H5; eval: `solver=lip solver.actor_path=...`
  (self-contained). `_s2.pt` (1198/1200) included for the family claim.
- Local (`tworoom_min0_20260714/`): WRITEUP (this file), results/summary_tworoom.csv
  (every cell), actors/ (all 6 arms + seed replicas), metrics/ (teachers),
  logs/ (driver + per-arm training logs + phase1), code_snapshot/
  (train_lip_ac.py, lip.py, train_metric.py, env.py, drivers).
- Pod: /workspace/{actors,metrics,results,logs} as above; videos per eval in
  /workspace/results/videos_*/; DONE marker /workspace/results/tworoom_min0.DONE.
- LIPv4: actors `trm_v4_{a20,a25,a30,hs}_s*.pt` + teachers; drivers
  `run_tworoom_lip4.sh` + `run_tworoom_lip4_a20ext.sh`; updated
  solver/lip.py + train_lip_ac.py + train_lip.py in code_snapshot/ (synced to
  the local repo).
- Repro: `tworoom_phase1.sh` (data+caches, deterministic seed 7) →
  `run_tworoom_min0_v2.sh` (anchors → screen → cards → seeds; idempotent) →
  `run_tworoom_lip4.sh` (LIPv4 sweep).
