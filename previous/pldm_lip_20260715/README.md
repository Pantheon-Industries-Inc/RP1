# LIPv4(min0) + AC tandem on the PLDM OGBench-cube checkpoint (2026-07-15)

**Question:** does the LIP-AC recipe (min0/LIPv4 actor + tandem TD critic) transfer to a
different frozen world model — PLDM (Sobal et al. 2025, JEPA/VICReg-style, trained from
scratch by the LeWM authors as a baseline) — and how does it compare against base
PLDM + CEM planning (the WM's own latent-MSE cost) on the same benchmark?

Benchmark: lewm-cube full protocol (50 tasks/draw, draws = eval seeds 42/43/44,
h25/h50 primitive-step goals, success = block within 4cm, 224px, bf16 eval).

## Checkpoint provenance & conversion (important)

`stable-worldmodel/checkpoints/PLDM_OgBench/` (received 2026-07-15): ViT-tiny/14@224,
192-d, predictor depth 6 (num_frames 3), action 25-d (5 blocks × 5), projector+pred_proj
MLP-2048-BN — architecturally identical module-for-module to LeWM v2WM (wm/pldm/module.py
is byte-identical to wm/lewm/module.py; only the top-level wrapper class differs).

The checkpoint's encoder was saved in the OLD HuggingFace ViT key layout
(`encoder.layer.N.attention.attention.query`...). Our env (transformers 5.13) builds the
refactored layout (`layers.N.attention.q_proj`...). Converted with
`convert_pldm_encoder.py` (pure key-rename bijection, 198 keys) and validated:

- from-scratch reference forward of the old ViT semantics (LN eps 1e-12, exact gelu)
  vs converted-weights new implementation: max abs diff 1.1e-5 (rel 2.8e-6) ✓
- `LeWM.encode ≡ PLDM.encode` on identical weights: bit-exact ✓
- full CEM cost path (encode → 5-step AR rollout → MSE-to-goal), PLDM class vs LeWM
  class: rel diff ~2e-6 ✓

⇒ all experiments run the weights through the LeWM wrapper
(`checkpoints/PLDM_OgBench_lewm/`, on-pod `/workspace/ckpts/PLDM_OgBench_lewm`) which is
the bf16-capable, grad-efficient implementation the whole harness was validated on.

## Setup

Pod: 213.181.105.210:14755 (4×H100). Bootstrap: code = local repo rsync (supersets pod A);
dataset = HF quentinll/lewm-cube (102GB h5); v2WM + champion actor + cf_dE_t003n50 from
pod A (31.24.80.32:15419) for harness anchors.

Everything driven by `/workspace/pipeline_pldm.sh` (idempotent, summary.csv-cached):
caches (PLDM-encoded fs1 2.01M + fs5 410k) → TD warm-starts (t003n50 primary, t01n50
diagnostic; 6k steps, quasimetric head) ∥ anchors → round-1 sweep → selection evals +
baselines.

Harness anchors (must reproduce or nothing else counts):
- champion schedamax + v2WM, s43 h25 → expect 96.0
- v2WM TD+CEM (cf_dE_t003n50), s42 h25 → expect 82.0

## Round-1 sweep (8 arms, all min0 = --drop-z0 --drop-zg --no-gate, K8/8k/n50, warm from cf_pldm_t003n50)

Center = "schedamax" (v2WM champion recipe): τ 0.1→0.03, clr 1e-3→1e-4, alr 3e-4→3e-5, amax 3.5.

| arm | delta vs center | why |
|---|---|---|
| smx_s0 / smx_s1 | none (2 seeds) | center + training-seed variance |
| amax25_s0 | amax 2.5 | amax is env/WM-specific (TwoRoom: 3.5 hurt −16) |
| mw03_s0 | mean-weight 0.3 | standing untried >90 candidate (S2 monotone on v3) |
| flat_s0 | no schedules, τ0.03 static, amax 2.5 | the pre-schedamax "warm" recipe |
| alr1e4_s0 | actor-lr 1e-4→1e-5 | lr sensitivity on new WM |
| k12_s0 | K=12 | refinement depth (saturated on v2WM; new WM may differ) |
| t02_s0 | τ 0.2→0.03 | stronger early smoothing |

Selection: s42+s44 h25 sum (lessons: ±4 noise, treat +2 as tie, compare seed-paired
means; NEVER select on E_final — it anti-correlates on min0).

Baseline (the comparison target): PLDM+CEM (solver=cem, plain latent MSE cost),
3 draws × h25/h50. Diagnostics: TD+CEM with cf_pldm_{t003,t01}n50 on s42 h25
(v2WM references: latent+CEM 80/84/62 h25; TD+CEM 82/90/66; LIP 88/96/78).

Then: round 2 composes round-1 winners + seed replicas; winner confirmed on 3 draws
before any claim (2-draw deltas <8 pts are noise — paid for twice).

## Failure autopsy (2026-07-16, per-episode analysis of all 50 h25/h50 evals)

Method: episode_successes arrays parsed from every eval log; task features (block Δxy,
Δz, goal height) from the h5 via the drawn row indices printed in the logs; cross-
referenced against the v2WM autopsy's core-task lists (same eval seeds ⇒ same tasks).
Script: valscripts/analyze_failures.py (pod).

1. **v2WM's 12 core grasp-with-airborne-goal tasks fail at ~100% on PLDM too** — every
   planner (LIP arms, PLDM+CEM, TD+CEM). The counterfeit-grasp mode is expert-only-data
   driven, WM-independent. (One exception: TD+CEM solved s42 core task 11.)
2. **PLDM adds ~14 "PLDM-HARD" tasks (≥75% LIP fail) in two families:**
   - **long lateral transports** (Δxy 0.15–0.41m, mostly flat goals): s42 {31,24,46,1,0},
     s43 {20,44,34}, s44 {44,7}. v2WM handled these.
   - **extra airborne lifts** beyond the core: s44 {29,21,46,49} (goal 12–27cm up) —
     PLDM's version of the counterfeit-grasp net is wider.
3. **Fat marginal band** (25–62% fail; ~15 tasks/draw, mostly place-down precision,
   Δz −0.15…−0.27 → table): this band is where the variance lives. Seed-pair symdiff
   = 4–12 tasks (~±9 pts/cell); k12's two seeds are the most consistent (symdiff 6/4).
   On v2WM this band was ~2.7 episodes/100.
4. **Smoking gun on transports: TD+CEM (same WM, same value family) SOLVES the s42
   long-transport set {31,24,0,46,1} that ALL LIP arms fail.** The value ranks these
   fine; the K-step refinement from A(0)=0 can't reach the plans. Actor-side search
   deficit, not WM/value — the opposite attribution from the airborne-core (WM
   optimism, planner-independent).
5. Sum: PLDM's −12 vs v2WM ≈ transports (actor-searchable, ~4-5 tasks/draw ≈ −6 pts)
   + wider lift net (WM data problem, ~−3) + fatter precision noise (~−3).

## Mode-2 fix design: value-guided initialization ("LIPv4-VI")

Mechanism hypothesis: min0's input is [A, ∇_A V, E] only. At A(0)=0 on a long-transport
task, small plans don't move the block → imagined terminal ≈ current state → E flat →
∇_A V ≈ 0. The actor's input is then task-degenerate (≈ [0, 0, E]): it cannot know which
direction the block must go — direction can only enter through the gradient, and the
gradient is uninformative exactly on this task family. CEM escapes via sampling variance;
K-step refinement cannot cross a flat basin. Predicts the observed near-deterministic
transport failures (not noise), and why TD+CEM (same value) solves them.

Fix (stays pure LIPv4 — actor arch/input unchanged): **A(0) = argmin-E over a raw
candidate set** {zero, N/2 iid Gaussian, N/2 time-tiled Gaussian (sustained-motion
plans — the transport family)}, clamped to amax; then the standard K-step refinement.
This is CEM's iteration-0, run once, feeding the learned refiner. Direction information
enters through A(0), still carried solely by the value — "the value is the only teacher"
is preserved.

Distinction from the harmful v2WM restarts: restarts = small noise AROUND ZERO + argmin
over REFINED outputs (actor-exploitation pathology + off-distribution inits). Here:
full-scale candidates from the action prior, selection on RAW plans (pure value ranking,
= what CEM provably does well here), single deterministic refinement, and (phase 2)
training-matched init distribution (--init-prob mixing zeros and value-selected inits).

Phases: (1) eval-time probe of solver-side VI on the existing k12 actors (zero retrain;
+solver.init_mode=value +solver.init_samples=64 +solver.init_scale=1.5) — decisive on the
mechanism via the s42 transport set {31,24,0,46,1}; (2) if it flips, train-side matching
in train_lip_ac.py + 2-seed arms, 3-draw confirm, deploy-cost ~64+K rollouts/replan
(still ~100× under CEM).

## Mode-2 mechanism (REVISED after video autopsy) + fix

Value-init probe NEGATIVE (no transport flips; broad marginal breakage — actors are
zero-init-specialized, and no raw 25-primitive sample grasps+carries anyway). Videos +
block-tracking on k12_s0 s42 transports {0,24,31,46}: **grasp and carry initiation are
FINE; block motion collapses exactly at the step-25 replan boundary** (plan 2 drops/
freezes; task 31 recovers but runs out of budget). Mechanism: plan 2 is queried from the
agent's own mid-carry state — off the expert manifold that actor+critic trained on
(cache states only; plus deploy always passes ZERO action history while training uses
expert histories). Short tasks hide this (plan 1 ≈完成); long-distance transports expose
it. CEM tolerates off-manifold queries (coarse ranking); the gradient refiner does not.

Fix ("replan curriculum" / DAgger-in-imagination, --replay-prob in train_lip_ac.py):
with prob r, an actor-step context is replaced by the PREVIOUS step's final imagined
window (traj[-3:], same goal, ah=0) — exactly the query the deployed replan makes.
Inputs only; critic TD targets stay expert-anchored (no optimism feedback loop).
Round 4 queued: k12 recipe × replay 0.5/0.3 × 2 seeds.

TD n-step sweep (user question, answered): TD+CEM s42 h25: n25=70, **n50=78**, n100=60.
Inverted-U, n=50 optimal on PLDM (v2WM: n-inert in this band). Tandem critic n=50 stands.

## Status log

- 2026-07-15 ~08:20 pod time: pipeline launched, gated on dataset download.
- ~09:40: dataset complete (102GB), but tar's final chown failed on the network FS →
  extract.done never written → pipeline hit its 4h wait limit. Marker written manually,
  relaunched 15:32.
- 15:38: caches built+verified (fs1 2.01M / fs5 410k, 5.5 min on 4 GPUs). TD warm-starts
  trained (cf_pldm_t003n50, cf_pldm_t01n50).
- **ANCHORS BOTH EXACT: champion+v2WM s43 lip = 96.0 (expect 96), v2WM TD+CEM s42 = 82.0
  (expect 82).** (Champion anchor needed lip_ac90_schedamax_value.pt uploaded — LIPSolver
  loads the teacher value via the absolute path stored in ck["value"].) Harness fully
  validated on this pod. Eval wall-clock ≈ 5–6 min/cell — much cheaper than feared.
- 15:43: round-1 batch 1 training (smx_s0/smx_s1/amax25_s0/mw03_s0, flags verified via
  /proc cmdlines). E_final @4k ≈ 8.5–10.2 (higher than v2WM's reference curve — PLDM
  latent scale differs; not a selection signal anyway).
- 21:45 ROUND 1 selection (s42+s44 h25 sums): **flat 144 (72/72), k12 142 (68/74)**,
  amax25 138, smx_s1 138, mw03 134, t02 130, smx_s0 128, alr1e4 124. Center seed spread
  10 pts (128/138). Reads: schedules buy nothing on PLDM (flat ≥ schedamax — the
  schedamax composition was v2WM-specific), amax 2.5 ≥ 3.5, K12 competitive, low
  actor-lr hurts. All top arms > baseline on both cells.
- **PLDM+CEM baseline h25: 64/78/50 (s42/43/44), mean 64.0** — vs v2WM latent+CEM
  80/84/62 (75.3): PLDM is a ~11-pt weaker CEM-planning substrate on this benchmark.
- **PLDM+CEM baseline h50: 48/56/32, mean 45.3** (v2WM latent+CEM h50 was 40.0 — PLDM
  slightly better at h50 native cost, worse at h25).
- **TD+CEM diagnostics s42 h25: t003n50 = 78, t01n50 = 72** vs native cost 64. The
  learned TD value repairs +14 of PLDM's cost signal (on v2WM the same gap was +2:
  82 TD vs 80 latent). t003n50 confirmed as the warm start.
- 21:55: round 1 complete; round 2 auto-started (flat_s1/s2, flatk12_s0/s1/s2, k12_s1,
  flat35_s0, flatt01_s0 — first 4 verified on live cmdlines).
- 2026-07-16 04:14 ROUND 2 selection sums: flat replicas 128/128 (flat mean → 133.3:
  round-1's 144 was seed luck), flatk12 132/132/138 (mean 134), **k12_s1 150 (k12 mean
  146 on 2 seeds)**, **flat35_s0 152 (best single)**, flatt01 140. Reads: a ~133 recipe
  plateau with seed sd ~7-10; k12 (schedamax+K12) and flat35 (static τ, amax3.5) stand
  above it; K12 without amax3.5+something = plateau (flatk12 134). t01 warm not better.
- 04:20: round 3 launched: k12 confirm cells first (s43 h25 + h50×3 for k12_s0/s1),
  then flat35_s1/s2/s3, flat35k12_s0/s1, k12_s2 trainings + selection.
- 04:34 **k12 FULL 3-DRAW CONFIRM (both seeds)**:
  | | s42 | s43 | s44 | mean | baseline | Δ |
  |---|---|---|---|---|---|---|
  | h25 s0 | 68 | 88 | 74 | 76.7 | 64.0 | +12.7 |
  | h25 s1 | 72 | 78 | 78 | 76.0 | 64.0 | +12.0 |
  | h50 s0 | 52 | 62 | 42 | 52.0 | 45.3 | +6.7 |
  | h50 s1 | 48 | 66 | 44 | 52.7 | 45.3 | +7.4 |
  No cell below its baseline; biggest gains on the hard draw s44 h25 (+24/+28). Seed
  consistency excellent (76.7/76.0, 52.0/52.7). k12 = schedamax schedules + K12 + amax3.5,
  min0 no-gate, warm t003n50.
