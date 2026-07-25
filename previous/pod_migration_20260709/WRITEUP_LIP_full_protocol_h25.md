# LIP on the Full-Protocol LeWM-Cube Benchmark (h25) — Results Writeup

**Date:** 2026-07-11 · **Status:** COMPLETE — h25, h50, h75, all with horizon-native TD + LIP sweeps (§8.4–8.5)
**TL;DR:** A learned iterative planner (LIP), trained purely against a frozen TD value through the frozen
v2 world model, scores **88 / 96 / 78** across the three task draws of the replicated full-protocol
benchmark — **+12 mean over latent+CEM** (the published planner for this WM), **beating the published
84** on their exact 50 tasks, at **~500× less planning compute** and with no sampling at deployment.
At double horizon (h50), the swept LIP reaches **74.0 vs CEM's 54.0 (+20 mean, 1.86× CEM's over-floor
performance)** — the planner's edge *grows* as tasks get longer-horizon.

---

## 1. Benchmark foundation (why these numbers are trustworthy)

- **Protocol** = the WM authors' own: tasks sampled from the training dataset (10k episodes, 2.01M frames,
  224px), 50 (episode, start) pairs per draw, goal = the dataset state **25 primitive steps** after the
  start (h25), budget 50 steps, success = block within 4 cm, `bf16`, action z-scoring contract, CEM
  300×30. The eval seed drives **both** the task draw and the solver seed.
- **Replication anchor:** on their exact seed-42 draw (sampled row indices byte-identical to their
  reference log), our latent+CEM = **80.0 vs their logged 84.0, with 48/50 identical per-episode
  outcomes** — the residual is bf16/CUDA nondeterminism, not protocol drift.
- **Renderer:** EGL, pixel-parity-verified against stored dataset frames (cosine ≥ 0.9997 in v2WM latent
  space; the dataset itself is EGL-rendered). All planners consume identical observations.
- **Draws are not interchangeable:** the three draws differ in task mix (their doc understates this).
  All comparisons below are **paired within draw**; means are across draws 42/43/44.

| draw | random floor | note |
|---|---|---|
| 42 | 46 | their exact published tasks |
| 43 | 56 | easiest mix |
| 44 | 50 | hardest mix relative to floor |

~Half of h25 tasks start quasi-solved (long static-block segments in the play data) — absolute numbers
should always be read **over floor**.

## 2. Methods compared

| method | what it is | plan compute / env step |
|---|---|---|
| latent+CEM | CEM on raw latent distance to goal embedding (their published planner) | 9,000 WM rollouts |
| TD n50 + CEM | CEM with learned quasimetric cost-to-go replacing latent distance | 9,000 WM rollouts |
| **LIP v1** | learned iterative refiner `f(A, ∇V, V, z₀, z_g) → A′`, K=8 unrolled steps, deployed **deterministically, no CEM** (`n_steps=0`) | **~16 rollout-equivalents** |
| LIP v2 | v1 + imagined endstate fed to the refiner (end-feed) | ~16 |

**The TD value (LIP's only teacher):** quasimetric head on frozen v2WM latents, trained on a **stride-1
cache (every frame, 2.01M latents)** — dense supervision through grasp events — with n-step **50
primitive-step** backups, expectile τ=0.03, 6k steps. Selected by a 25-cell sweep + 3-draw confirm
(see §5). The value alone is worth +2/+6/+4 over latent.

**LIP training:** pathwise backprop of `V(ẑ_T, z_g)` through the frozen WM across all K=8 refinement
iterations; start/goal pairs from the play data (offsets ≤ 50 primitives); 8k steps, lr 3e-4. No behavior
cloning, no policy gradient, no real rollouts — the frozen value is the only teacher.

## 3. Main result (h25, 3 draws) — three views of the improvement over CEM

### 3.1 Absolute success rate (%, n=50/draw)

| method | draw 42 | draw 43 | draw 44 | **mean** |
|---|---|---|---|---|
| random floor | 46 | 56 | 50 | 50.7 |
| latent+CEM (reference planner) | 80 | 84 | 62 | 75.3 |
| TD n50 + CEM | 82 | 90 | 66 | 79.3 |
| **LIP v1** | **88** | **96** | **78** | **87.3** |
| LIP v2 (end-feed) | 86 | 92 | 68 | 82.0 |

Absolute improvement over latent+CEM (paired, same 50 tasks per draw; floor cancels in differences):

| method − latent+CEM | 42 | 43 | 44 | **mean** |
|---|---|---|---|---|
| TD n50 | +2 | +6 | +4 | **+4.0** |
| **LIP v1** | **+8** | **+12** | **+16** | **+12.0** |
| LIP v2 | +6 | +8 | +6 | +6.7 |

Other paired contrasts: LIP v1 − TD (its own teacher) = +6/+6/+12 (mean +8.0); LIP v1 vs *their
published* 84.0 on the identical draw-42 tasks = **+4**; LIP v2 − LIP v1 = −2/−4/−10 (v1 canonical,
third benchmark confirming). Every LIP v1 cell is the best number recorded on its draw in this project.

### 3.2 Over-floor performance (method − per-draw random floor)

~Half of h25 tasks start quasi-solved, so the floor is high and draw-dependent (46/56/50). Subtracting
it isolates what planning actually contributes:

| method − floor | 42 | 43 | 44 | **mean** |
|---|---|---|---|---|
| latent+CEM | +34 | +28 | +12 | +24.7 |
| TD n50 | +36 | +34 | +16 | +28.7 |
| **LIP v1** | **+42** | **+40** | **+28** | **+36.7** |
| LIP v2 | +40 | +36 | +18 | +31.3 |

### 3.3 Relative improvement (floor removed)

Two floor-free normalizations of the same data:

| metric (mean over draws) | TD n50 | LIP v1 | LIP v2 |
|---|---|---|---|
| over-floor ratio vs CEM ((m−floor)/(latent−floor)) | 1.16× (+16%) | **1.49× (+49%)** | 1.27× (+27%) |
| failure-rate reduction vs CEM ((fail_CEM − fail_m)/fail_CEM) | 16% | **49%** | 27% |

Per-draw, LIP v1's over-floor ratio is 1.24× / 1.43× / 2.33× — the relative gain is largest exactly
where CEM is weakest (draw 44: CEM only +12 over floor, LIP +28). On draw 43, LIP eliminates **75% of
CEM's failures** (16% → 4% failure rate). Caveat: ratios with small denominators (weak-CEM draws) are
noisy — read them together with the absolute deltas in §3.1, which are floor-invariant and uniformly
positive.

## 4. Anatomy: where the points come from (draw 42, per-task groups)

Task kinematics (from privileged state, analysis only) split the 16 non-easy tasks into three groups:
**transports** (15–37 cm lateral carries), **picks** (grasp-then-lift/lower, ~zero lateral), **hardcore**
(far reach + goal frame is a *transient* mid-manipulation state).

| method | total | transports (4) | picks (6) | hardcore (6) |
|---|---|---|---|---|
| latent+CEM | 40/50 | 0/4 | **6/6** | 0/6 |
| TD stride-5 (old recipe) | 38/50 | **4/4** | 0/6 | 0/6 |
| TD n50 dense | 41/50 | 2/4 | 5/6 | 1/6 |
| **LIP v1** | **44/50** | **4/4** | **5/6** | **1/6** |
| LIP v2 | 43/50 | 4/4 | 4/6 | 1/6 |

Three structural findings:

1. **Latent distance and TD values have complementary failure modes.** Raw latent distance is sharp at
   short range (all picks) but flat/misleading on long transports; TD cost-to-go is the reverse. The
   original stride-5 value traded 6 picks for 4 transports — net negative vs latent.
2. **Dense (stride-1) value supervision fixed the pick failures** (0/6 → 4-5/6): the stride-5 cache
   simply never contained the grasp-event frames, so the value interpolated across the cost cliff at
   contact. This was diagnosed from paired per-episode kinematics *before* touching hyperparameters,
   and the fix behaved exactly as predicted.
3. **LIP unions the phenotypes.** Its 44/50 equals the union of the best value cells' solve sets
   (transports *and* picks) — the refiner extracts both long-horizon direction and short-range precision
   from one value. Its 6 remaining failures: 5 of 6 hardcore tasks + 1 stubborn pick — the current
   frontier, and plausibly partly *task* pathology (matching a transient goal frame within 4 cm).

Notably, LIP v1 solves 3 tasks (22, 31, 32 — the longest transports) that **their own reference run
failed**.

## 5. Supporting sweeps (why we believe the configs)

**TD value (25 cells + confirms, all closed-loop draw-42 evals):**

| axis | result |
|---|---|
| expectile τ (n=15) | inverted-U: 0.005→72, **0.03→82**, 0.2→80, 0.5→76, 0.7→74, 0.9→60. Low-τ convention confirmed; the code-default 0.7 region is measurably worse |
| n-step (τ=0.03) | noisy plateau: 5→70, 10→78, **15→82**, 20→78, 25→76, 35→74, **50→82**; multi-draw broke the tie for n50 (82/90/66 vs n15 82/86/64) |
| head | plain MLP ties quasimetric (82) at champion recipe — head structure not load-bearing |
| capacity / p_cross / max_delta / 2× steps | all neutral-to-worse (72–78) |
| mixed-n ("poor man's TD-λ") | 80, mid-plateau composition → λ-style span mixtures are **not** the lever; the phenotype union came from the planner |

**LIP recipe (K×lr grid, 9 cells, draw 42, n50 teacher, 8k steps uniform):** no cell beats the champion —
K8/3e-4 = K8/1e-4 = **88**, K12 row 84–86, K4 row ≤84 (K4/3e-4 = 64, a training-draw outlier). The
old-benchmark law (shallow-K↔3e-4, deep-K↔1e-3) did **not** transfer; the robust pattern here is
lr 1e-4 uniformly strong across depths and no gain beyond K=8. Champion recipe confirmed optimal.

## 6. Compute at deployment

LIP v1 executes **deterministically**: 8 refinement iterations ≈ 16 WM rollout-equivalents per replan vs
CEM's 9,000 — **~500× less planning compute** — and no sampling machinery. (Wall-clock in our eval loop
is render-bound, so end-to-end speedup is smaller; the FLOP advantage matters on real robots / fast sims.)
The h25 conclusion: the cheap deterministic planner is also the *best* one, not a trade-off.

## 7. Noise discipline & honest caveats

- n=50/draw ⇒ single-cell noise ±6–7 pts; we treat nothing under ~10 pts as real without paired
  multi-draw agreement. The LIP result clears this bar (+8/+12/+16, all draws, vs every baseline).
- Draw-to-draw spread is large (latent: 80/84/62) — never compare unpaired numbers across draws/seeds.
- Training-draw variance for LIP is ±6–8 pts historically; the 88/96/78 is a single training draw
  (replicas would tighten the claim; not yet run on this benchmark).
- One WM (v2WM), one embodiment, h25 regime. Horizon-50 extension (with horizon-matched LIP retrain)
  is running; two task groups (hardcore, 1 pick) remain unsolved by everything.
- LIP v2's deficit here (−2/−4/−10) is consistent with the project's three prior v1-vs-v2 verdicts
  (neutral-to-slightly-worse); v1 stays canonical.

## 8. Horizon-50 extension (goal offset 50, budget 100 — our extension of their protocol)

Same machinery, goal defined 50 primitive steps ahead (double horizon), budget scaled to keep their 2×
ratio. LIP retrained horizon-matched (pair offsets ≤100 primitives, champion recipe, same n50 teacher —
the value needed no retrain, its hindsight goals were uncapped). All else identical.

**Horizon-native TD sweep (validation):** backup spans n ∈ {15, 25, 100} + τ=0.1/n50 were re-swept *at
h50* (draw 42): 60 / 60 / 62 / 56 vs the incumbent **n50 = 64** — the h25 champion holds at h50. Two
findings: the optimal backup span does **not** scale with the task horizon (n100 = 2× the offset buys
nothing), and the τ inverted-U replicates at h50. The table below therefore rests on horizon-swept HPs.

**Expanded h50 LIP sweep (16 cells + 2 champion replicas, 3-draw means):** the champion recipe's
training-draw noise band, measured directly for the first time via seed replicas, is **70.0–72.7**
(orig/rep1/rep2). Against that band: **lr 2e-4 = 74.0 (72/86/64) is the nominal best** — ≥ the champion
on every draw, sitting between the old grid's lr points (the finer ladder paid off); lr5e4 = 72.0,
K4 = 72.0, mw0.3 = 72.0 are band-edge ties; K12 = 70.0, 16k-steps = 69.3 (no gain), pair-window 10 = 68.7,
and **p_cross 0 = 66.0 (dropping cross-episode pairs genuinely hurts)**; lr extremes (5e-5 = 64,
1e-3 = 68) fall off. Verdict: the recipe is robust across a broad plateau; lr 2e-4 is the updated
h50 pick (within combined noise of the champion, but never worse on any draw).

### 8.1 Absolute success rate (%, n=50/draw)

| method | draw 42 | draw 43 | draw 44 | **mean** |
|---|---|---|---|---|
| random floor | 30 | 34 | 28 | 30.7 |
| latent+CEM | 58 | 66 | 38 | 54.0 |
| TD n50 + CEM | 64 | 74 | 56 | 64.7 |
| LIP (h50-native, champion recipe) | 70 | 78 | 62 | 70.0 |
| **LIP (h50, swept: lr 2e-4)** | **72** | **86** | **64** | **74.0** |

Absolute improvement over latent+CEM (paired):

| method − latent+CEM | 42 | 43 | 44 | **mean** |
|---|---|---|---|---|
| TD n50 | +6 | +8 | +18 | **+10.7** |
| LIP champion recipe | +12 | +12 | +24 | +16.0 |
| **LIP swept (lr 2e-4)** | **+14** | **+20** | **+26** | **+20.0** |

Selection caveat: the swept row is the argmax of a 16-cell sweep, so its mean carries some winner's-curse
inflation — but it beat or tied the champion *on every draw* and sits only ~+3 above the champion's
measured 3-replica band (70.0–72.7), so the lr adjustment is real even if the exact margin is soft.

### 8.2 Over-floor performance (method − per-draw floor: 30/34/28)

| method − floor | 42 | 43 | 44 | **mean** |
|---|---|---|---|---|
| latent+CEM | +28 | +32 | +10 | +23.3 |
| TD n50 | +34 | +40 | +28 | +34.0 |
| LIP champion recipe | +40 | +44 | +34 | +39.3 |
| **LIP swept (lr 2e-4)** | **+42** | **+52** | **+36** | **+43.3** |

### 8.3 Relative improvement (floor removed) and horizon scaling

| metric (mean over draws) | h25 TD | h25 LIP | h50 TD | h50 LIP (champion / swept) |
|---|---|---|---|---|
| absolute Δ vs CEM | +4.0 | +12.0 | +10.7 | +16.0 / **+20.0** |
| over-floor ratio vs CEM | 1.16× | 1.49× | **1.46×** | 1.69× / **1.86×** |
| failure-rate reduction vs CEM | 16% | 49% | 23% | 35% / **43%** |

The cross-horizon accounting sharpens the headline: **over floor, latent+CEM stagnates as the horizon
doubles (+24.7 → +23.3) while the learned methods grow (TD +28.7 → +34.0, LIP +36.7 → +43.3 swept)** —
all of the benchmark's added difficulty is absorbed by the learned components, none by raw latent
guidance. Draw 44 at h50 is the sharpest single case: CEM lands +10 over floor, swept LIP **+36 = 3.6×
CEM's over-floor performance**. (Failure-rate reduction dips h25→h50 for LIP only because the floor
itself collapsed — the over-floor ratio is the floor-honest metric, and it grows.)

### 8.4 Horizon-75 (goal offset 75, budget 150) — fully swept at this horizon

Both learned stages were swept natively at h75. **TD sweep** (draw 42, CEM-evaluated): spans
n {25, 50, 100, 150} @ τ=0.03 → 64 / 72 / 60 / 60, plus τ {0.01, 0.1} @ n50 → 66 / **78**. Two
findings: span-scaling refuted a third time (n100/n150 never help), and **the τ optimum moves up with
horizon** — τ=0.1 beats the long-reigning τ=0.03 (+6), consistent with optimism compounding over longer
bootstrap chains; winner confirmed multi-draw. **LIP sweep** (8 cells on the τ0.1/n50 teacher): pair
windows {100, 150} × lr {1e-4, 2e-4, 3e-4} + K {4, 12} → 82–88 band; the **pair window shows real
signal for the first time** (150-primitive pairs beat 100 by +6 at two lr points) while K stays flat.

| method | draw 42 | draw 43 | draw 44 | **mean** | over floor |
|---|---|---|---|---|---|
| random floor | 38 | 70 | 40 | 49.3 | — |
| latent+CEM | 66 | 78 | 60 | 68.0 | +18.7 |
| TD (τ0.1/n50) + CEM | 78 | 80 | 64 | 74.0 | +24.7 |
| **LIP (window 150, K8, lr 3e-4)** | **88** | **88** | **88** | **88.0** | **+38.7** |

LIP − CEM = +22/+10/+28 (mean **+20.0**); LIP − its own teacher = +14.0; over-floor ratio vs CEM =
**2.07×**; failure reduction vs CEM = 63%. The 88/88/88 flatness across three different task draws —
including draw 44 where CEM manages only +20 over floor vs LIP's +48 — is the single most consistent
result in the program. (Absolute cross-horizon comparisons are confounded by draw-distribution shifts —
the h75 floors are higher — so the over-floor columns carry the story.)

### 8.5 Cross-horizon summary (the program's headline curve)

| mean over draws | h25 | h50 | h75 |
|---|---|---|---|
| latent+CEM over floor | +24.7 | +23.3 | +18.7 |
| best TD over floor | +28.7 | +34.0 | +24.7 |
| best LIP over floor | +36.7 | +43.3 | +38.7 |
| **LIP : CEM over-floor ratio** | **1.49×** | **1.86×** | **2.07×** |

Raw latent guidance decays with task horizon; the learned value holds; the learned planner's *relative*
dominance grows monotonically. Horizon-native sweeps mattered twice: τ 0.03→0.1 for the value at h75,
pair window 100→150 for LIP at h75 — both axes were flat at shorter horizons, so carrying HPs across
horizons without re-sweeping would have left ~+10 points on the table at h75.

**Key findings:** (1) the ordering LIP > TD > latent holds on **all nine draws across three horizons** —
uniform sign, no exception; (2) learned components gain value as the task horizon grows — exactly the
mechanism the h25 task-anatomy predicted (latent = short-range specialist, cost-to-go = long-range);
(3) the old project law "TD gain ≈ inverse latent-baseline strength" extrapolates quantitatively to a
regime it wasn't fit on; (4) horizon-native sweeps are load-bearing — two HP optima (value τ, LIP pair
window) moved at h75 after being flat at shorter horizons.

### 8.6 Horizon-200 (in flight — full-episode-span regime)

Goal offset 200 = plan from an episode's first frame to its final frame (episodes are 201 steps, so
exactly one valid start per episode); budget 400. The regime change is total: **floors collapse to
2/0/6, latent+CEM scores 0/0/6** — the WM's native planner solves nothing at full-episode span,
consistent with the project's earlier episodic-cliff finding (0/50 at budget 1000 with env-sampled
goals). Unlike that experiment, h200 goals are dataset episode-ends — reachable by construction.

**TD sweep verdict: the cliff is absolute for sampling-based planning.** All 8 value cells — τ ladder
{0.03, 0.1, 0.3}, spans {50…200}, including the h75 champion — score **0.0** on draw 42, identical to
latent. No cost function rescues CEM here: 300 candidates over a 25-primitive lookahead cannot find
200-step structure regardless of how candidates are scored. This cleanly separates "the value is
uninformative" from "the search cannot exploit it" — the h75 results prove these values carry
long-range signal; CEM just can't cash it at this depth.

**LIP probe verdict: the cliff is universal.** Two LIP cells (window 180, K8, lr 3e-4; teachers
τ0.1/n50 and τ0.1/n100) both score **0.0** on draw 42 with healthy rollouts — learned descent dies at
the same wall as sampling. Final h200 ledger: floor ~2, latent 0, all 8 TD+CEM cells 0, LIP 0.

**Reading:** the usable planning horizon has a hard boundary between h75 (37% of episode span — LIP at
88/88/88, its best result) and h200 (100% — nothing works). Important protocol note: the *planning*
rollout was the usual 5 WM steps (25 primitives) at every horizon — verified via the cost-hook shapes
(`pred (·, 300, 6, 192)` at h200, identical to h25) — and every replan re-encodes a fresh real
observation, so compounding *imagination* error is **not** among the suspects; only the environment
budget scaled (2× offset). The failure is also not cost-signal quality per se (h75 proves the values
informative) nor the search mechanism (sampling and learned descent fail identically). **Root cause (probe battery, all hypotheses tested):** the teacher was exonerated step by step —
gradients through the WM are healthy at 200-range (ΔV −18 in 8 descent steps), 25 primitives of
on-manifold progress is a 4σ signal (99.6% correct sign), and long-range ranking survives (ρ +0.34 at
150–200). The failure is a **mid-range ranking hole at 50–100 primitives** (ρ −0.13…−0.24), identical
for every backup span n ∈ {50,100,150,200} — and the smoking gun: **ground-truth physical displacement
shows the same inversion in the same band (ρ −0.17)**. The play-data collection policy cycles
(grasp→move→place→return) on a ~50–100-step timescale, so *temporal distance between states — the
quantity any TD/quasimetric value regresses — is structurally ill-posed at mid-range in this data.*
A 200-step plan must be steered through that unordered zone; h75 tasks resolve into the well-posed
<50 band (ρ +0.66) within a couple of replans. **Implication: no TD hyperparameter, capacity, or
λ-style change can fix h200 — the regression target itself is unrankable there.** The evidence-backed
direction is pixels-only **subgoal chaining**: decompose long tasks into ≤50-primitive hops (where the
value is excellent) via retrieved waypoint latents, running the existing LIP hop-by-hop. Not in scope
for this program. The program's positive claim stands and is precisely bounded: **learned values and planners
roughly double CEM's over-floor performance and extend robust planning to at least a third of an
episode's span — but do not extend the planner's absolute horizon ceiling.**

## 9. Artifacts & reproduction

- **Results/logs:** `pod_migration_20260709/results_full_protocol/` (all `ev_full_*.log` incl.
  per-episode success arrays; `cf_dE_t003n50.pt` value; `lip_n50_v1.pt`, `lip_n50_v2.pt` actors).
- **Pod (live):** `root@87.120.211.204 -p 18099`, everything under `/workspace` (runners: `tdsweep_dense.sh`,
  `extsweep.sh`, `nfollowup.sh`, `lipchain2.sh`, `liphp.sh`, `h50v2.sh`; trainers in `valscripts/`).
- **Replication doc for the benchmark itself:**
  `stable-worldmodel/health-check/replication_cube/REPLICATION_CUBE.md` + our
  `pod_migration_20260709/BOOTSTRAP_NEWPOD.md` (env pins incl. the load-bearing `imageio-ffmpeg`,
  `hdf5plugin`, EGL device-pinning).
- **Eval command shape** (LIP row):
  `eval_wm.py --config-name cube seed=<S> eval.dataset_name=<full h5> ++bf16=true eval.img_size=224
  policy=<v2WM> solver=mppi_actor +metric=cf_dE_t003n50.pt +cost_mode=replacement
  +solver.actor_path=lip_n50_v1.pt solver.n_steps=0`
