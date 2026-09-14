# PushT diagnostics E1-E3 (2026-09-08)

Setup: config-B PushT actors (`pusht-v2l05-s{0,1,2}-20260902`, w4, K=8,
md20, acr 0.5, n-step 1), official LeWM base, held-out draws 42/43/44
(50 tasks each), one K=8 pass per decision. Conditions evaluated on the SAME
draws: `rlp` (zero init, K=8), `rlp_ceminit` (K=8 from a CEM plan on the
latent cost, 300x30, top-30), `rlp_ceminit_k0` (the CEM plan through the LIP
path, K=0: control), `rlp_k64` (64 iterations of the learned update),
`cem_latent`, `noop`. Code: `LIPSolver(init_mode="cem", iters_override=K)`,
per-iteration probes (`E_iters`, `lat_iters`), rollout recording
(`RECORD=1`), in-job analysis `scripts/pusht_diag/analyze.py`. Jobs
`pusht-diag3-s*-20260908` (full); the numbers below are from the salvaged
first launch (`pusht-diag2-*`, killed by the legacy volume's disk quota):
draw 42 for all three actors, draw 43 for actor 2.

## E1 -- oracle initialisation (all three actors, draws 42/43/44; success %, means over draws)

| actor | zero-init K=8 (`rlp`) | CEM plan, K=0 (control) | **CEM plan + K=8** | CEM (`cem_latent`) | K=64 |
|---|---|---|---|---|---|
| s0 | 68.0 | 80.0 | **68.7** | 79.3 | 38.7 |
| s1 | 64.0 | 79.3 | **68.7** | 78.7 | 9.3 |
| s2 | 64.7 | 79.3 | **69.3** | 79.3 | 30.0 |
| mean | 65.6 | 79.6 | **68.9** | 79.1 | 26.0 |

Paired over the 9 (actor, draw) cells: the K=0 control agrees with
`cem_latent` on 41-47 of 50 episodes per cell (implementation check passed).
Refinement FROM the CEM plan fixes 3-11 of the mode-A episodes per cell but
loses 2-15 episodes CEM had solved (sum over cells: fixes 55, loses 76).

**Verdict: the refiner cannot hold a good plan.** Handed CEM's plan, eight
learned iterations lower the critic energy (E 11 -> 7.5-8.2, below the
zero-init endpoint of 8.7-9.1) while real success drops from ~79 to ~69. The
imagined objective improves, the outcome worsens: the gradient-based update
walks into critic error that CEM's coarse population search never reaches.
Mode A is therefore not "the basin was never found"; it is "the descent
direction is exploitable".

## E2 -- refinement audit (critic energy E_k, t=0 replan, means per class)

| actor | class | E_0 | E_2 | E_4 | E_8 (deployed) | E_64 | imagined final latent distance at k=8 |
|---|---|---|---|---|---|---|---|
| s0 | RLP ok (n=33) | 21.1 | 16.3 | 12.8 | 8.7 | 12.4 | 5.4 |
| s0 | mode A (n=10) | 24.1 | 18.9 | 16.8 | 14.6 | 16.6 | 8.9 |
| s1 | RLP ok (n=30) | 20.0 | 15.5 | 12.1 | 7.8 | 19.0 | 4.7 |
| s1 | mode A (n=11) | 24.6 | 19.2 | 14.6 | 10.1 | 21.7 | 6.3 |
| s2 (2 draws) | RLP ok (n=70) | 21.5 | 16.7 | 12.9 | 9.1 | 15.4 | 5.3 |
| s2 (2 draws) | mode A (n=16) | 23.9 | 18.4 | 14.7 | 11.5 | 18.1 | 7.1 |

- Descent through k=8 is monotone (91-95 % of steps) and NOT converged:
  only 35-46 % of the total drop has happened by k=2 and E_7 -> E_8 still
  falls by ~0.7. K=8 is truncated descent.
- Continuing the learned update past its trained horizon diverges: by k=64
  the energy is back at 12-20 (monotone fraction 0.27-0.38) and success
  collapses to 14-52. The iterate is a K=8-unrolled network, not a fixed-point
  solver; more iterations of it are out of distribution. This is why the
  counterstrike K ladder saturated: K must be trained, not extended.
- Mode-A episodes start harder (E_0 24 vs 21) and end higher (E_8 14.6 /
  10.1 / 11.5 vs 8.7 / 7.8 / 9.1); their imagined final distance is 6-9 vs
  5, i.e. the planner's own scores flag them at t=0 (reproduces the
  counterstrike self-flagging result).

## E3 -- WM-error attribution (actor s0, draws 42/43/44)

Imagined-vs-real latent divergence ||z_imag_t - z_real_t|| over the first plan
(5 blocks x 5 steps), rolling the frozen WM from the real start frames on the
EXECUTED actions of each arm (recorded rollouts) and on the expert's actions
from the same start rows (h5). Means over episodes; block 5 = plan end.

| actions | draw 42 | draw 43 | draw 44 | n per draw |
|---|---|---|---|---|
| expert (dataset) | 1.35 2.06 2.68 3.26 **3.74** | 1.14 1.65 2.19 2.64 **3.13** | 1.29 1.78 2.18 2.89 **3.64** | 50 |
| CEM (latent cost) | 1.45 2.76 4.52 5.55 **6.01** | 1.42 2.12 2.37 4.15 **5.32** | 2.60 3.57 4.22 5.31 **6.40** | 16 / 15 / 17 |
| RLP (K=8) | 1.93 3.85 5.16 5.46 **5.77** | 2.02 2.86 2.99 3.55 **4.30** | 1.93 3.46 4.73 5.61 **6.56** | 25 / 20 / 34 |

Caveat: the recorder drops episodes shorter than 25 steps and successful
episodes terminate early, so the planner rows are biased toward long
(failing) episodes; the expert rows are all 50 tasks.

Reading: the WM is ~1.7x worse off the expert distribution, and **equally so
under CEM's and RLP's actions**. The world model does not single out the
refiner's plans; both planners drive it into the same error band. So the
RLP-vs-CEM gap is not "RLP finds WM holes CEM avoids" -- it is the OBJECTIVE:
RLP descends the learned critic energy E, CEM optimises the raw latent L2
distance, and E1 showed E improving while reality worsens. The exploitable
component is the critic evaluated on imagined states (both planners' imagined
states are equally wrong; only the critic-driven descent turns that error into
a confident bad plan).

## What this points to
1. Do not extend K at deploy; if more descent is wanted it must be trained in
   (K=24 was flat in the counterstrike campaign, so this is not the lever).
2. E3 says the WM error is planner-agnostic, so PushT Dyna would lift CEM
   and RLP alike and is not the gap-closer. The exploitable part is the
   critic on imagined states. Decisive next test (E4, one eval-only job):
   CEM on the critic energy (`core/value=metric`, replacement mode) vs CEM on
   latent L2 on the same draws. If critic-CEM also drops to ~65, the critic
   is the defect (fix: critic training on planner-visited imagined states /
   a conservative critic); if critic-CEM stays ~79, the defect is the
   gradient descent path itself (fix: an anchor toward the latent objective
   or toward CEM/expert plans in the refiner's training, `planner.bc_weight`).
3. In either case the July "value exonerated" reading was for the old
   stack; under config B the objective is where the gap lives.

## Critic experiments launched 2026-09-08 (E4, E5, E6a, E7, E8)

- **E4 -- objective swap for the sampler** (`pusht-e4-s{0,1,2}`): CEM driven by
  the co-trained critic (`cem_value`), CEM driven by the offline TD teacher
  (`cem_tdvalue`), and `cem_latent` on the same draws. Critic-CEM ~65 => the
  critic ranks plans wrongly on its own; ~79 => only the gradient path is at
  fault.
- **E6a -- deploy-time critic swap for the refiner** (same jobs,
  `rlp_tdvalue`): the s0-s2 actors run with the OFFLINE teacher as the deploy
  critic instead of the co-trained one. Measures how much of RLP's outcome is
  the co-trained critic's drift (TwoRoom precedent: teacher swap +0.9).
- **E5 -- real-vs-imagined calibration** (`critic_probe.py`, POST_ONLY on
  `pusht-diag3-s0`): critic energy of the REAL window reached at the end of
  the first plan vs the IMAGINED window on the same executed actions, for
  RLP and CEM rollouts, co-trained and teacher critics; optimism gap by
  outcome; AUC of E_real / E_imag for predicting success. Separates "critic
  wrong on real states" from "critic wrong on imagined states".
- **E8 -- velocity channel** (same job, rlp probes): energy of the imagined
  4-frame window vs a static stack of the terminal frame. A mode-A-specific
  gap says the refiner's advantage is manufactured in the window's velocity
  components.
- **E7 -- epistemic disagreement** (`pusht-teacher-s10/s11`: two more TD
  teachers on the same cache, different seeds; analysis follows): std across
  teachers of the energy at RLP's imagined terminal states, mode-A vs ok.
  High disagreement on mode-A => the exploit sits in epistemic uncertainty;
  a pessimistic ensemble critic is then the fix candidate.

## Results of the critic experiments (2026-09-08, evening)

### E4 -- objective swap for the sampler (3 actors x draws 42/43/44, success %)

| objective for CEM (300x30) | s0 | s1 | s2 | mean |
|---|---|---|---|---|
| latent L2 (`cem_latent`) | 78.7 | 78.7 | 79.3 | **78.9** |
| co-trained critic (`cem_value`) | 72.7 | 73.3 | 68.0 | **71.3** |
| offline TD teacher (`cem_tdvalue`) | 67.3 | 68.7 | 72.0 | **69.3** |
| RLP actor, deploy critic = teacher (`rlp_tdvalue`) | 54.0 | 60.0 | 52.0 | 55.3 (vs 65.6 with the co-trained critic) |

**The critic is a worse planning objective than raw latent distance even for a
sampler** (-7.6 with the co-trained critic, -9.6 with the teacher). So the
defect is in the critic itself, not only in the gradient path through it; the
refiner (E1: 65.6 -> 68.9 from a CEM init) adds a further ~3-7 on top.
Co-training helps: swapping the teacher in at deploy costs the refiner 10
points, so critic drift is not the problem.

### E5 -- real-vs-imagined calibration (actor s0, co-trained critic; teacher in brackets)

| rollouts | AUC(E on REAL end state predicts success) | AUC(E on IMAGINED end state) | optimism gap E_imag - E_real, successes / failures |
|---|---|---|---|
| RLP d42 | 0.68 (0.74) | 0.56 (0.60) | +1.0 / **-2.9** |
| RLP d43 | 0.58 (0.61) | 0.59 (0.61) | +0.2 / +0.1 |
| RLP d44 | 0.70 (0.66) | 0.56 (0.46) | -0.4 / **-3.6** |
| CEM d42 | 0.57 (0.57) | 0.57 (0.57) | -2.6 / -2.0 |
| CEM d43 | 0.88 (0.77) | 0.73 (0.66) | -0.7 / -1.6 |
| CEM d44 | 0.72 (0.73) | 0.82 (0.85) | -0.5 / +1.1 |

Two defects, both on the critic side: (1) even on the REAL states actually
reached, the critic separates success from failure only weakly (AUC
0.6-0.7); (2) on RLP's failing rollouts the imagined end state is scored
2.9-3.6 more optimistically than the real one, while successes show no such
gap -- the refiner's plans land where the critic is optimistic about the
imagined state. (Recorder bias: planner rows over-represent long episodes.)

### E8 -- velocity channel (rlp probes, t=0 plan): NULL
Static-stack minus window energy is +0.3 / +2.6 / +0.8 for mode-A vs
+1.3 / +0.9 / -0.2 for successes (co-trained critic); no consistent
mode-A-specific gap, AUCs unchanged (0.67-0.78 either way). The refiner's
advantage is not manufactured in the window's velocity components.
Teacher-vs-co-trained energies correlate 0.90-0.94 on the same inputs.

### E3 replicated on actors s1 and s2
Plan-end divergence: expert 3.1-3.7; CEM 5.3-6.4; RLP 5.6-6.2 (one outlier
cell, s1 draw 42: 10.2). Planner-agnostic WM error confirmed on all three
actors.

### Where this leaves the critic
- Not the WM (E3), not the window's velocity channel (E8), not critic drift
  from co-training (E4/E6a), not the descent path alone (E4: the sampler
  loses 8 points on the same critic).
- The critic's VALUE TARGET is the suspect: it ranks end states only weakly
  by real success and is optimistic on imagined failing states. The
  heatmaps point at a concrete mechanism: V over agent position has a sharp
  minimum at the expert's goal-frame AGENT position, i.e. the critic behaves
  largely as an agent-position distance. PushT success is a BLOCK-pose
  criterion; a critic that rewards "agent where the expert's agent ended"
  is exactly what a gradient refiner will satisfy without moving the block.
- **E9 (launched): block-vs-agent sensitivity.** Sweep the block pose with
  the agent fixed and the agent with the block fixed for the same tasks;
  compare V's dynamic range and the decomposition
  V(agent@goal, block@start) vs V(agent@start, block@goal) vs both. If V
  moves far more with the agent than with the block, the fix is on the
  value target: block-centric goal conditioning (mask the agent in the goal
  frame, or a privileged block-pose distance target), not more optimisation.

### E9 -- block-vs-agent sensitivity (actor s0, draw 42, first 15 tasks, 24x24 sweeps)

| scorer | V range over AGENT sweep (block fixed) | V range over BLOCK sweep (agent fixed) | V(start) | V(agent at goal, block at start) | V(agent at start, block at goal) | V(goal config) |
|---|---|---|---|---|---|---|
| co-trained critic | 19.7 | 26.1 | 22.9 | **18.6** | 17.8 | 1.3 |
| offline TD teacher | 10.8 | 8.2 | 13.4 | **8.6** | 12.0 | 0.6 |
| latent L2 | 9.1 | 9.1 | | | | |

- Moving ONLY the agent to its goal-frame position drops the co-trained
  critic by more than 10 on 5 of 15 tasks (2, 4, 5, 9, 10); on task 9 (a
  mode-A failure) V falls from 22.9 to 0.6 with the block untouched, on task
  4 from 15.1 to 1.6. The critic declares those tasks essentially solved by
  agent placement alone.
- The offline teacher is worse: agent-only move 13.4 -> 8.6 vs block-only
  13.4 -> 12.0, i.e. the teacher is mostly an agent-position distance.
  Co-training (imagined rollouts, value expansion) partially repaired this
  (block range 26 > agent range 20 on average) but left the agent shortcut
  on a third of the tasks. This ordering matches E4 (teacher-CEM 69.3 <
  co-trained-CEM 71.3 < latent-CEM 78.9): latent L2 weights agent and block
  equally (9.1 vs 9.1).
- Figure: `docs/figures/pusht_diag/e9_agent_vs_block.png`; data
  `block_sensitivity.json`.

**Diagnosis of the critic.** The value target -- temporal distance to a goal
FRAME in a latent space where the agent is the salient moving object --
lets the critic satisfy itself by putting the agent where the expert's agent
ended, with the block wherever it is. A gradient refiner finds that shortcut
(E1: energy down, success down); a sampler on latent L2 does not, because the
block's pixels count as much as the agent's there. This is a specification
defect of the target, not a capacity or optimisation defect.

**Fix candidates (value-target side, untested):**
1. Block-centric goal conditioning: erase/randomise the agent in the goal
   frame (or condition the critic on the block-only latent), so the target
   cannot be reached by agent placement.
2. Privileged-state target: train the critic's temporal distance in
   block-pose space (position + angle, the eval's own success metric), as the
   cube stack does with `privileged_block_0_pos`, and read it through the
   latent at deploy.
3. Agent-position augmentation: for each training pair, re-render the goal
   frame with the agent moved, so the critic learns agent invariance.

### E7 -- epistemic disagreement (two single-frame TD teachers, seeds 10/11, same cache; actor s0's t=0 plans)

| draw | std across teachers: RLP ok / mode A / both fail | mean energy: ok / mode A / both fail | AUC(std separates mode A from ok) | AUC(mean E separates mode A from ok) |
|---|---|---|---|---|
| 42 | 0.21 / 0.19 / 0.35 | 3.0 / 5.7 / 5.0 | 0.48 | 0.73 |
| 43 | 0.18 / 0.18 / 0.46 | 2.6 / 2.3 / 7.9 | 0.61 | 0.52 |
| 44 | 0.17 / 0.33 / 0.37 | 2.9 / 3.9 / 4.0 | 0.67 | 0.64 |

Independently seeded teachers agree to within 0.2-0.4 on energies of 3-8;
mode-A plans are not where they disagree (AUC 0.48-0.67, inconsistent). The
agent-placement shortcut is a SHARED systematic bias of the target, not
epistemic uncertainty, so a pessimistic ensemble critic would not remove it.
Caveat: two members, single-frame (the first teacher batch was trained
without the w4 window flags); a three-member 4-frame ensemble
(`pusht-teacher-w4-s12/s13` + the s0 teacher) is training to confirm.

### E7 confirmed with three 4-frame teachers (s0 teacher + seeds 12/13, identical window and cache)

| draw | std across teachers: RLP ok / mode A / both fail | mean energy: ok / mode A / both fail | AUC(std: mode A vs ok) | AUC(mean E: mode A vs ok) |
|---|---|---|---|---|
| 42 | 0.45 / 0.60 / 0.67 | 4.4 / 7.5 / 6.5 | 0.67 | 0.79 |
| 43 | 0.39 / 0.56 / 0.64 | 3.7 / 4.9 / 10.3 | 0.73 | 0.65 |
| 44 | 0.41 / 0.53 / 0.71 | 4.7 / 5.8 / 6.7 | 0.61 | 0.61 |

Same conclusion as the single-frame pair, with a small refinement: mode-A
states are slightly less certain (disagreement 1.3-1.4x that of successes,
AUC 0.61-0.73), but the disagreement (0.5-0.6) is 2-6x smaller than the
energy gap between classes (1.1-3.1) and far smaller than the 10-15 point
energy the refiner removes along its path. The shortcut is shared target
bias; an ensemble-pessimistic critic could shave a little off mode A but
cannot be the fix. Final ranking of the critic story: E9 (agent-placement
target) explains the mechanism, E4 quantifies the cost even for a sampler,
E7 excludes uncertainty-based remedies.

## E10 -- the fix: counterfactual agent augmentation of the critic (launched 2026-09-08)

Correction to the fix framing above: the PushT success check (`swm/PushT-v1`
`eval_state`) compares the **agent and block positions jointly** (`||goal[:4] -
state[:4]|| < 20` and block angle < 20 deg). The agent's goal position is part
of the task, so erasing the agent from goal frames would break the criterion.
The E9 defect is not that the critic reads the agent; it is that expert data
never contains "agent where the expert's agent ended, block elsewhere", so
nothing pins V high there and the refiner walks into that hole.

Fix = cover the counterfactual on the query side, goal untouched:

* `tools/cache_agent_aug` re-renders every train frame from its logged state
  with the agent displaced by a per-episode constant offset (uniform target
  over the arena, agent keeps its real motion), encodes it with the frozen WM
  and stores a row-aligned fs1 cache with `transit` = steps to walk the agent
  back (`|delta| / p90 per-step agent speed`).
* TD teacher (`train/metric`, learner td) and the co-trained critic
  (`train/lip_ac`): with prob `aug_p` the query `z_t` (window) is swapped for
  its displaced-agent latent and the label becomes `d + transit` (reached:
  `dist + transit`; bootstrap: `c(n_eff + transit) + gamma^(n_eff+transit)
  d(z_tn, z_g)` with the REAL successor). Upper bound on the true cost (walk
  back, then follow the data); the low expectile keeps the usual optimism.
* Actor sampling, goals, expansion tuple and deployment are unchanged, so the
  deployed queries stay in distribution (`aug_p = 0.5` keeps half real).
* `model=rlp` gains stage `cache_aug` and the `agent_aug.{p,transit_scale,
  cache,workers}` subtree; `counterstrike_pusht.yaml` gains `AGENT_AUG`,
  `AUG_TRANSIT`, `AUG_CACHE_TAG` (the aug cache lives on `/newcheckpoints`).
  Local smoke (synthetic PushT h5, TwoRoom WM stand-in): re-render fidelity
  pixel MAE 0.00, latent L2 re-render 0.02 vs consecutive-frame 1.48.

Arms (config B recipe exactly as `pusht-v2l05-s*-20260902`, incl. within-run
ES on val draws 50/51; report draws 42/43/44, one K=8 pass):
`scripts/sky/launch_pusht_agentaug.sh`

| arm | aug_p | transit_scale | tags |
|---|---|---|---|
| A | 0.5 | 1.0 | pusht-augA-s{0,1,2}-20260908 |
| B | 0.5 | 0 (label-preserving) | pusht-augB-s0-20260908 |

Readouts per job: `rlp`, `cem_value`, `cem_tdvalue` (E4 pairing: base 65.6 /
71.3 / 69.3 vs latent-CEM 78.9), and the E9 probe on the new critics (POST
script; base: co-trained critic V(agent@goal, block@start) 18.6 vs V(start)
22.9, 5/15 tasks satisfied by agent placement).

### E10 results, arm A seed 0 (2026-09-08 evening; job 21043, ES picked step 4000 val 63)

| condition | d42 | d43 | d44 | mean | base s0 (E1/E4) |
|---|---|---|---|---|---|
| rlp (aug critics) | 46.0 | 54.0 | 50.0 | **50.0** | 68.0 |
| cem_value (aug co-critic) | 74.0 | 70.0 | 64.0 | 69.3 | 72.7 |
| cem_tdvalue (aug teacher) | 58.0 | 74.0 | 54.0 | 62.0 | 67.3 |

Aug cache: 2.0M rows in 14 min, expert agent speed p50 8.1 / p90 18.8 px/step,
mean transit 9.4 steps; renderer fidelity pixel MAE 0.43 but latent L2
re-render 0.66 vs consecutive-frame 1.23 (JPEG artifacts; `jpeg_quality=auto`
added for a v2 cache, not used here).

E9 probe on the augmented critics (same 15 tasks): the counterfactual hole is
UNCHANGED -- collapse tasks {2, 4, 5, 9} before and after (task 9: 22.9 -> 0.6
base, 12.8 -> 1.1 aug; task 4: 15.1 -> 1.6 vs 13.6 -> 1.2). Means: ac
V(agent@goal, block@start) 18.8 (base 18.6), V(agent@start, block@goal) 12.7
(base 17.8), agent sweep range 15.7 (19.7), block 28.5 (26.1). Half of all
critic queries were displaced-agent frames, so an augmentation that left the
probe untouched on exactly the same tasks points at the probe, not the critic.

**Confound found in the base heatmaps:** on task 2 the block's start and goal
positions coincide (right panel, o on *), i.e. the block barely moves in the
25-step window and the expert spends the window repositioning the agent. For
such tasks a low V(agent@goal, block@start) is CORRECT -- the remaining work
IS agent transit. If the collapse tasks {2, 4, 5, 9} are the small-block-move
tasks, E9's "agent shortcut" reading is wrong and E10 targeted a non-defect
(consistent with it hurting: the transit term reshapes V at real states for
no gain). Check running: `scripts/pusht_diag/task_geometry.py` (job
pusht-geom-20260908) reports per-task block displacement/rotation and the
within-tolerance flag for draws 42-44; `block_sensitivity.py` now records the
same geometry per task.

### Task geometry (job pusht-geom-20260908): the E9 confound is confirmed

Per draw (50 h25 tasks): block already within the success tolerance at the
START on 18 / 18 / 9 tasks (d42 / d43 / d44); block moves < 40 px on 29 / 31 /
25; median block move 32 / 23 / 40 px vs median agent move 143 / 118 / 142 px.
Draw-42 first-15 tasks with the block already in tolerance: {2, 4, 5, 9, 10,
12} -- E9's collapse set {2, 4, 5, 9, 10} is a subset. **E9's "agent shortcut"
was the critic being right**: on those tasks the remaining work is agent
transit, and a temporal-distance critic that drops to ~1 once only the agent
is placed is correctly calibrated. h25 PushT is agent-transit dominated; the
critic's agent weighting is legitimate. E10 (E9-motivated) therefore
addressed a non-defect, consistent with its seed-0 result (rlp 50 vs 68).
What E4/E5 still show -- the critic ranks plans worse than latent L2 and is
optimistic on imagined failures -- needs a different mechanism; see the
per-class outcome cross below.

### Outcomes by task class (per-episode arrays from the diag3 / e4 / augA logs; 3 actors x draws 42-44 = 450 cells)

Classes from the task geometry: block already within the success tolerance at
the start (135 cells), block moves < 40 px (120), block moves >= 40 px (195).

| condition | block in tolerance | block < 40 px | block >= 40 px | overall |
|---|---|---|---|---|
| rlp | 74.8 | 57.5 | 64.1 | 65.6 |
| cem_latent | 84.4 | 75.0 | 77.9 | 79.1 |
| CEM plan through the LIP path (K=0) | 76.3 | 75.0 | 84.6 | 79.6 |
| CEM plan + K=8 refinement | 72.6 | 67.5 | 67.2 | 68.9 |
| cem_value (co-trained critic) | 77.0 | 65.0 | 71.3 | 71.3 |
| cem_tdvalue (teacher) | 78.5 | 65.0 | 65.6 | 69.3 |
| rlp_tdvalue | 71.1 | 45.0 | 50.8 | 55.3 |
| AUG-A rlp s0 (150 cells) | 53.3 | 45.0 | 50.8 | 50.0 |
| AUG-A cem_value s0 | 77.8 | 67.5 | 64.6 | 69.3 |
| noop | 2.2 | 0.0 | 0.0 | 0.7 |

Mode-A cells (CEM ok, RLP fail) by class: 23 / 27 / 42 -- proportional to
class size (30 / 27 / 43 % of cells), i.e. RLP's deficit is NOT block-specific.

Reading: the critic loses to latent L2 in EVERY class, including the pure
agent-transit tasks where the block must merely not be disturbed (77-78.5 vs
84.4 for the sampler; 74.8 for the refiner). That is a precision problem near
the goal, not an agent/block attribution problem: the success tolerance
(20 px joint) is ~2 agent steps, and a temporal-distance head at expectile
0.03 is flat at V ~ 1-3 there, while latent L2 has a sharp minimum at the goal
configuration. Together with E5 (optimism on imagined off-manifold latents)
this points at the deployable, fully-offline fix: keep the temporal-distance
critic for long-range ordering and let the latent distance carry the
near-goal precision (hybrid cost / residual-on-latent critic), tested first
as a zero-training deploy-time blend for CEM, then as the training objective
of the refiner.

### E10 arm B seed 0 (label-preserving augmentation, transit 0; job 21066, ES final val 32)

| condition | d42 | d43 | d44 | mean | by class (in-tol / <40 / >=40) |
|---|---|---|---|---|---|
| rlp | 28.0 | 34.0 | 12.0 | **24.7** | 13.3 / 32.5 / 27.7 |
| cem_value | 36.0 | 44.0 | 30.0 | 36.7 | 24.4 / 35.0 / 46.2 |
| cem_tdvalue | 40.0 | 46.0 | 32.0 | 39.3 | 31.1 / 40.0 / 44.6 |

E9 on the arm-B co-critic: agent sweep range 9.8 (base 19.7, arm A 15.7),
block 32.4, V(agent@start, block@goal) 9.3. Telling the critic that the agent's
position does not matter (same label for the displaced frame) halves its agent
sensitivity and collapses success on the pure agent-transit class (13-31%),
where the block only has to be left alone and the agent placed within 20 px.
Arm A (transit label) sits between base and B. So: agent sensitivity is
load-bearing on h25 PushT; the critic's remaining deficit vs latent L2 is
precision, not attribution. E10 closed negative; E11 (hybrid cost) running.

### E10 arm A seed 1 (job 21064, ES step 4000 val 63): rlp 52/72/48 = **57.3** (base s1 64.0), cem_value 64.7 (73.3), cem_tdvalue 67.3 (68.7); E9 ac V(agent@goal, block@start) 17.9 vs V(start) 21.1. Consistent with seed 0: E10 negative.

## E12 -- critic near-goal resolution arms (launched 2026-09-09, PushT + Cube)

Both arms keep the config-B recipe and change only the critic training
(teacher + co-critic): **E01** expectile 0.1 flat (teacher 0.03 -> 0.1;
co-critic 0.1 -> 0.03 schedule -> 0.1 flat) as the control on expectile
flattening; **NEAR** near-goal hindsight oversampling (30 % of in-episode
goals drawn 1..3 steps ahead, `near_frac=0.3 near_max=3`). PushT tags
`pusht-{e01,near}-s{0,1,2}-20260909` (readouts rlp / cem_value / cem_tdvalue
vs base 65.6 / 71.3 / 69.3); Cube tags `rlp-cu-{e01,near03}-20260909`
(unigamma yaml, CACHE_VERSION `cu-v2-n1s2-{e01,near03}-v1`, h25/h100 vs
base 90.0 / 86.0). Success-tolerance relabeling (`TOL_RELABEL=1`) is
implemented and smoke-tested but not launched (user chose the two arms).

## E13 -- critic granularity: probe + arms (launched 2026-09-09, p1)

Question: can the temporal-distance critic be made to resolve the last steps
(the joint 20 px / 20 deg success set ~ 2 agent steps) while staying the only
planning objective? Knobs (teacher + co-critic unless noted), all offline:

| arm | change | tags |
|---|---|---|
| E01 | expectile 0.1 flat (control on flattening) | pusht-e01-s{0,1,2} (running) |
| NEAR | 30 % of in-episode goals at 1..3 steps | pusht-near-s{0,1,2} (running) |
| TOL | success-tolerance relabeling: MC target = first entry into the env's success set (zero set = task success set) | pusht-tol-s{0,1,2}-20260909 |
| EXPN | distance-dependent expectile: 0.5 for targets < 3 steps, 0.03 else | pusht-expn-s0-20260909 |
| N5 | teacher n-step 5 (exact MC targets within 5 steps; co-critic stays n50) | pusht-n5-s0-20260909 |
| NW | near-goal loss weight (1+d)^-1, mean-normalised | pusht-nw-s0-20260909 |
| COMBO | TOL + NEAR + EXPN + N5, three seeds, after the singles report | -- |

Resolution probe (`scripts/pusht_diag/resolution_probe.py`, job
pusht-resprobe-20260909 on the base s0 critics): goal configs of the 50
draw-42 tasks perturbed radially (block-only / agent-only / joint 0..60 px,
block angle 0..40 deg), rendered from state, V(pert, goal) per critic and
latent L2; readouts = V(r)-V(0) curves, relative slope inside 30 px / 20 deg,
AUC(inside tolerance vs 20-40 near miss). Re-run with `CRITICS=` on each new
critic; a critic that scores well here AND plans better validates the
granularity reading. Cube: E01 + NEAR running (`rlp-cu-{e01,near03}`);
TOL/EXPN/N5/NW port to the legacy overlays once PushT picks winners.

### E11 final (3 actors x draws 42-44, 450 cells per row)

| CEM objective (standardized latent L2 + lam * standardized critic) | co-trained critic | teacher |
|---|---|---|
| lam 0.5 | 76.9 | 74.7 |
| lam 1.0 | 77.6 | 76.2 |
| lam 2.0 | 74.0 | 73.3 |
| reference: latent only 78.9, critic only 71.3 (ac) / 69.3 (td) | | |

No blend beats latent distance alone; the critic adds no ranking information
on these h25 tasks and degrades the objective as its weight grows.

### E12 arm E01 (expectile 0.1 flat, teacher + co-critic), seeds 0 / 1

| condition | s0 (base s0) | s1 (base s1) |
|---|---|---|
| rlp | 56.0 (68.0) | 57.3 (64.0) |
| cem_value | 68.7 (72.7) | 67.3 (73.3) |
| cem_tdvalue | 70.7 (67.3) | 72.0 (68.7) |

ES picked step 2000 (val 68) / final (val 76). Less optimism makes the
offline teacher a slightly better sampler objective (+3) but the co-critic
and the refiner worse (-10 on rlp): the expectile-flattening explanation
does not carry over to the planner. Seed 2 pending.

### E10 arm A complete (3 seeds): rlp 50.0 / 57.3 / 58.0 = median **57.3** (base 68.0 / 64.0 / 64.7, median 64.7); cem_value 69.3 / 64.7 / 68.0 (base 72.7 / 73.3 / 68.0); cem_tdvalue 62.0 / 67.3 / 67.3 (base 67.3 / 68.7 / 72.0). Agent-displacement augmentation: -7 median on the planner, -4 on critic-CEM. CLOSED negative.

### E12 arm NEAR (near-goal oversampling 0.3 at 1..3 steps, teacher + co-critic), seeds 1 / 2 (seed 0 pending)

| condition | s1 (base s1) | s2 (base s2) |
|---|---|---|
| rlp | 68.7 (64.0) | 68.7 (64.7) |
| cem_value | 74.7 (73.3) | 75.3 (68.0) |
| cem_tdvalue | 70.0 (68.7) | 70.7 (72.0) |

ES: final (val 75) / step 4000 (val 76) -- validation 12 points above the
E10 arms. First positive signal of the campaign: +4.7 / +4.0 on the planner
and the co-trained critic up as a CEM objective on both seeds. Resolution
probe (job 21233) still waiting for a CPU node.

### Resolution probe (job pusht-resprobe2, base s0 critics, draw 42, 50 goal configs; query = real 4-frame history window with the perturbation applied to every frame, goal = goal frame tiled as at deployment)

`rel slope` = fraction of the 60 px / 40 deg rise already reached at 30 px / 20 deg (higher = sharper near the goal); AUC = P(V inside the 20 px / 20 deg tolerance < V at a 25-40 px / 25-40 deg near miss).

| objective | block slope / AUC | agent slope / AUC | joint slope / AUC | angle slope / AUC | joint curve at 10/20/30/60 px |
|---|---|---|---|---|---|
| co-trained critic | 0.46 / 0.95 | 0.27 / 0.65 | 0.43 / 0.86 | 0.47 / 0.93 | 0.6 / 2.0 / 3.8 / 8.9 |
| offline teacher | 0.49 / 0.95 | 0.31 / 0.71 | 0.42 / 0.88 | 0.48 / 0.91 | 0.3 / 1.0 / 1.8 / 4.4 |
| latent L2, single frame (the WM cost) | 0.63 / 0.95 | 0.53 / 0.92 | 0.60 / 0.94 | 0.66 / 0.91 | 2.5 / 4.8 / 6.9 / 11.5 |
| latent L2, 4-frame window vs tiled goal | 0.56 / 0.94 | 0.43 / 0.76 | 0.52 / 0.89 | 0.59 / 0.92 | 3.7 / 8.4 / 13.2 / 25.5 |

The granularity reading is confirmed and localised: the critic's curve is convex
(7 % of its 60 px rise inside 10 px vs 22 % for latent L2), and the deficit
is almost entirely the AGENT term -- an agent 30 px off the goal moves V by
1.0 (out of 8.9 at 60 px) and the inside-vs-near-miss AUC is 0.65, where
latent L2 gives 0.92. Block position and angle are resolved as sharply as
latent L2 (AUC 0.95 / 0.93). Since success needs the agent inside the joint
20 px too, this is the precision the refiner lacks. JPEG vs re-rendered goal
frame: identical numbers (no artifact confound). Figure
docs/figures/pusht_diag/resolution_probe_base.png.

### Windowed latent-L2 control (job rlp-pusht-l2w4ctrl, actor-independent, draws 42-44)

`cem_l2w4` (4-frame window L2 vs tiled goal, the w4 critic's exact input) = 68 / 78 / 66 = **70.7**, vs single-frame latent L2 78.9 and the w4 co-trained critic 71.3 on the same draws. Like-for-like, the critic ties a windowed L2; the 8-point gap to the WM cost is the WINDOW at the objective, not the learned function: summing four frames' distances to a static tiled goal rewards arriving early and dwelling, which a 25-step task with a 5-block plan cannot do. Two consequences: (i) the w4 input that helped the refiner (+8.7 over w1 under config B) carries a deploy-side handicap of the same size, so w1/w2 critics with near-goal training deserve a retest; (ii) the tiled-goal train/deploy mismatch (TILE arms, running) sits exactly here.

### E13 seed-0 arms landed (base s0: rlp 68.0 / cem_value 72.7 / cem_tdvalue 67.3)

| arm | change | rlp | cem_value (co-critic) | cem_tdvalue (offline teacher) | ES val |
|---|---|---|---|---|---|
| TILE | tiled goal frames, teacher + co-critic | 58.0 | 56.7 | 72.7 | 65 |
| NEARTILE | tiled goals + near-goal 0.3 | 56.7 | 64.0 | **76.0** | 56 |
| N5 | teacher n-step 5 | 62.7 | 70.7 | **76.7** | -- |
| NW | near-goal loss weight (1+d)^-1 | 66.0 | 71.3 | 73.3 | -- |

**The teacher/co-critic split.** Every one of these changes made the OFFLINE
teacher a sharper sampler objective (+5 to +9, N5 and NEARTILE within 3 of
latent L2's 78.9) and none of it reached the planner: the co-trained critic
came out WORSE than base in all four (56.7-71.3 vs 72.7) and the refiner with
it. The co-training stage -- 3,000 steps at n-step 50, expectile 0.1 -> 0.03,
plus value expansion on the refiner's imagined rollouts -- erodes the sharpness
the teacher brings in. NEAR (seeds 1/2) is the one arm where cem_value rose,
and it is the only one that also changed the co-critic's own sampling.
Follow-ups launched at seed 0: freeze the critic at the sharpened teacher
(`planner.freeze_critic_frac=0`, arms N5NEARTILE_FRZ / N5NEAR_FRZ) and
co-train without value expansion (`EXPAND=0`, arms NEAR_NOEXP / N5NEAR_NOEXP).

### Resolution probe over the trained critics (job pusht-resprobe3; same protocol)

| critic | agent slope / AUC | joint slope / AUC | agent curve at 10/20/30/60 px |
|---|---|---|---|
| base ac | 0.27 / 0.65 | 0.43 / 0.86 | 0.2 / 0.5 / 1.0 / 3.8 |
| NEAR s1 ac / s2 ac | 0.28 / 0.64, 0.29 / 0.67 | 0.41 / 0.88, 0.43 / 0.88 | 0.1 / 0.4 / 0.9 / 3.3 |
| E01 s0 ac / s1 ac | 0.33 / 0.65, 0.34 / 0.69 | 0.42 / 0.85, 0.42 / 0.89 | 0.3 / 0.8 / 1.4 / 4.3 |
| augA s0 ac | 0.31 / 0.69 | 0.45 / 0.89 | 0.1 / 0.4 / 0.7 / 2.2 |
| latent L2 | 0.53 / 0.92 | 0.60 / 0.94 | 1.2 / 2.4 / 3.7 / 6.9 |

Two readings. (1) NEAR did not change the static near-goal curves at all
(agent AUC 0.64-0.67 = base), yet it gained +4-5 on the planner: its effect
is not "sharper V on rendered near-goal states"; whatever it fixed lives on
the imagined-rollout side or in the 1-3 step labels along real trajectories.
(2) The agent flatness is common to EVERY temporal-distance critic and is
largely the target's semantics: at 8-19 px/step an agent 30 px off IS 1-2
steps of work, so V rises by ~1 where latent L2 rises by 3.7. Combined with
E5 (imagined-state optimism gap ~3) that 1-step margin is below the critic's
own noise on imagined latents -- the refiner cannot see it. Consequence: the
lever is not a sharper offline target but the critic's calibration on
IMAGINED latents near the goal -> imagination-MC term (WM rollouts along the
data's own actions, exact remaining-steps labels; arms NEAR_IMAG,
NEAR_IMAG_NOEXP, N5NEAR_IMAG_NOEXP).

### E12 closes: NEAR 3 seeds, E01 3 seeds; W2NEAR seed 0

| arm | rlp s0 / s1 / s2 (base 68.0 / 64.0 / 64.7) | median (base 64.7) | mean (base 65.6) | cem_value mean (base 71.3) | cem_tdvalue mean (base 69.3) |
|---|---|---|---|---|---|
| NEAR (near-goal 0.3 at 1..3 steps) | 63.3 / 68.7 / 68.7 | **68.7** | 66.9 | 73.8 | 72.0 |
| E01 (expectile 0.1 flat) | 56.0 / 57.3 / 54.0 | 56.0 | 55.8 | 68.4 | 72.0 |
| W2NEAR (2-frame critic + near-goal), s0 | 56.0 | -- | -- | 50.7 | 64.7 |

NEAR: +4.0 median, +1.3 mean -- real but small, and seed 0 moved the other
way (63.3 vs 68.0), i.e. inside the seed lottery. Its offline teacher is
consistently sharper as an objective (72.0 vs 69.3) and its co-critic slightly
(73.8 vs 71.3). E01 closed negative (-8.7 median). W2 critics stay behind w4
(56.0), matching config B's w4 > w2 finding.

### W1NEAR seed 0 (single-frame critic + near-goal 0.3): rlp 61.3, cem_value 68.7, **cem_tdvalue 78.0**

The first learned objective at parity with latent L2 (78.9 on these draws): a
single-frame offline teacher with near-goal oversampling ranks CEM samples as
well as the WM's own cost. Its co-trained successor drops to 68.7 and the
refiner to 61.3 -- the same split as every other arm, now at the strongest
teacher. Follow-up: refiner trained against this teacher frozen
(W1NEAR_FRZ, W1N5NEAR_FRZ).

### NEAR5 seed 0 (near-goal fraction 0.5): rlp 63.3 (base 68.0), cem_value 73.3, cem_tdvalue 72.0 -- no gain over 0.3 at seed 0.

### Cube E01 (expectile 0.1 flat, teacher + co-critic; job rlp-cu-e01, config-B cube recipe, 3 seeds)

h25 87.3 / 88.7 / 89.3 = median **88.7** (base 88.7 / 90.0 / 92.7, median 90.0); h100 84.7 / 84.0 / 87.3 = median **84.7** (base 82.7 / 86.0 / 86.7, median 86.0). Slightly negative on both horizons, same direction as PushT (-8.7). Expectile 0.1 closed negative on both environments. Cube NEAR (job 21172) has been RECOVERING for 7 h waiting for a 4-GPU node.

### E13/E14 batch (2026-09-09 morning). Single-seed arms are compared with the base MEAN 65.6 / median 64.7 (seed 0's 68.0 is the best base seed); base cem_value 71.3, cem_tdvalue 69.3.

| arm | seeds | rlp | cem_value (co-critic) | cem_tdvalue (teacher) |
|---|---|---|---|---|
| TOL (success-tolerance relabeling) | s0/s1/s2 | 52.7 / 52.0 / 48.0 = median **52.0** | 66.0 / 64.7 / 68.0 | 70.0 / 72.7 / 70.7 |
| EXPN (expectile 0.5 below 3 steps) | s0 | 66.7 | 70.7 | 68.0 |
| NEARM5 (near band 1..5) | s0 | 59.3 | 75.3 | 69.3 |
| NEARA (+ actor near-goal problems) | s0 | 63.3 | 72.0 | 76.0 |
| NEAR_IMAG_NOEXP (near + imagination-MC, no expansion) | s0 | 58.0 | 74.7 | 75.3 |

TOL is decisively negative on the planner (-12.7 median at 3 seeds) while
again improving the teacher as an objective (+2). EXPN is neutral. The
near-goal variants keep the pattern: co-critic 72-75 (>= base 71.3), teacher
69-76, planner 58-63 (<= base).

**Campaign-level reading (2026-09-09).** Across 14 arms we have moved the
critic's quality as a *ranking objective* from 67-71 to 75-78 (W1NEAR teacher
78.0 = latent parity) by several independent routes, and the refiner has not
benefited from any of them: the best three-seed planner effect is NEAR's +4
median, and every sharper critic gives a planner at or below base. The E4
premise "the critic is the defect" was right about the ranking deficit and
wrong about where the RLP-vs-CEM gap comes from: with a latent-parity critic
in hand, the remaining ~14 points are the refiner's optimisation -- E1/E2
showed the learned K=8 update lowering the critic's energy while lowering
success, i.e. exploiting critic error along the WM, and that exploitation
does not disappear when the critic is sharper. Pending: frozen-teacher arms
(does removing co-training change the refiner?), remaining imagination-MC
and W1-frozen arms, cube NEAR.

### NEAR_IMAG seed 0 (near-goal 0.3 + imagination-MC 1.0, expansion kept): rlp 64.0, **cem_value 76.7**, cem_tdvalue 75.3 (ES step 2000, val 73)

The imagination-MC term lifts the CO-TRAINED critic to 76.7 as a sampler
objective (base 71.3; without expansion 74.7), the closest any deployed
critic has come to latent L2's 78.9, and the highest ES validation of the
campaign (73). The refiner with it is at base (64.0 vs 65.6). Same split.

### N5NEARTILE_FRZ seed 0 (refiner trained against the FROZEN n5 + near + tiled-goal teacher, no co-training): rlp **50.7**, cem_value = cem_tdvalue 70.0 / 69.3

Removing co-training costs the refiner ~15 points even with a sharpened
teacher. Co-training's value expansion on the refiner's own rollouts is what
makes a critic usable as a gradient field on imagined states, even though it
makes the same critic a worse ranking objective on rendered states (the
teacher/co-critic split has this sign). Consistent with NEAR_IMAG: exact
labels on imagined states (imagination-MC) give the best deployed critic
(76.7) AND a refiner at base, where the frozen teacher gives neither.

### W1N5NEAR_FRZ seed 0 (refiner against the FROZEN single-frame n5 + near teacher): rlp 60.0, critic 72.7 / 72.0 (frozen, so value_ac = value_td)

Single-frame frozen teacher: refiner 60.0 (vs 50.7 with the frozen w4 teacher, base mean 65.6) -- the window handicap shows on the refiner side too, but a frozen teacher still loses to co-training. Its n5 teacher scores 72 as an objective, below the n1 W1NEAR teacher's 78.0.

### W1NEAR_FRZ seed 0 -- the first arm where the planner moves WITH the critic

Refiner trained against the FROZEN single-frame near-goal teacher (w1,
n-step 1, near 0.3; no co-training): **rlp 72 / 84 / 60 = 72.0** (base s0
68.0, base mean 65.6, best previous single-draw 80), deployed critic 78.7 as
a CEM objective (latent 78.9). Contrast: the same teacher co-trained (W1NEAR)
gave 61.3, and its n-step-5 frozen sibling 60.0 -- the n1 teacher's 78.0 is
what carries it. Single seed; replication at seeds 1/2 launched
(pusht-w1near_frz-s{1,2}-20260909). Also landed: N5NEAR_IMAG_NOEXP s0 rlp
54.0 (critics 74.0 / 74.0), negative.

### N5NEAR_FRZ seed 0 (frozen w4 n5 + near teacher, no co-training): rlp 66.0, critic 73.3 (ES val 76)

Neutral on the planner (base mean 65.6) with a 73.3 frozen critic; the tiled
variant of the same teacher gave 50.7, so tiled goals hurt the frozen route
too. Frozen-teacher ledger at seed 0: w4 n5+near 66.0, w4 n5+near+tile 50.7,
w1 n5+near 60.0, **w1 n1+near 72.0** -- the planner tracks the teacher's
objective quality (73 / 69 / 72 / 78.7) only once co-training is out of the
loop, and the n1 single-frame teacher is the one at latent parity.

### N5NEAR_NOEXP seed 0 (co-training WITHOUT value expansion, n5 + near): rlp 64.0, cem_value 70.7, cem_tdvalue 73.3 (ES val 77)

Removing value expansion from co-training is neutral on the planner (base
mean 65.6) and leaves the co-critic at base (70.7 vs 71.3): expansion is not
what erodes the sharper teacher during co-training -- the plain TD phase on
the n-step-50 / expectile-annealed recipe already does.

### NEAR_NOEXP seed 0 (co-training without value expansion, near 0.3): rlp 62.0, cem_value 72.0, cem_tdvalue 75.3 (ES val 76). Neutral-to-slightly-negative; with N5NEAR_NOEXP (64.0) it closes the expansion question: expansion is not the erosion mechanism. NEARN5 (job 21302) preempted mid-eval, recovering.

### W1NEAR_FRZ seed 1: rlp 76 / 72 / 64 = **70.7** (base s1 64.0), critic 76.7 / 77.3 (ES: final, val 70)

Two seeds: 72.0 / 70.7 vs base 68.0 / 64.0 (+4.0 / +6.7). Seed 2 pending.

### W1NEAR_FRZ_SM seed 0 (frozen w1 near teacher + randomized smoothing, M=4, sigma 0.1): rlp 66 / 78 / 52 = **65.3**, critic 78.0 / 76.7 (ES step 4000, val 68)

Same teacher as W1NEAR_FRZ (72.0 at this seed), smoothing in the actor loss
only: -6.7. Draw 44 collapses (52 vs 60); draws 42/43 hold (66/78 vs 72/84).
Not the smoothing scale we want, or smoothing trades sharpness for robustness
the refiner did not need at sigma 0.1; sigma 0.3 pending.

### E15 -- refiner anti-exploitation arms on the frozen single-frame near-goal teacher (seed 0; reference W1NEAR_FRZ s0: rlp 72.0, critic 78.7)

| arm | rlp | cem_value (deployed critic) | cem_tdvalue (teacher) | ES val |
|---|---|---|---|---|
| randomized smoothing, M=4, sigma 0.1 | 65.3 | 78.0 | 76.7 | 68 |
| randomized smoothing, M=4, sigma 0.3 | 64.0 | 78.0 | 78.0 | 62 |
| pessimistic imagination, TD phase kept | 24.7 | 46.0 | 78.7 | 27 |
| **pessimistic imagination only** (td_weight 0: hinge is the critic's only co-training signal) | **70.0** | **82.0** | 78.0 | 66 |

Smoothing is negative at both scales (-7 / -8 vs the unsmoothed twin) with
the critic unchanged: the refiner's exploits are not sharp minima that noise
removes. Pessimistic imagination with the TD phase collapses (the n-step-50
TD pull and the hinge push fight; the co-critic ends at 46). **Pessimistic
imagination alone gives the first deployed critic ABOVE latent L2 as a
sampler objective: 82.0 (76 / 88 / 82) vs 78.9**, with the refiner at 70.0
(70 / 84 / 56), the same level as the frozen teacher (72.0). So the critic
side is now beyond CEM's own objective; the refiner still converts ~70 of it.
Replication of PESSONLY at seeds 1/2 launched.

### Cube NEAR arm blocked (2026-09-09 evening)

Both `rlp-cu-near03` attempts after the first preemption (jobs 21172
recovery, 22026) died in setup: `git clone` of the private
`Value_Metric_LeWM` repo fails with "Invalid username or token" -- the
`GIT_TOKEN` fine-grained PAT that the unigamma yaml pulls from the SkyPilot
secrets manager has expired/been revoked since the cube E01 job (which cloned
fine ~24 h earlier). The PushT yaml does not clone that repo and is
unaffected. Needs a refreshed GIT_TOKEN secret before any cube/unigamma job
can run; the near-goal cube arm is otherwise ready (CACHE_VERSION
cu-v2-n1s2-near03-v1, NEAR_FRAC=0.3).

### Head-to-head on the pessimistic-only actor (draw 42, one actor, three planners, recordings kept)

`rlp` 70.0 | `cem_value` (its own critic) 74.0 | `cem_latent` (WM cost) 78.0.
So on this draw the ordering RLP < critic-CEM < latent-CEM persists even with
the critic that beats latent L2 on the 3-draw mean (82.0 vs 78.9) -- draw 42
is the draw where cem_value is weakest (76 in the training job, 74 here).
Recordings for all three conditions are on
`/newcheckpoints/.../pusht-ivr-pessonly-s0-20260909` for the imagined-vs-real
analysis (`scripts/pusht_diag/imagined_vs_real.py`).

## E16 -- imagined vs real: the residual gap is WORLD-MODEL exploitation, not critic ranking (2026-09-10)

Setup: one actor (pessimistic-only critic, seed 0), draw 42, all three
planners recorded; every executed 25-step plan re-imagined through the frozen
WM from the real history; a ridge probe latent -> state (held-out R2: agent
0.95, block 0.97, angle 0.91; median error agent 21 px, block 11 px) decodes
both the imagined and the real latents. `scripts/pusht_diag/imagined_vs_real.py`,
summary `docs/figures/pusht_diag/imagined_vs_real_pessonly_s0_d42.json`.

First plan, FAILED tasks (the class that decides the gap):

| planner | critic V imagined -> real (final block) | optimism gap | WM latent divergence (final) | decoded BLOCK error to goal, imagined / real | decoded AGENT error, imagined / real |
|---|---|---|---|---|---|
| rlp (refiner) | 3.6 / 8.5 | **+4.97** | **9.75** | **22.8 px / 73.5 px** | 63.1 / 62.7 px |
| cem_value (same critic) | 3.5 / 7.4 | +3.96 | 8.61 | 35.3 px / 66.7 px | 47.7 / 46.9 px |
| cem_latent (WM cost) | 5.3 / 6.6 | +1.21 | 6.86 | **56.3 px / 61.7 px** | 35.1 / 36.9 px |

Successful tasks are calibrated for all three (optimism gap 0.0-0.7,
divergence 4.3-4.5, imagined and real block error within 3 px).

**Mechanism.** On failures the imagined AGENT trajectory is accurate (imagined
vs real agent error within 1 px for every planner) while the imagined BLOCK is
fiction: the refiner's plans imagine the block arriving at 22.8 px from the
goal when it really ends 73.5 px away -- a 51 px hallucination. The WM
invents block motion for near-miss pushes, and the critic then correctly
scores a state that never happens (V falls 10.2 -> 3.6 in imagination,
10.4 -> 8.5 in reality). The hallucination is ordered exactly like success:
latent-CEM 5 px < critic-CEM 31 px < refiner 51 px, as is the divergence
(6.9 < 8.6 < 9.8) and the optimism gap (1.2 < 4.0 < 5.0).

**Why latent L2 wins despite being a worse ranking objective.** Its low-cost
set is tight -- to fake it the WM would have to render a latent nearly
identical to the goal frame's -- whereas a learned V has a broad low-V basin
that hallucinated latents can enter. Optimizer strength then makes things
worse, not better: the gradient refiner finds those basins more effectively
than 9,000 CEM samples do. This is why 14 arms of critic sharpening (67 -> 82
as a ranking objective) never moved the planner, and why E1 (refining a CEM
plan makes it worse while lowering energy) and E2 (K=64 diverges) were the
same phenomenon all along.

**Consequence for the fix.** The lever is the imagined trajectory's
trustworthiness, not the critic. The probe that produced this table is itself
the cheapest candidate: it decodes block xy from a latent at R2 0.97, so a
plan-time penalty on imagined block displacement that no agent contact
supports is computable from the imagined latents alone (no second WM, fully
offline), differentiable, and usable both in the refiner's training loss and
as an eval-time cost term.

### W1NEAR_PESSONLY seed 1: rlp **12.0** (16/16/4), cem_value 45.3, cem_tdvalue 76.0 (ES val 17)

The seed-0 result (critic 82.0, refiner 70.0) does NOT replicate: with the
hinge as the critic's only signal the seed-1 critic collapses to 45.3 while
its frozen teacher stays at 76.0. Expected in hindsight -- the hinge is
one-sided (it can only raise V on planner-visited states) so nothing anchors
the scale once the TD term is off; seed 0 was a lottery win. Pessimistic-only
is therefore NOT a usable recipe as implemented; a two-sided version (hinge +
the data-anchored TD loss at low weight) is the only form worth retrying, and
the E16 finding says the priority is elsewhere. Seed 2 pending.

### W1NEAR_PESSONLY seed 2: rlp 74 / 78 / 56 = **69.3**, cem_value **77.3**, cem_tdvalue 74.7 (ES val 73)

Three seeds of pessimistic-only: rlp 70.0 / **12.0** / 69.3, cem_value 82.0 /
45.3 / 77.3. Two seeds land at ~70 with a critic at 77-82 (at or above latent
L2's 78.9); seed 1 collapses. Median rlp 69.3 vs base median 64.7, but the
variance is a disqualifier: the one-sided hinge has no scale anchor, so the
recipe is not deployable as implemented. If revisited, pair the hinge with the
data-anchored TD loss at low weight (`td_weight` 0.1-0.3) rather than 0.

### NEARN5 seed 0 (near-goal 0.3 + teacher n-step 5): rlp 66 / 86 / 54 = 68.7, critics 73.3 / 73.3 (ES val 78). Neutral-to-positive at one seed; draw 43 hits 86.

### E16 figures
`docs/figures/pusht_diag/ivr_distance_curves_pessonly_s0_d42.png` (critic V,
latent L2, decoded agent and block error vs plan step; imagined dashed vs real
solid, successes green vs failures red, one row per planner) and
`ivr_plans_pessonly_s0_d42.png` (arena: imagined vs real agent and block paths
for the four mode-A tasks 2, 3, 24, 29 and two both-success tasks 0, 4).

## E17 -- contact-consistency penalty (2026-09-10)

Term: `mean_t relu(move_t - move_eps) * clamp((gap_t - gap_eps)/gap_norm, 0, 1)`
over the imagined rollout, where `move` is the decoded block pose change (px,
rotation at 40 px/rad) and `gap` is the decoded agent-block centre distance
minus a contact radius. Reads as pixels of unsupported block motion; the
weight converts px into energy units. Differentiable, so the refiner descends
the constrained energy (added inside `LIPSolver._score`, which feeds both the
gradient and the plan selection) and every sampler gets it through
`MetricCost` (the latent cost is wrapped for this purpose alone).
Code `src/rlp/core/value/contact.py`, probe `tools/fit_state_probe`,
conditions `rlp_contact` / `cem_latent_contact` / `cem_value_contact`.

Probe on the PushT training cache: held-out R2 agent 0.95 / 0.95, block 0.97 /
0.97, angle 0.91 / 0.87; median error agent 21 px, block 11 px (p90 28), angle
4.3 deg. **Contact calibration over 13,913 expert transitions that move the
block: agent-block separation median 55.6, p95 93.6, p99 134.9, max 140.1 px.**

### First run used a WRONG radius (60 px placeholder in the config, not the calibration)

| condition | d42 / d43 / d44 | mean |
|---|---|---|
| rlp (frozen-teacher actor s0) | 74 / 84 / 60 | 72.7 |
| rlp + contact, w 0.15, radius 60 | 70 / 80 / 60 | 70.0 |
| cem_latent | 78 / 82 / 76 | 78.7 |
| cem_latent + contact, w 0.15, radius 60 | 74 / 78 / 80 | 77.3 |

At radius 60 the penalty fires on HALF of all legitimate expert pushes (their
median separation is 55.6 px), so it taxed real contact rather than
hallucination -- a mild loss for both planners, as expected of a mis-specified
constraint. `contact_radius` now defaults to null, which adopts the p99
(134.9 px), so the term can only fire beyond a separation at which the data
never moves the block. Reruns: `pusht-cal-w{015,100}-frz` and
`pusht-cal-w050-base` (weights 0.15 / 1.0 / 0.5).

### E17 with the calibrated radius (134.9 px): the rule is a no-op where it matters

| actor / weight | rlp | rlp + contact | cem_latent | cem_latent + contact |
|---|---|---|---|---|
| frozen-teacher s0, w 1.0 | 72.7 | **72.7** (74/84/60, draw-for-draw identical) | 78.7 | 76.7 |
| base config-B s0, w 0.5 | 68.0 | 67.3 | 77.3 | 77.3 |

At weight 1.0 the refiner's per-draw results are IDENTICAL to the
unconstrained ones, which can only mean the penalty never fired: the refiner's
imagined rollouts keep the agent within 135 px of the block even while
fabricating its motion. So the two radii bracket a hole rather than a
solution -- 60 px taxes half of all legitimate expert pushes, 135 px catches
nothing the refiner does. Separation alone may simply not discriminate,
because the world model hallucinates at plausible separations.

Rather than guess a third radius, `scripts/pusht_diag/contact_discriminator.py`
(job rlp-pusht-contactdisc) measures it on the E16 recordings: per imagined
step it labels hallucination as "the WM moved the block much more than reality
did" and scores agent-block separation, push alignment (cos angle between the
agent's displacement and the imagined block displacement), approach speed and
agent step size, plus the catch-vs-tax curve of every gap threshold.

### E17 closed: agent-block separation cannot discriminate (job rlp-pusht-contactdisc)

908 imagined steps over three planners on draw 42: 43 hallucinated (the WM
moved the block >20 px more than reality) and 274 legitimate (reality moved
the block).

| feature | AUC hallucinated vs legitimate | hallucinated p10/p50/p90 | legitimate p10/p50/p90 |
|---|---|---|---|
| agent-block gap (px) | 0.68 | 69 / 79 / 94 | 32 / 66 / 100 |
| separation at step start | 0.67 | 73 / 85 / 111 | 39 / 74 / 118 |
| push alignment (cos) | 0.48 | 0.5 / 0.8 / 1.0 | -0.2 / 0.8 / 1.0 |
| approach toward block | 0.21 | 17 / 54 / 79 | -12 / 24 / 54 |
| agent step size (px) | 0.64 | 30 / 61 / 88 | 26 / 47 / 78 |

| gap threshold | hallucination caught | real pushes taxed |
|---|---|---|
| 60 px | 90.7 % | 58.0 % |
| 80 px | 48.8 % | 29.6 % |
| 100 px | 7.0 % | 10.2 % |
| 135 px (p99 calibration) | 0.0 % | 1.5 % |

**The distributions overlap almost completely.** Hallucinated steps sit at a
median gap of 79 px, legitimate pushes at 66 px, and every threshold either
taxes more real pushes than it catches hallucination (100 px and above, where
the tax exceeds the catch) or is a blunt tax on both (60 px catches 91 % but
taxes 58 %). Push alignment is at chance, 0.48: the WM hallucinates block
motion *in the direction the agent is travelling*, which is exactly what makes
it plausible. Approach is inverted (0.21), i.e. hallucination happens while
the agent moves TOWARD the block -- near-miss pushes, as E16's arena plots
showed.

So the world model's error is not geometrically naive; it is a plausible
near-miss physics error, and no threshold on decoded contact geometry
separates it from real contact. E17 is closed negative, consistent with the
three eval arms (all within noise, draw-identical at weight 1.0). Any fix has
to come from a signal that knows the WM is wrong -- disagreement between two
world models, or a learned discriminator on real-vs-imagined transitions --
not from hand-specified physics on one model's output.

### W1NEAR_FRZ three seeds complete: rlp **72.0 / 70.7 / 67.3**, median 70.7 (base 68.0 / 64.0 / 64.7, median 64.7)

Critic as a sampler objective 78.7 / 76.7 / 76.7 (latent L2 78.9). The best
PushT recipe of the campaign: a frozen single-frame near-goal-oversampled TD
teacher, no co-training, one K=8 pass. **+6.0 on the median, +3.9 on the
mean** (72.0/70.7/67.3 = 70.0 vs 65.6), and every seed is at or above its base
counterpart. Still 8-9 points short of latent-CEM (78.9) at 1/1000 the
compute; the residual is E16's world-model exploitation, which E17 showed
cannot be removed by contact geometry.

## E18 -- draw-43 failure atlas (2026-09-10)

Actor `pusht-w1near_frz-s0` (frozen single-frame near-goal teacher, W=1 so the
heatmaps are exactly what the planner queries). Draw 43: all three planners
score 84.0 here; the refiner fails 8 tasks, 5 of them mode A (the samplers
succeed): tasks 0, 8, 9, 15, 25. Figures
`docs/figures/pusht_diag/atlas43/atlas_task{00,03,08,09,15,25}.png`, per-task
numbers `failure_atlas.json`.

### The critic is NOT the cause on any of the five

V along the expert's own continuation (which reaches the goal by construction)
falls monotonically on **every** task, 100 % of steps, from 10-15 down to 0:

| task | V along expert path | block move required | rho(V agent-sweep, BFS geodesic) |
|---|---|---|---|
| 0 | 14.6 / 12.7 / 9.3 / 6.7 / 3.2 / 0.0 | 31 px + 65 deg | 0.61 |
| 8 | 14.5 / 13.5 / 10.8 / 7.8 / 4.1 / 0.0 | 148 px + 68 deg | 0.34 |
| 9 | 12.1 / 10.4 / 7.8 / 4.8 / 3.3 / 0.0 | 78 px | **-0.36** |
| 15 | 9.9 / 7.5 / 6.5 / 5.0 / 2.7 / 0.0 | 58 px | -0.01 |
| 25 | 11.0 / 8.5 / 7.1 / 5.6 / 2.7 / 0.0 | 40 px | 0.68 |

So the objective ranks the true solution path correctly on all of them. The
agent-space BFS geodesic correlation is a red herring: it is near zero or
negative exactly on the tasks that need real block transport (9, 15), because
the critic's agent-space landscape SHOULD track "where do I have to stand to
push", not "how far to walk to the expert's final agent spot". The block-sweep
panels show the critic's minimum sitting on the block's goal pose, which is
correct.

### What actually fails, per task

**Task 8 (block must travel 148 px).** The refiner gets the block from 96 px
to 23 px of error by step 20 -- then loses it: real V rises 9.12 -> 9.97 and
the final block error is 60 px. The imagined plan claims 5.0 while reality is
10.0. Both samplers finish the same task in 24-25 steps. This is the
E16 mechanism: the last replan's imagined push is fabricated, the block slips
off, and there is no budget left to recover.

**Task 25 (40 px).** The refiner is AHEAD at step 10 (block error 12 px vs the
samplers' 38 px) and then destroys it: block error 12 -> 32 px, V 3.83 -> 5.28,
and the agent ends 83 px from its goal spot. The heatmap shows why it is
tempted: V over agent xy has a broad low basin toward the bottom-right, away
from the block, so once the block is nearly placed the cheapest direction in
the critic's landscape is to leave.

**Task 9 (78 px, no rotation).** The refiner actually SOLVES the block, final
block error 1.1 px, but ends 41.7 px from the required agent position, so the
joint 20 px test fails. The samplers end at 16.0 and 11.9 px. Pure
agent-placement miss with the block perfect.

**Task 15 (58 px).** Nobody moves the block usefully: real block error ends
19.6-26.7 px for all three, and the refiner is the closest of the three. It
fails the joint test by a whisker; this is a lottery cell, not a mechanism.

**Task 0 (31 px + 65 deg rotation).** The refiner reaches 17.1 px block error
at step 25 and 11.0 px at the end, with the agent at 13.3 px -- both inside
tolerance -- yet the recorded episode ran the full 50 steps and is scored a
failure, so the success must have been missed at the *graded* step. Worth a
separate check of first-hit scoring on rotation-heavy tasks.

**Task 3 (both fail, block barely needs to move).** Every planner drives the
block AWAY: real block error 44 -> 56 px (rlp), 45 -> 39 (cem_value), 37 -> 50
(cem_latent), while the imagined paths all claim 16-20 px. Unanimous
world-model exploitation on a task whose block starts 38 px from its goal.

### Reading

Three of the five mode-A failures are late-plan world-model exploitation
(8, 25, 3) and two are terminal-precision misses (9 agent, 15 joint). None is
a critic-ranking failure: V is monotone along the true path in every case, and
V's minimum in block space sits on the block goal. The refiner's specific
disadvantage is that it re-optimises its own imagined future every replan and
therefore keeps re-entering the fabricated-push basin, where a sampler that
never leaves the data-like action neighbourhood does not. Combined with E17
(contact geometry cannot detect the fabrication) the remaining levers are
world-model disagreement, a learned real-vs-imagined discriminator, or simply
executing fewer blocks per replan so a fabricated push is corrected sooner.

## E19 -- trust region on the plan (tighter action clip), 2026-09-10

Motivation: E18's mode-A failures are late-plan excursions (task 25 ends with
the agent 83 px off, task 8 loses a nearly-placed block at the last replan),
so a smaller per-block displacement should keep the imagined rollout inside
the action magnitudes the WM models accurately.

### Deploy-time clip rewrite on the frozen-teacher actor (job rlp-pusht-damax-frz, 3 draws)

| deploy amax | d42 / d43 / d44 | mean |
|---|---|---|
| 2.5 (trained value) | 74 / 86 / 60 | **73.3** |
| 2.0 | 74 / 84 / 54 | 70.7 |
| 1.6 | 72 / 82 / 50 | 68.0 |
| 1.2 | 68 / 76 / 56 | 66.7 |
| 0.8 | 60 / 70 / 42 | 57.3 |

Monotone degradation, -16 points from 2.5 to 0.8, and no local optimum in
between. Truncating an actor trained at 2.5 removes reach without removing
the exploitation: the refiner still walks toward the fabricated-push basin,
it just cannot get there, and it also loses the legitimate long pushes (draw
44, the transport-heavy draw, falls hardest: 60 -> 42). Reproduces the config-B
finding that deploy-amax {1.6, 2.0} was null-to-negative, now on the new recipe
and with the full curve. Trained-inside-the-clip arms (amax 1.6 and 1.0,
jobs 22216 / 22218) are the honest test and still running.

### E19b -- trained INSIDE a tighter clip, amax 1.0 (job 22218, seed 0)

rlp 68 / 82 / 54 = **68.0** (frozen-teacher baseline at amax 2.5, seed 0:
72.0), cem_value 78.0, cem_tdvalue 76.7, ES val 67. Training inside the clip
recovers most of what deploy-time truncation to 1.2 lost (66.7) but still
sits 4 points below the untightened recipe, and the transport-heavy draw 44 is
again the casualty (54 vs 60). The critic is unaffected (78.0 / 76.7 ~ the
frozen teacher's 78.7), confirming the clip acts purely on the actor's reach.
amax 1.6 (job 22216) pending.

### E19 closed: the trust region is negative on PushT, both ways

| clip | how applied | rlp (seed 0) | critic as CEM objective |
|---|---|---|---|
| 2.5 | trained (baseline) | **72.0** | 78.7 |
| 2.0 | deploy rewrite | 70.7 | -- |
| 1.6 | deploy rewrite | 68.0 | -- |
| 1.6 | trained | 62.0 | 78.0 |
| 1.2 | deploy rewrite | 66.7 | -- |
| 1.0 | trained | 68.0 | 78.0 |
| 0.8 | deploy rewrite | 57.3 | -- |

Nothing beats the untightened recipe, trained or truncated, and the trained
arms are not even monotone in the clip (1.6 -> 62.0 is worse than 1.0 -> 68.0,
and ES picked step 2000 there on val 70, so it is partly a training-noise
draw). The critic is identical across all of them (78.0-78.7), so the clip
only ever touched the actor's reach. E19 closed negative: the fabricated
pushes happen at ordinary action magnitudes, so a magnitude constraint removes
the legitimate long transports (draw 44 falls hardest in every arm) before it
removes the exploitation.

## E20 -- the last two critic arms, both negative (2026-09-10)

### Two-sided pessimism (hinge + anchored TD at weight 0.2), 3 seeds

| seed | rlp | cem_value (deployed critic) | cem_tdvalue (its frozen teacher) |
|---|---|---|---|
| 0 | 10.7 | 25.3 | 77.3 |
| 1 | 9.3 | 23.3 | 77.3 |
| 2 | 12.0 | 27.3 | 76.7 |

Worse than the one-sided version, and now consistently: the deployed critic
collapses to 23-27 on all three seeds while its own teacher stays at 77. So
the instability of PESSONLY was not a missing scale anchor -- keeping the TD
term at 0.2 makes the collapse reliable rather than curing it. The mechanism
is the hinge itself: it is applied to the refiner's own imagined terminals
every step, the refiner keeps visiting the region the hinge is inflating, and
the pair runs away. Pessimistic imagination on planner-visited states is
closed negative in every form tried (optimistic expansion 24.7, hinge only
70.0/12.0/69.3, hinge + TD 10.7/9.3/12.0).

### Depth 3 on the frozen teacher, 3 seeds

| seed | rlp (depth 3) | rlp (depth 2 baseline) | critic depth 3 | critic depth 2 |
|---|---|---|---|---|
| 0 | 69.3 | 72.0 | 76.0 | 78.7 |
| 1 | 62.7 | 70.7 | 74.0 | 76.7 |
| 2 | 68.7 | 67.3 | 73.3 | 76.7 |
| median | **68.7** | **70.7** | 74.0 | 76.7 |

-2.0 on the median, and the critic's own ranking quality drops 2-3 points on
every seed. Freezing the teacher does soften the historical penalty (-26 to
-39 under co-training) but does not flip its sign: depth 2 remains right for
PushT, on the critic as well as the actor.

**Campaign state.** The best recipe stays W1NEAR_FRZ: a frozen single-frame
near-goal-oversampled teacher, no co-training, one K=8 pass -- median 70.7 vs
base 64.7. Closed negatives now cover the target (E10 augmentation, E13
tolerance relabeling, E12 expectile), the sampling dose/band/actor-side
variants, the window, capacity (E20), the optimiser's trust region (E19),
randomized smoothing, pessimistic imagination (E20), and hand-specified
contact physics (E17). The unexplored levers are all on the world-model side:
ensemble disagreement, a learned real-vs-imagined discriminator, and the
replan frequency (RH=5 today; RH=1 corrects a fabricated push after 5
primitive steps instead of 25, at 5x the decisions per episode).

## E21 -- the "drift out of the goal" story is FALSE (2026-09-10)

The E18 atlas decoded poses from latents, but the probe's median agent error
(21 px) exceeds the success tolerance (20 px on the JOINT agent+block
distance), so it could not decide whether a rollout entered the success set.
The recorder now stores PushT's own `pos_agent` / `block_pose`
(`src/rlp/environment/world.py`), and `scripts/pusht_diag/drift_analysis.py`
evaluates the env's exact test at every step (job rlp-pusht-drift43, draw 43,
frozen-teacher actor).

**No failing rollout ever entered the success set** -- neither planner, on any
of its 8 failures:

| task | rlp min joint distance (step) | rlp final | angle at min | cem_latent min (outcome) |
|---|---|---|---|---|
| 0 | 21.7 @ 49 | 21.7 | 5.4 deg | 15.6 (ok) |
| 3 | 53.0 @ 22 | 84.0 | 8.6 deg | 35.5 (fail) |
| 8 | 20.1 @ 25 | 61.3 | **317.8 deg** | 17.6 (ok) |
| 9 | 27.4 @ 49 | 27.4 | 0.2 deg | 15.7 (ok) |
| 15 | 20.9 @ 24 | 24.6 | 6.7 deg | 17.1 (ok) |
| 25 | 29.8 @ 31 | 38.2 | 7.9 deg | 18.0 (ok) |
| 31 | 49.3 @ 49 | 49.3 | 20.6 deg | 54.8 (fail) |
| 43 | 38.7 @ 24 | 50.4 | 0.7 deg | 22.3 (fail) |

Corrections this forces:

1. **The min-over-plan objective is NOT supported.** Only 3 of 8 failures grow
   by >10 px after their minimum (3, 8, 43), and none of them was ever inside
   the success set, so a "keep the best point" objective would have kept a
   point that still fails. My E18 reading -- "passes through the goal and
   drifts out" -- was an artifact of quoting decoded BLOCK-only error against
   a joint agent+block criterion.
2. **Four of the eight are near misses, not exploitation:** tasks 0, 9, 15
   end at 21.7 / 27.4 / 20.9-24.6 px against a 20 px threshold, with angle
   error under 7 deg. The samplers clear the same tasks at 15.6-17.1 px. The
   gap is a few pixels of terminal precision, which is exactly the
   agent-resolution deficit the resolution probe measured (inside-vs-near-miss
   AUC 0.65 for the agent term vs 0.92 for latent L2).
3. **Task 8 is an angle failure, not a position failure:** 317.8 deg of error
   at its best position. `PushT.eval_state` compares RAW angles, so the block
   was rotated the "long way" and the unwrapped difference can never satisfy
   |dtheta| < 20 deg. Worth checking how many tasks across draws carry an
   unwrappable angle target.
4. Only tasks 3 and 31 are genuine gross failures (min 53 and 49 px), and
   cem_latent also fails both.

**Revised priority.** Terminal precision, not objective mismatch: 4 of 8
failures sit within 8 px of the threshold. That points at the
success-indicator head (a sharp boundary at the graded set) rather than the
best-point objective, and it is consistent with everything the resolution
probe already said.

## E22 -- per-episode attribution, draw 43 (2026-09-10)

`scripts/pusht_diag/attribution.py` (job rlp-pusht-attr43), frozen-teacher
actor. Truth = the env's recorded poses; belief = the deployed critic; promise
= the executed plan re-imagined through the frozen WM. Rule, applied in order:
**3 planner** a sampler on the same critic and WM solved it; **2 critic** the
critic calls the endpoint nearly solved (V < 4) at a true error outside
tolerance; **1 world model** the imagined terminal was rated > 3 better than
the state reality delivered; **0** none of the above.

| task | label | true min / final px | V at endpoint | imagination gap | samplers | needed |
|---|---|---|---|---|---|---|
| 0 | **3 planner** | 21.7 / 21.7 | 3.42 | 5.41 | both ok (min 15.6) | 31 px + 65 deg |
| 8 | **3 planner** | 20.1 / 61.3 | 9.70 | 4.94 | both ok (17.6) | 148 px + 68 deg |
| 9 | **3 planner** | 27.4 / 27.4 | 1.41 | 0.28 | both ok (15.7) | 78 px |
| 15 | **3 planner** | 20.9 / 24.6 | 2.20 | 0.42 | both ok (17.1) | 58 px |
| 25 | **3 planner** | 29.8 / 38.2 | 4.39 | 2.07 | both ok (18.0) | 40 px |
| 3 | **1 world model** | 53.0 / 84.0 | 12.10 | **9.22** | both fail (35.5) | 9 px |
| 31 | **2 critic** | 49.3 / 49.3 | **2.80** | 1.83 | both fail (54.8) | -- |
| 43 | **2 critic** | 38.7 / 50.4 | **3.70** | 1.70 | both fail (22.3) | -- |

**Counts: planner 5, critic 2, world model 1.**

Reading. The refiner-specific failures are all planner-attributed by
construction, and four of the five end within 8 px of the 20 px threshold
(0, 9, 15 at 21.7 / 27.4 / 20.9-24.6) while the samplers clear the same tasks
at 15.6-18.0 px: a terminal-precision gap, not a wrong objective. Tasks 31
and 43 are the first CLEAN critic failures found in the campaign: V says 2.8
and 3.7 (i.e. "3 steps to go") at 49 and 50 px of true error, on tasks every
planner fails. Note E18 measured V as monotone along the EXPERT path for the
mode-A five; 31 and 43 were not in that set, and they show the complementary
defect -- V is wrong on states the planner actually reaches. Task 3 is the
one clean WM failure, with the imagined terminal rated 9.2 better than
reality, on a task whose block only needs to move 9 px.

## E23 -- the terminal-precision hypothesis is FALSE; near-goal flatness is the expectile (2026-09-11)

E22 left four of five planner failures at 20.1-29.8 px against a 20 px
threshold with V in [1.4, 4.4], and the reading was that V regresses INTEGER
steps-to-go, so it is a staircase and grad V ~ 0 across the final step -- a
gradient refiner would have nothing to descend exactly where these episodes
miss. Two candidate fixes followed from that: a success-indicator head
(rejected on inspection -- a step function is flat on both sides of the
boundary) and a continuous target.

The continuous target was built and measured instead of assumed.
**Sub-step query interpolation**: a fraction of TD queries placed alpha of the
way from frame t to frame t+1, every remaining-distance term reduced by the
same alpha. A unit check on a cache whose latents encode the step index
confirms label == true remaining distance to 2e-06, so what follows is a fact
about the data, not an implementation slip.

`scripts/pusht_diag/value_resolution_probe.py` walks a query from frame t to
t+1 in 9 sub-steps and reports how far V moves (one step should be ~1.0),
how monotone it is, and how separated consecutive frames are. Four TD fits per
cache, 2000 steps, n-step 1, gamma 0.98, near_frac 0.3.

| cache | ||z_t - z_t+1|| / ||z_t - z_goal|| | tau | subgrid | drop across one step | monotone |
|---|---|---|---|---|---|
| PushT (job rlp-pusht-vres) | 1.284 / 2.207 = **0.68** | 0.03 | 0.0 | **+0.429** | **98%** |
| PushT | | 0.03 | 0.5 | +0.126 | 96% |
| PushT | | 0.5 | 0.0 | **+0.864** | 97% |
| PushT | | 0.5 | 0.5 | +0.667 | 98% |
| TwoRoom (local) | 7.395 / 8.443 = 1.03 | 0.03 | 0.0 | +0.444 | 66% |
| TwoRoom | | 0.03 | 0.5 | +0.258 | 62% |
| TwoRoom | | 0.5 | 0.0 | +0.870 | 68% |
| TwoRoom | | 0.5 | 0.5 | +0.748 | 65% |

Three readings, all against the hypothesis:

1. **V is not a staircase.** On PushT it is **98% monotone** inside a single
   primitive step. There is no gradient starvation from integer targets.
2. **Consecutive frames are well separated** (PushT ratio 0.68: one primitive
   step is two thirds of the whole 1-3 step goal distance), so the sub-step
   information was always present in the latent. The premise that the critic
   *cannot* resolve within a step is wrong.
3. **What compresses V near the goal is the OPTIMISTIC EXPECTILE.** tau 0.03
   recovers 43% of a step where tau 0.5 recovers 86% -- a clean 2x, and the
   same 2x on both environments.

And the proposed fix is actively harmful: subgrid cuts the per-step drop 3.4x
at tau 0.03 (0.429 -> 0.126) and 1.3x at tau 0.5, biasing V down rather than
adding resolution. Deleted, with its knob, sampler fields and plumbing; the
probe stays.

**Why this does not become an expectile arm.** The campaign already measured
that direction: E01 (teacher + co-critic expectile 0.1) scored 56.0 vs base
64.7, and EXPN (neutral expectile inside the last 3 steps, the arm closest to
what this probe says would help) was neutral. So a measurably better-resolved
critic does not transfer to the planner -- the third independent confirmation,
after E16 and E18, that critic quality is not the binding constraint. Cost of
closing this line: ~35 GPU-minutes and no training arms.

## E24 -- RLP-on-latent: the deficit is OPTIMIZER-generic, not objective-specific (2026-09-11)

The control that separates "the critic is a bad objective" from "the refiner is
a bad optimizer". Arm LATENT (`value.learner=l2`, `planner.freeze_critic_frac=0`,
w1): the deployed critic IS the single-frame latent distance, so the refiner
descends **exactly the objective CEM scores 79.3 with**. Tags
`pusht-latent-s{0,1,2}-20260911`, held-out draws 42/43/44.

| seed | rlp | cem_latent | cem_value (= the deployed critic) |
|---|---|---|---|
| 0 | 66.0 / 78.0 / 54.0 = **66.0** | 78.0 / 84.0 / 76.0 = 79.3 | 79.3 |
| 1 | 68.0 / 80.0 / 54.0 = **67.3** | 79.3 | 79.3 |
| 2 | 64.0 / 72.0 / 50.0 = **62.0** | 79.3 | 79.3 |
| **median** | **66.0** | **79.3** | **79.3** |

`cem_value` reproduces `cem_latent` to the digit on every draw, which is the
wiring check: with `learner=l2` the deployed critic and the latent cost are the
same function.

**Result: 66.0 against 79.3 on the identical objective, the identical world
model and the identical draws -- a 13.3-point optimizer gap.** Handing the
refiner the best objective the campaign knows does not help; it is *worse* than
the TD-critic refiner (W1NEAR_FRZ 70.7) and no better than the config-B base
(64.7).

The inversion is the interesting part. Measured as a **ranking** objective the
latent distance is the best available (CEM 79.3 vs 76.7 on the co-trained
critic) and the least hallucinated (E16: 5 px imagination gap vs 31 and 51).
Measured as a **descent** objective for the learned refiner it is the worst
tried: the refiner's gap to its own sampler widens from 6.0 (critic) to 13.3
(latent). A smooth, sharp cost is exactly what K=8 steps of learned gradient
descent through a differentiable world model can exploit; CEM cannot reach
those minima because its candidates come from a fixed Gaussian.

**What this closes.** Every critic-side arm in this campaign -- E10
augmentation, E12 expectile, E13/E14 relabeling and granularity, E17 contact,
E20 pessimism and capacity, E23 resolution -- was optimising a component that
is not the binding constraint. The binding constraint is the refiner's training
objective: `loss = e_path[-1] + mean_weight * mean(e_path)` over WM-imagined
rollouts, with no term asking whether the imagined outcome is achievable. An
action sequence that fools the world model is a global optimum of that loss.

**What remains.** Only levers that change the optimizer or the model it
descends through: Dyna on PushT (fine-tune the WM where the deployed planner
actually goes -- never run, and the mechanism this result points at),
descent/selection decoupling (score the K+1 iterates with a function the
refiner did not descend), and variance reduction on the seed lottery.

## E25 -- three probes into E1: the refiner is a working optimizer with a capped reach, and its corrections are chatter (2026-09-11)

Actor `pusht-w1near_frz-s0-20260909` (the best recipe), draws 42/43/44, 50
episodes. Jobs `rlp-pusht-p{1,2,3}-*`.

### Probe 2 -- the K ladder (job 22679)

| K | from zero init (`rlp_k`) | from the CEM plan (`rlp_ceminit_k`) |
|---|---|---|
| 0 | 1.3 | 76.7 |
| 1 | 4.7 | 74.0 |
| 2 | 18.0 | 77.3 |
| 4 | 48.0 | 76.7 |
| 8 | **72.7** | **77.3** |

Two readings, and the first corrects the campaign's working story:

1. **From zero the refiner is essential and strongly monotone in K**: 1.3 (the
   noop floor) -> 72.7. It is a working optimizer, not a broken one.
2. **From the CEM plan refinement is FLAT within noise** (76.7 -> 77.3), not
   monotonically destructive.

### Probe 1 -- learned rule vs plain gradient (job 22681)

| condition | mean | vs its own control |
|---|---|---|
| `rlp_ceminit_k0` | 78.7 | -- |
| `gd_ceminit_k8` lr 0.01 | 77.3 | -1.4 |
| `gd_ceminit_k8` lr 0.03 | 76.0 | -2.7 |
| `rlp_ceminit_k8` | 75.3 | -3.4 |
| `gd_ceminit_k8` lr 0.1 | 58.7 | **-20.0** |

Plain gradient descent degrades the CEM plan too, in proportion to step size,
so the landscape punishes large moves regardless of who makes them. But note
the **measurement-noise caveat**: `rlp_ceminit_k0` scored 78.7 here and 76.7 in
job 22679 -- the same condition, actor and draws, 2 points apart. At 150
episodes and p ~ 0.77 the binomial sd is ~3.4 points, so the k0-vs-k8 deltas in
this table (-3.4) and in probe 2 (+0.6) are both inside one sd and disagree in
sign. **The honest statement is that refinement from a good plan is flat, not
that it destroys it.** E1's -10.7 was measured on the config-B BASE actor and
with a paired per-episode ledger (fixes 55, loses 76), which is far more
sensitive than these aggregates; the better critic has evidently shrunk the
effect, but only the paired analysis can resolve its sign now. Only the lr 0.1
collapse (-20.0) is safely outside noise.

### Probe 3 -- what the refiner changes (job 22680)

Per-replan decomposition of (refined - init), ~70 replans per draw:

| plans built | delta norm / init norm | late/early | **high/low DCT power** | frac at clip |
|---|---|---|---|---|
| from zero (`rlp`) | 4.9 / 0.0 | 1.21-1.27 | **0.45 - 0.62** | 0.003 |
| from the CEM plan (`rlp_ceminit`) | 3.1 / 7.0 | 1.07-1.16 | **1.66 - 1.94** | 0.023 - 0.044 |

**The refiner's own plans are smooth and its corrections are chatter.** Building
from scratch it emits a low-frequency action sequence (high/low power ~0.5),
which is what a real trajectory looks like. Handed a CEM plan it applies a
correction that is high-frequency dominated (~1.9), perturbs ~45% of the plan's
norm, and hits the action clip 10x more often -- and buys nothing (probe 2).
Horizon position is nearly flat (late/early ~1.1), so this is a temporal-BAND
effect, not a late-plan effect: E16's late-plan fabrication does not show up in
where the update is spent.

### What this changes

The story is no longer "the refiner destroys good plans". It is:

* the refiner **reaches ~72.7 from zero** and **cannot improve a 77 plan**, so
  its reachable plan quality is capped below CEM's rather than actively
  harmful;
* the budget it spends trying is mostly high-frequency chatter that no real
  action sequence contains.

That is a concrete, cheap lever that touches neither the objective nor the
world model: **band-limit the update** -- project each refinement step onto the
low-frequency DCT modes its own from-scratch plans already occupy. It is not a
behaviour-cloning anchor (no data, no imitation), just a bandwidth constraint.

Still open and one eval away: `gd_k8` from ZERO init, i.e. whether the learned
rule beats plain descent in the deployed setting. The ladder says the rule is
doing real work; it does not say the learned part is what does it.

## E26 -- band-limiting is negative; plain descent from zero is at the floor (2026-09-11)

Job `rlp-pusht-bandlimit` (22850), actor `pusht-w1near_frz-s0-20260909`, all ten
conditions in ONE job so every comparison is within-job. `bl<N>` keeps the first
N of 5 temporal DCT modes of each refinement step.

| condition | 42 / 43 / 44 | mean |
|---|---|---|
| `rlp` (baseline) | 74 / 84 / 60 | **72.7** |
| `rlp_bl3` | 76 / 84 / 54 | 71.3 |
| `rlp_bl2` | 66 / 70 / 58 | 64.7 |
| `rlp_bl1` (DC only) | 34 / 44 / 20 | 32.7 |
| `gd_k8` lr 0.03 (plain descent, zero init) | 10 / 8 / 0 | **6.0** |
| `rlp_ceminit_k0` | 78 / 82 / 74 | 78.0 |
| `rlp_ceminit_k8` | 74 / 86 / 72 | 77.3 |
| `rlp_ceminit_bl3` | 74 / 84 / 72 | 76.7 |
| `rlp_ceminit_bl2` | 78 / 80 / 70 | 76.0 |
| `rlp_ceminit_bl1` | 72 / 78 / 74 | 74.7 |

**Band-limiting closed negative**, monotone in how much bandwidth is removed:
from zero init -1.4 / -8.0 / -40.0 for N = 3 / 2 / 1. E25's measurement stands
(corrections to a good plan ARE chatter-dominated) but the inference drawn from
it does not: the high-frequency content is not spare capacity, it is load
bearing. The asymmetry is informative -- removing bandwidth costs 40 points when
the refiner must CONSTRUCT a plan from zero and only 3 when it is correcting a
good one, which is what one would expect if the modes are doing real work rather
than exploiting.

**`gd_k8` from zero init scores 6.0** against the learned rule's 72.7 and a noop
floor near 1. This is the campaign's first direct measurement of what the
LEARNED part of the refiner contributes, and it is nearly all of it. Caveat
before quoting: lr 0.03 was chosen because it was harmful from a CEM init, and
from zero the plan must travel to norm ~5, which 8 steps at that rate may simply
not reach -- a fair version needs an lr sweep from zero (0.1 / 0.3 / 1.0).

**Reproducibility, localised.** `rlp` here is 74/84/60 = 72.7, identical to
`rlp_k8` in job 22679 and consistent with the campaign's 72.0 for this actor.
The ~2-point wobble noted in E25 is confined to the `init_mode=cem` conditions
(`rlp_ceminit_k0`: 78.7 / 76.7 / 78.0 across three jobs), which is expected --
that init runs a stochastic CEM search. Zero-init conditions reproduce exactly,
so within-job zero-init comparisons need no noise allowance and CEM-init ones
need ~2 points.

**Draw 44 is where the gap lives.** Per draw, RLP vs CEM-latent: 74 vs 78 (-4),
84 vs 84 (0), 60 vs 76 (-16). RLP ties CEM on draw 43 and the entire 6.6-point
mean deficit is draw 44 -- so the target is not a uniform shortfall but one
draw where the refiner collapses. E22's per-episode attribution was run on draw
43, the draw where RLP is already at parity; redoing it on draw 44 is the
better-aimed diagnostic.

## E27 -- six seeds, and the first data-volume ladder in the campaign (2026-09-12)

Training seeds 0-5, eval draws 42/43/44 x 50 episodes, the protocol the other
environments got. Seeds 0-2 are the 2026-09-09 / 2026-09-02 runs; 3-5 are new.

**Poolability check first.** `w1near_frz-d16000-s0` retrains the seed-0 recipe
under post-cleanup code at full data and returns 74.0 / 84.0 / 60.0 -- the same
three draws as the 2026-09-09 actor, as probe 2's `rlp_k8`, and as E26's `rlp`.
Training is reproducible across the refactor, so the two seed generations pool.
(Those draws mean 72.7, where the E22-era table recorded seed 0 as 72.0 -- one
episode in 150; three current measurements agree on 72.7 and the n=6 median is
69.0 either way.)

### Six seeds

| arm | s0 | s1 | s2 | s3 | s4 | s5 | n=3 median | **n=6 median** | spread |
|---|---|---|---|---|---|---|---|---|---|
| W1NEAR_FRZ | 72.7 | 70.7 | 67.3 | 66.0 | 68.0 | 70.0 | 70.7 | **69.0** | 6.7 |
| base (config B) | 68.0 | 64.0 | 64.7 | 66.0 | 61.3 | 65.3 | 64.7 | **65.0** | 6.7 |

- **The recipe's advantage shrinks from +6.0 to +4.0.** Three of the three new
  W1NEAR_FRZ seeds (66.0 / 68.0 / 70.0) land at or below the old worst seed, so
  n=3 was flattering it; the base is unchanged (64.7 -> 65.0).
- **The gap to latent-CEM widens from -8.6 to -10.3** (79.3).
- Seed spread is 6.7 points on BOTH arms -- larger than every critic-side effect
  this campaign chased after E12. Any future arm reported at n=3 should be read
  with that in mind.

### Data-volume ladder (W1NEAR_FRZ, seed 0, shared 16k cache capped at [0, N))

| train episodes | rlp | delta per doubling |
|---|---|---|
| 2000 | 59.3 | -- |
| 4000 | 63.3 | +4.0 |
| 8000 | 70.0 | +6.7 |
| 16000 | **72.7** | +2.7 |

**Monotone and NOT saturated.** 8x the data buys +13.4 points, and the last
doubling is still positive. This was never measured before -- the campaign
swept step budget (2x budget: -12) and read it as "more optimisation overfits",
but data volume itself was pinned at 16000 throughout and is the largest single
lever measured on PushT to date, larger than every critic intervention
(near-goal oversampling +4 being the best of them).

Caveat: the ladder is n=1 and the seed spread is 6.7, so the top-rung
deceleration (+2.7) is inside noise and the curve's shape at the top is not
resolved. The trend across the full range (+13.4) is far outside it.

**What this reframes.** PushT's dataset holds 18,685 episodes with 16000:18685
reserved for evaluation, so the training set cannot grow without touching the
held-out range -- this ladder is at the ceiling of the available data. The
finding is therefore not "add more data" but "the refiner is still
data-limited at the data we have", which makes data efficiency, not critic
quality, the live axis. Dyna is the one lever that manufactures more on-policy
data without touching the eval range, and it remains unrun on PushT.

## E28 -- 5x the actor's data changes nothing: the teacher is the data-limited learner (2026-09-12)

E27's ladder capped BOTH learners and found +13.4 over 8x data. This isolates
the actor: the fs5 cache kept only steps 0, 5, 10, ... of each episode; keeping
all five residue classes of the stride (`actor_phases=5`) gives the actor 5x
the block-aligned start states, goals and histories from the same 16k episodes,
while the teacher -- already on fs1, every step -- is untouched. Same recipe,
same three seeds, so the read is per-seed paired.

Verified in-job: `fs5 cache (phases=5): 2002226 latents, 80000 episodes`
(= every fs1 row; 16k x 5 phase-episodes, median 25 blocks) and `LIP-AC actor
cache is phase-multiplexed x5` on all three runs.

| seed | stride-0 (E27) | all-phase (p5) | delta |
|---|---|---|---|
| s0 | 72.7 | 72.0 / 90.0 / 54.0 = **72.0** | -0.7 |
| s1 | 70.7 | 74.0 / 74.0 / 62.0 = **70.0** | -0.7 |
| s2 | 67.3 | 76.0 / 80.0 / 54.0 = **70.0** | +2.7 |
| median | 70.7 | **70.0** | -0.7 |

Critic-as-sampler-objective is unchanged (cem_value 78.0 / 78.0 / 76.0,
cem_tdvalue 78.0 / 76.7 / 76.7), as it must be -- the teacher saw the same data.

**Null.** Five times the actor's start states moves nothing outside seed noise.
Since capping both learners cost 13.4 points and feeding only the actor gains
nothing, the data sensitivity E27 measured lives in the TD TEACHER. Caveat on
the strength of that inference: adjacent phases are one primitive step apart,
so this is denser sampling of the same trajectories, not new episodes -- it
shows the actor does not want more coverage of the data it has, which is
consistent with it already sitting at ~4 passes with 2x budget on record as
harmful. The clean confirmation is the 2x2's off-diagonal: teacher at 2k with
the actor at 16k, and the reverse. Two jobs.

Side observation: s0 draw 43 = 90.0 is the highest single-draw RLP number in the
campaign (stride-0 s0 was 84 there), offset by 54 on draw 44 (was 60) -- a
redistribution across draws, not a gain, and the same draw-44 collapse E26
identified as where the whole RLP-CEM gap lives.

The all-phase cache stays available (`actor_phases`, separate artefact); it
costs nothing at eval and may matter for recipes that are actor-limited.

## E29 -- the deploy-time history mismatch is real and irrelevant (2026-09-12)

Code review found that at every replan the LIP solver tiled the current frame
three times and zeroed the two past action blocks: the WM was asked "three
identical frames, ten idle steps" at the moment the agent had just been
pushing, an input neither the WM's pretraining nor the actor's training ever
contained (training: real histories for 50% of samples, imagined moving frames
with zero actions for the replay 50%). `planning.history_len` existed in the
eval config but the solver never read the history the policy published.

Correction along the way: rlp.core.policy.WorldModelPolicy has had a lagged
history mechanism all along -- frames one action block apart in `pixels_hist`,
the last ten executed primitive actions in `action_hist`, already in the
actor's normalised action space. The dead link was the CONSUMER. The solver now
prefers `pixels_hist` when present (planning.history_len=3) and can take
`action_hist` as-is (core.solver.use_action_history). A first attempt added a
duplicate World-side buffer; removed once the policy's was found.

Eval-only, all six W1NEAR_FRZ seeds, draws 42/43/44, paired per task against
the history_len=1 arrays. At h25 / budget 50 there are two decisions per
episode; the first has no past, so only the second is affected.

| regime | per-seed delta (s0..s5) | median | fixes / breaks over 900 tasks | sign test |
|---|---|---|---|---|
| real lagged frames, zero actions | -2.7 +0.7 0.0 +0.7 +0.7 +1.3 | 69.0 -> 69.3 | **10 / 9** (net +1) | p = 1.00 |
| real frames AND real actions | -1.3 +0.7 -2.7 -0.7 +1.3 -1.3 | 69.0 -> 69.0 | **10 / 16** (net -6) | p = 0.33 |

Every job logged the regime switch (`action history zeros` on the first plan,
`real` on the second), so the coherent frames+actions regime was genuinely
exercised. **Both null.** Handing the WM its true recent past, with or without
the actions, changes nothing outside seed noise; the frames+actions arm leans
slightly negative but not significantly. The mismatch was worth finding and
fixing for correctness -- planning.history_len now does what its name says --
but the planner's inputs were never the bottleneck. Consistent with E24/E25:
the problem is what the WM's derivative says near contact, not what the WM is
shown.

## E30a -- does the world model disagree with itself where it hallucinates? (2026-09-12)

Precheck for pessimism-by-disagreement (E30b, training in flight). The LeWM
predictor carries dropout 0.1, so a free ensemble is K stochastic passes of the
one WM. `scripts/pusht_diag/mc_disagreement.py` (job rlp-pusht-mcd43, in the
E21 directory) re-imagines every fully executed first plan of draw 43 once
deterministically and K=8 times with dropout on, then relates the spread to:
hallucination = ||decoded imagined terminal block - REAL terminal block (exact
pose)||; the V optimism gap; whether the block was actually pushed (> 10 px);
and the minimum agent-block gap. n = 22 RLP plans, 15 CEM-latent (short
successful episodes have no fully executed plan and are excluded).

| | RLP | CEM-latent |
|---|---|---|
| Spearman(block spread, hallucination) | **0.53** | 0.36 |
| AUC(block spread -> hallucination > 20 px) | **0.80** | 0.72 |
| AUC(V spread -> hallucination > 20 px) | 0.62 | 0.60 |
| Spearman(raw latent spread, hallucination) | -0.03 | 0.40 |
| AUC(block spread -> block NOT pushed) | **0.51** | 0.28 |
| block spread, not pushed / pushed (px) | 4.7 / 5.0 | 3.6 / 5.4 |
| hallucination, not pushed / pushed (px) | **36 / 22** | 19 / 18 |

Readings:
1. **Disagreement is a real predictor of hallucination magnitude** (AUC 0.80 in
   the decoded block channel). Unlike E7's critic ensembles, the world model
   does not fully agree with itself where it is wrong. Pessimism has something
   to grip.
2. **It does not mark the fake basin specifically.** RLP's not-pushed plans --
   imagined motion, no real push, the E1/E22 signature -- hallucinate 36 px
   against 22 px for pushed plans, yet their spread is flat (AUC 0.51). The
   spread tracks how far the imagination is off, not whether contact happened.
3. **Where the signal lives matters for the training arm.** Raw latent spread
   is uninformative (-0.03); spread in the decoded block position is the
   strong signal; spread in V sits between (0.62). E30b penalises std of V --
   the middle-strength quantity -- because the actor loss is E and a state
   probe is not available at training time. If E30b is weak, the upgrade is a
   penalty on the spread of the imagined terminal latent along the critic's
   sensitive directions, or a larger K / beta; if it is null, this table says
   why: the basin is a bias all dropout masks share, not a disagreement.

Caveats: n is small (22/15) and hallucination below ~20 px is at the ridge
probe's own error floor (median block decode error 11 px), so the > 20 px
threshold is the meaningful one. Per-plan data:
docs/figures/pusht_diag/mcd/mc_disagreement*.json (per draw).

### E30a, pooled over draws 42/43/44 -- the draw-43 signal does not replicate

Job rlp-pusht-mcd4244 recorded draws 42 and 44 on the same actor and ran the
probe per draw (n = 28 / 32 RLP, 16 / 17 CEM). Pooled with draw 43:

| | RLP (n=82) | CEM-latent (n=48) |
|---|---|---|
| AUC(block spread -> hallucination > 20 px) | 0.69 | 0.65 |
| AUC(V spread -> hallucination > 20 px) | **0.61** | 0.53 |
| **AUC(block spread -> block NOT pushed)** | **0.39** | **0.25** |
| block spread, not pushed / pushed (px) | 4.4 / 5.4 | 4.4 / 6.9 |
| hallucination, not pushed / pushed (px) | 26 / 30 | 29 / 32 |

Per draw the informative channel wandered (block on 43, V on 42, neither on
44), and the pooled predictor of hallucination is weak: 0.69 in the block
channel, **0.61 in V -- the quantity E30b penalises**. Draw 43's 0.80 was a
small-n outlier.

The one number stable across all three draws and both planners is the
inversion: **the model is MORE confident where the block was not pushed** (AUC
0.39 / 0.25; spread 4.4 px without contact vs 5.4-6.9 with it). Dropout
uncertainty rises during real contact, where the dynamics are genuinely hard,
and falls in the hover-near-the-block regime where the fabrication lives. This
is E7's finding for the world model: **the fake basin is a shared, confident
bias, not a disagreement.** Pessimism-by-disagreement therefore cannot mark it,
and taxing the spread penalises real pushes MORE than fabricated ones.

Consequences: (i) the running E30b arm (mean + std of V over K=4 dropout
passes) is predicted null or slightly negative; the timing guard shows it is
genuinely executing (1000 actor steps in 10.7 min vs 8.5 baseline, x1.3 --
consistent with four extra rollouts of the final plan per step). (ii) A
sharper variant (block-channel or critic-direction spread, larger K or beta)
is NOT worth building: the sign of the effect is wrong, not its strength.
(iii) What remains for the basin is changing the model where it is confidently
wrong -- on-policy data from exactly those hover rollouts (Dyna) -- or a
planner that does not trust the model's derivative there.

## E30b -- pessimism at training time: null, as the pooled probe predicted (2026-09-13)

Arm W1NEAR_FRZ_MCP: the final iterate re-imagined K=4 times with the LeWM
predictor's dropout (0.1) active, actor loss = mean + 1.0 * std over the passes,
inputs deterministic, deployed solver unchanged. Three seeds, paired per task
against the same seeds of W1NEAR_FRZ on draws 42/43/44.

| seed | W1NEAR_FRZ -> MCP | fixes / breaks | per draw |
|---|---|---|---|
| s0 | 72.7 -> 72.0 (-0.7) | 6 / 7 | 74->72, 84->86, 60->58 |
| s1 | 70.7 -> 69.3 (-1.3) | 10 / 12 | 76->74, 72->80, 64->54 |
| s2 | 67.3 -> 66.7 (-0.7) | 12 / 13 | 66->70, 80->70, 56->60 |
| median | **70.7 -> 69.3 (-1.4)** | **28 / 32** of 450 (net -4) | sign test p = 0.70 |

Every seed slightly negative, none outside noise. The timing guard (x1.3 per
actor step) confirms the pessimistic loss was executing. Note the churn: 13-25
task flips per seed against 1-6 for the E29 deploy-time arms -- retraining
reshuffles ~4% of tasks either way (the seed lottery), the net is zero.

Reading: consistent with E30a pooled. The world model is MORE confident where it
fabricates (block not pushed: spread AUC 0.39), so a disagreement penalty taxes
real pushes at least as much as the fake basin and has nothing to push against.
Pessimism-by-self-disagreement is closed; a sharper spread (block channel,
critic direction, larger K/beta) would have the wrong sign, not too little
strength. `pess_passes`/`pess_weight` removed with this entry; the probe
(scripts/pusht_diag/mc_disagreement.py) and per-draw data stay.

What the E23-E30 sequence leaves standing for PushT's basin: the model must be
corrected where it is confidently wrong, with data from exactly the hover
rollouts that fool it (Dyna), or the planner must stop trusting the model's
derivative there. Nothing on the critic, the inputs, the bandwidth, or the
model's own uncertainty moves it.

## E31 -- watching the failures: rendered draw-43 episodes with the imagined block overlaid (2026-09-13)

Ten draw-43 tasks rendered for the W1NEAR_FRZ seed-0 actor
(`scripts/pusht_diag/failure_videos.py`, job `rlp-pusht-vid43`): the seven
RLP failures E22 triaged as planner (0, 8, 9, 15, 25), critic (31, 43) or
world-model (3), plus two controls (1 both solve, 2 RLP solves and CEM does
not). Each MP4 is three panels: the RLP rollout with the world model's
re-imagination of the executed plan decoded to a block centre (pink ring)
next to the true centre (yellow), CEM-latent on the same task, and the goal.
The RLP caption prints the critic's value of the imagined and of the real
latent for the current plan block. Assets: `docs/figures/pusht_diag/videos43/`
(MP4 + GIF per task, `index.html` contact sheet).

Budget is 50 steps = two 25-step plans, so `t=25` is the end of the first
plan and the one replan.

| task | E22 label | RLP V imag / real at t=25 | at t=49 | what the frames show |
|---|---|---|---|---|
| 0 | planner | **1.3 / 6.5** | 3.4 / 3.4 | WM reports the first plan as essentially solved; real block far short. Second plan lands the block on the footprint rotated ~15-20 deg, agent parked at the left tip |
| 8 | planner | **4.7 / 10.0** | **7.2 / 9.7** | imagined centre sits at the T's tip, real centre at the crossbar: WM imagines translation while the agent is rotating the T about its tip. Hallucination persists through both plans |
| 25 | planner | 3.2 / 5.3 | 4.4 / 4.4 | second plan flat at 4.4-4.5 from t=29; block against the bottom arena edge, agent hovering to its right. No hallucination -- a real stall |
| 9 | planner | 3.6 / 3.7 | 0.9 / 1.4 | block visually on the goal, agent already withdrawn; tolerance near-miss |
| 15 | planner | 2.8 / 3.0 | 2.7 / 2.2 | second plan makes no progress; block on the footprint with an angle residual, agent sitting at the stem |
| 3 | world model | 8.3 / 8.9 | 12.0 / 12.1 | value RISES through the episode: block pushed away from the goal. CEM fails too |
| 31 | critic | 6.0 / 6.6 | 1.4 / 2.8 | angle residual; imagined centre offset from the real one. CEM fails too |
| 43 | critic | 6.2 / 6.3 | 3.5 / 3.7 | flat since t=29, near-miss. CEM fails too |
| 1 | control | RLP solved at t=23 (1.0 / 0.9) | -- | CEM solves at t=47 |
| 2 | control | RLP solved at t=44 (3.3 / 3.7) | -- | CEM fails |

Two visible modes, often both in one episode:

- **First-plan hallucination** (tasks 0, 8, 25). At the replan the world
  model's value of its own imagined outcome is 2-5 units better than the
  value of the real latent: 1.3 vs 6.5 on task 0 is "solved" vs "not
  started". Half the budget is spent on a plan the model believed in. This
  is E16's block hallucination seen per episode with the deployed critic as
  the yardstick.
- **Second-plan stall** (tasks 0, 15, 25, 43; task 9 is the tolerance edge).
  Imagined and real values agree, the value is flat at 2.5-4.5 for the last
  twenty steps, the block is on or beside the footprint with an angle
  residual, and the agent parks at a tip or the stem. The model is not lying
  here; the gradient through it simply does not produce a plan that moves the
  real value.

In every planner-labelled task CEM-latent finishes inside ITS first 25-step
plan (t = 22-24) with the same critic. The failure is the gradient path
through the world model, not the objective (E24) and not the critic (E23):
sampling the same model finds the pushing plan; following its derivative
does not. Nothing in the frames suggests a data-volume story -- the
hallucinated block motion and the parked agent are model-side. Consistent
with E27/E28: more TD-teacher data buys a couple of points; the mechanism on
screen is the model at contact.

## E32 -- physics grounding of the energy: E16's untried consequence is a null, by mechanism (2026-09-13)

The world model stays frozen (user decision 2026-09-13: improve RLP only).
E16 ended with the one lever that follows from its mechanism and had never
been built: a plan-time penalty on imagined block displacement that no agent
contact supports, computable from the imagined latents alone. E31's videos put
half the RLP-only failures in exactly that class (first-plan hallucination,
V imagined 1.3 vs real 6.5 on task 0).

### The term (`src/rlp/core/grounding.py`)

Everything is a fixed linear read plus a few dozen scalar ops; nothing is
sampled and no second model is run. Training and deploy use the identical
term (it travels in the actor checkpoint), so the refiner's energy and
gradient inputs are grounded too.

1. **State probe.** Ridge regression latent -> (agent xy, block xy, cos, sin)
   fitted on 200k rows of the fs1 cache against the dataset's `state` column
   (E16's probe, now an exported (193 x 6) matrix).
2. **Agent path from the plan.** PushT's action is a relative position command
   (target = position + action x 100 px, PD-tracked over ten 10 ms substeps),
   so the commanded path is known exactly from the plan and the real agent
   position (`proprio`, raw px). A two-tap gain
   dp_t = g0 u_t + g1 u_{t-1} fitted on the data absorbs the controller lag.
3. **Contact test.** Distance from every commanded agent position to the T
   (two rectangles in the block body frame, `add_tee` geometry, verified
   against pymunk's own point query in the unit tests) minus the agent radius,
   against the block pose decoded at the start of each block. Soft contact
   c_k = sigmoid((margin - gap_k) / tau); "no contact yet" through block k is
   the cumulative product of (1 - c_j).
4. **Penalty.** sum_k (no contact yet)_k x relu(|b_k - b_0| - deadzone)^2 / ref^2,
   with b the decoded block position and b_0 its value at the real start
   frame; times `ground_weight` into the energy.

### Calibration, in-job, on the 16k-episode cache (smoke job 23279, seed 0)

| quantity | value |
|---|---|
| probe held-out R2 agent / block / angle | 0.949 / 0.971 / 0.893 |
| probe median error agent / block | 21.2 px / 10.9 px |
| controller gains g0 / g1 (fit R2) | 0.296 / 0.111 (1.000) |
| true steps in contact | 42.3 % |
| block moves given contact / given agent > 10 px clear | 60.7 % / **0.06 %** |
| contact margin (calibrated) | 11.0 px; 99.8 % of true pushes register |
| dead zone (calibrated) | 22.9 px (static-block decoded displacement: mean 11.7, P95 22.9) |
| decoded vs true displacement on push windows | 74.6 px vs 75.3 px |
| static windows with the agent hovering < 60 px that the test flags | 62.2 % (n = 5,227) |

The geometry explains the data (the block essentially never moves without
contact under the reconstructed T), real pushes are not penalised, the probe
tracks true motion to within a pixel on average, and the test keeps power on
the hover states E31 showed. Note the controller lag: one command lands ~30 %
in its own step and ~11 % in the next, so the commanded path is far shorter
than the summed commands -- the kinematic fit, not the raw command, is what
the contact test must see. At training step 0 (zero-init plans) the term
reads 0.023 energy units, decoded end displacement 16.8 px, 31 % of samples
without contact by plan end.

The probe's block error is ~11 px, so fixed thresholds would misread real
pushes. Both thresholds come from the data: the contact `margin` is the 95th
percentile of the closest approach the penalty itself sees on true-push
windows (block moved > 30 px over 25 steps) plus 2.2 tau, so 95 % of real
pushes register contact >= 0.9; the `deadzone` is the 95th percentile of
decoded displacement over 25 steps on windows where the true block is static
(the probe's noise floor). A geometry self-check on true states reports how
often the block moves with the agent > 10 px clear of the T (must be rare),
and the test's power is reported as the share of static windows with the
agent hovering within 60 px that the test still flags.

Arms: W1NEAR_FRZ + `GROUND=1.0` and `GROUND=0.3`, seeds 0-2, draws 42/43/44;
`rlp_ng` deploys the grounded actor with the term switched off
(training-time-only ablation); `cem_value` checks the teacher is unchanged.
Read: paired per-task ledgers against the six-seed history-1 arrays.

### First seed and the deploy-time rescoring: the term is blind by construction

`w1near_frz-g1.0-s1`: rlp 76 / 76 / 44 = **65.3** (term on) and 76 / 72 / 46 =
64.7 with the term switched off at deploy (`rlp_ng`); baseline seed 1 is
76 / 72 / 64 = 70.7; `cem_value` 78 / 80 / 78 = 78.7, the teacher unchanged.
The draw-44 drop is seed lottery, not the term: through the whole run the
grounding term contributed 0.005-0.018 energy units against a plan energy of
~7, i.e. training was effectively the baseline recipe with a different random
trajectory.

Why it never fired (`scripts/pusht_diag/grounding_terms.py`, job 23293: the
seed-1 actor re-evaluated on draws 43/44 with probe dumps, every deployed
decision rescored and joined with its outcome):

| draw, decision | fails imagining > 30 px of block motion | of which no-contact (term sees) | under contact (term blind) | median closest approach, fails |
|---|---|---|---|---|
| 43, first plan | 5 / 12 | **0** | 5 | -10.9 px (inside the T) |
| 44, first plan | 15 / 29 | **0** | 15 | -15.0 px |
| 43, second plan | 0 / 7 | 0 | 0 | -4.6 px |
| 44, second plan | 3 / 20 | 0 | 3 | -1.8 px |

**The refiner's plans DO put the agent on the block.** Every failed first plan
with fabricated motion has the commanded path in contact with the decoded T;
the fiction is the motion contact produces (E31 task 8: rotating the tip,
imagining translation), not motion without contact. A penalty on
unsupported motion therefore has nothing to penalise, and no threshold
rescues it: rescoring the same decisions with dead zones down to 5 px and
margins down to 0 leaves the share of failures above the successes' 90th
percentile at 0-29 % (chance 10 %), and at tight settings the successes pay
more than the failures. Second plans are the stall class from E31: failures
imagine 12-16 px of motion, the model is honest and the plan passive.

E16's consequence, as literally written, is closed. The remaining question is
whether the under-contact fiction violates a pushing law the same probe can
check (block outrunning the agent, moving against the push, moving towards
the agent): `scripts/pusht_diag/grounding_laws.py`, next.

### The under-contact fiction obeys quasi-static pushing in imagination (job 23294)

Same decisions, rescored per block against the laws a pushed block must obey,
with the decoded block displacement db and the kinematic agent displacement dp
over each block (`scripts/pusht_diag/grounding_laws.py`; score = worst block;
"sep" = share of failures above the successes' 90th percentile, chance 10 %):

| draw, first plan | imagined block path fail / succ | agent path fail / succ | ratio |db|/(|dp|+5) sep | against-the-push sep | towards-the-agent sep | spin sep |
|---|---|---|---|---|---|---|---|
| 43 | 53.3 / 56.5 px | 179 / 154 px | 8 % | 17 % | 0 % | 17 % |
| 44 | 67.3 / 72.1 px | 203 / 182 px | 14 % | 3 % | 7 % | 14 % |

Failing first plans imagine slightly LESS block motion than successful ones,
move the agent more, and never let the block outrun the agent (ratio medians
0.5-0.8 for both classes); direction and side of the push are as plausible on
failures as on successes; imagined rotation is the same. Second plans are
the same picture at smaller magnitudes (one 43 % cell on 7 failures is
noise). At the 11 px resolution of a linear probe the refiner's imagined
plans are physically plausible pushes -- the model's error is the response
of the T to a plausible push, which a decoded-position prior cannot
adjudicate.

**E32 verdict (mechanism; aggregate over six jobs to follow).** Grounding the
energy with physics priors read from imagined latents is closed on PushT:
(i) the refiner never fabricates motion without contact, so E16's proposed
penalty has nothing to act on (0.01 energy units in training, no separation
at deploy at any threshold); (ii) under contact, the fabricated motion
satisfies the quasi-static pushing laws. What separates fabricated from real
pushes is not in the imagined positions; it is in the real outcome. The lever
this leaves on the RLP side with a frozen model is a discrepancy detector
trained on real rollouts (E32's arm 2 in the 2026-09-13 proposal), not a
prior.

### Aggregate: six jobs, paired against the same-seed baselines

W1NEAR_FRZ recipe, seeds 0-2, draws 42/43/44 x 50; `rlp` = term on at deploy,
`rlp_ng` = the same actor with the term off, `cem_value` = the teacher as CEM's
objective (unchanged, 75-79). Jobs 23280-23285.

| arm | seed | baseline | rlp (42/43/44) | rlp_ng | cem_value | ground term in training (median / max) |
|---|---|---|---|---|---|---|
| lambda 1.0 | 0 | 72.7 | 68/78/54 = 66.7 | 65.3 | 78.7 | 0.010 / 0.039 |
| lambda 1.0 | 1 | 70.7 | 76/76/44 = 65.3 | 64.7 | 78.7 | 0.011 / 0.018 |
| lambda 1.0 | 2 | 67.3 | 78/74/54 = 68.7 | 66.7 | 76.0 | 0.010 / 0.053 |
| **lambda 1.0 median** | | **70.7** | **66.7** | 65.3 | 78.7 | |
| lambda 0.3 | 0 | 72.7 | 74/80/62 = 72.0 | 72.7 | 78.0 | 0.007 / 0.013 |
| lambda 0.3 | 1 | 70.7 | 72/72/64 = 69.3 | 70.0 | 75.3 | 0.008 / 0.013 |
| lambda 0.3 | 2 | 67.3 | 68/78/52 = 66.0 | 64.7 | 76.7 | 0.005 / 0.022 |
| **lambda 0.3 median** | | **70.7** | **69.3** | 70.0 | 76.7 | |

Paired per-task ledgers against the six-seed history-1 arrays (150 tasks per
seed): lambda 1.0 fixes 33 / breaks 48 (sign test p = 0.12), its no-term deploy
28 / 49 (p = 0.02); lambda 0.3 fixes 27 / breaks 32 (p = 0.60), no-term deploy
28 / 33 (p = 0.61). Switching the term off at deploy changes at most one
episode in 150 per seed on average -- as it must for a term worth 0.01 units.

**Closed: null at lambda 0.3, null-to-negative at lambda 1.0, and the two
rescorings say why.** The penalty had nothing to act on (no fabricated motion
without contact in any failed plan) and the under-contact fiction obeys the
pushing laws a decoded-position prior can check. The negative lean at
lambda 1.0 is three fresh lottery tickets (energy traces and validation
values track the baselines); n = 3 cannot separate it from noise and the
mechanism gives no reason to expect a real effect either way. Code stays
(`planner.grounding`, off by default; `rlp_ng` eval condition), since the
probe, kinematics and rescoring scripts are the tooling any outcome-based
detector will reuse.

## E33 -- an offline block-level discrepancy model: the data does not know where the model lies for the refiner's plans (2026-09-13)

E32 located the fiction: physically plausible pushes whose real response the
frozen model gets wrong, at the level of a single five-step action block. The
question this entry asks is whether that response error is *learnable from the
logged data alone* -- no environment interaction, no world-model retraining --
because if it is, a learned correction can enter the refiner's energy and
bypass the one derivative the model gets wrong.

### D (`scripts/pusht_diag/discrepancy_fit.py`)

Every 25-step window of training episodes 0-15999 (300k sampled of ~1.3M) is
re-imagined through the frozen model from its REAL three-frame history,
free-running over the five blocks (so blocks 2-5 start from imagined states, as
at deploy). The target for block k is the decoded block pose of the real frame
at the block's end minus the decoded pose of the imagined one, in the body
frame of the block's start pose: position (2) and angle (1). Same probe on both
sides, so probe noise largely cancels and the residual is the latent fiction
as the probe sees it. Features per block, all differentiable w.r.t. the plan at
deploy: decoded imagined pose at block start, the agent's position relative to
the block in the body frame (real at the plan start via proprio, kinematic
thereafter), the five commanded displacements in the body frame, the imagined
block motion over the block, the closest kinematic approach to the T, and the
end gap (21 features); variant `geolat` adds the imagined latent (192).
MLP 3 x 256 SiLU, Huber loss, AdamW, 8 epochs; validation = episodes 14000-15999.

### The test

D is then evaluated where it has to work: along the refiner's own imagined
trajectories on the recorded E32 decisions (seed-1 grounded actor, draws 43/44,
probe dumps with start latent, imagined trajectory, plan, agent anchor). Per
decision: the predicted fiction magnitude summed over blocks, and the terminal
block error to the decoded goal after adding D's correction, each compared
between failed and successful episodes (AUC and share of failures above the
successes' P90; the critic's own energy and the uncorrected terminal error as
references). Separation here, which no physics prior achieved, is the
go/no-go for wiring D into the training energy.

### Fit (job 23317 / 23326, 300k windows = 1.5M block samples, validation = episodes 14000-15999)

| real minus imagined block motion | value |
|---|---|
| per block, before D | 9.9 px mean; 7.5 px at block 1 rising to 12.4 px at block 5 (free-running compounding) |
| block 1, contact vs no-contact blocks | 7.3 px vs 7.8 px -- **no contact dependence** |
| after D, `geo` / `geolat` | 6.9 px / 6.5 px |
| variance explained, position / angle | 0.35 / 0.11 (`geo`), 0.37 / 0.11 (`geolat`) |

On the logged random play the frozen model's block-level error is small,
diffuse and no larger at contact than away from it. D learns about a third of
what structure there is.

### The test: the refiner's recorded decisions (seed-1 grounded actor, draws 43/44)

AUC = P(score of a failed episode > score of a successful one); chance 0.5.
None of the refiner's commands exceeds the dataset's 100 px per-step range.

| draw, first plan | critic energy E | uncorrected terminal error | `geo` predicted fiction | `geo` corrected terminal error | `geolat` predicted fiction | `geolat` corrected error |
|---|---|---|---|---|---|---|
| 43 (12 fails / 50) | **0.81** | 0.69 | 0.60 | 0.67 | 0.55 | 0.58 |
| 44 (29 fails / 50) | **0.77** | **0.81** | 0.54 | 0.57 | 0.48 | 0.59 |

Second plans: every D score 0.34-0.66, i.e. chance both ways. D predicts about
40-48 px of summed fiction for failed AND successful plans alike (its
average-drift prior), and its correction moves every terminal estimate 20-25
px away from the goal regardless of outcome: on the refiner's plans the
offline model carries no information about failure and adds noise. The two
references are the telling part: the refiner's own imagined energy already
ranks failures above successes (AUC 0.8) -- failures are the harder tasks and
imagination is optimistic on them, as E16 measured -- and nothing learned from
random play improves on it.

**Closed.** The block-level response error the refiner exploits is not present
in the logged data at a learnable level: on random play the model is accurate
to ~10 px per block with no contact signature, so an offline discrepancy
model has nothing to learn that transfers to sustained, goal-directed pushes.
The fiction lives in the tail of the refiner's own action distribution. That
leaves exactly one version of the outcome-grounding lever: on-policy --
record the refiner's own rollouts on training tasks (episodes < 16000, via
the existing recorder), fit D to THOSE residuals, and test on held-out
decisions exactly as here. Training-time interaction only; deploy unchanged.
D checkpoints: `/newcheckpoints/armin@pantheon.inc/pusht-disc-20260913/discrepancy_{geo,geolat}.pt`.

## E34 -- hyperparameter sweep on the W1NEAR_FRZ recipe, wave 1 (2026-09-13, in flight)

User direction: find hyperparameters that improve PushT within the recipe
(frozen world model, K = 8, no restarts, no behaviour cloning). Closed axes
from the August campaign and E19: amax (2.5 sits on the plateau; tighter is
monotone worse, 3.0 falls off a cliff), mean_weight (0.1 best), actor_lr
(3e-4 >= 1e-4), actor training length (6k best, 12k/18k worse; ckpt-select
already picks early snapshots), expand_weight (moot with a frozen critic), K
(directive), critic window/expectile/n-step/near-goal/depth (E10-E20).
Untested on this recipe, each with a reason to expect a gain; three seeds
each, paired against the six-seed history-1 arrays; `cem_value` reads the
teacher.

| tag | change | why |
|---|---|---|
| `rp0` | `planner.replay_prob=0` | August: +5 at K=8 on the old recipe; replay samples are imagined starts (off-distribution) and E29 showed the deployed replan does not match them anyway |
| `md6pc01` | `planner.max_delta=6 planner.p_cross=0.1` | eval goals are exactly 25 steps = 5 blocks ahead in the same episode; the actor trains on 1-20 blocks with 30 % cross-episode goals, so 75 % of its goals are farther than any eval goal |
| `tch24k` | `value.steps=24000` | E27/E28: the TD teacher is the data-limited learner; 12k x 1024 is ~6 passes over 2M rows |
| `tchg1e01` | `value.gamma=1.0 value.expectile=0.01` | the August offline-value corner scored TD+CEM 80.0, above the config-B teacher's 78.7 as a CEM objective |

Jobs `rlp-pusht-w1near_frz-<tag>-s{0,1,2}-20260913`. Wave 2, launched behind
wave 1 on the user's go (2026-09-14): `md10` (`planner.max_delta=10`, the
horizon axis between 6 and the recipe's 20) and `ab512` (`planner.batch=512`,
actor batch doubled at fixed steps). Combinations of winners follow the read.
Ops note: twelve rapid-fire `sky jobs launch` calls all failed silently at
the API; sequential submission with per-job retries and a 15 s pause
succeeded on the first attempt every time.

### Wave-1/2 read (2026-09-14, partial: arms with >= 1 finished seed)

Medians over the seeds finished so far; `base` is the same-seed median of the
six-seed history-1 baseline arrays, so every comparison is seed-matched. Paired
per-task ledger vs those arrays in the last column.

| arm | seeds done | base (matched) | rlp | per-seed rlp | cem_value | paired fixes/breaks (sign p) |
|---|---|---|---|---|---|---|
| **`rp0`** (replay_prob 0) | 3 | 70.7 | **72.7** | 73.3 / 72.7 / 68.0 | 77.3 | 31 / 26 (0.60) |
| `md6pc01` (max_delta 6, p_cross 0.1) | 3 | 70.7 | 70.7 | 74.0 / 70.0 / 70.7 | 76.7 | 33 / 27 (0.52) |
| `md10` (max_delta 10) | 3 | 70.7 | 68.7 | 67.3 / 72.0 / 68.7 | 77.3 | 27 / 31 (0.69) |
| `ab512` (actor batch 512) | 3 | 70.7 | 68.0 | 67.3 / 71.3 / 68.0 | 77.3 | 26 / 32 (0.51) |
| `tch24k` (teacher 24k steps) | 3 | 70.7 | **63.3** | 64.0 / 62.7 / 63.3 | 75.3 | 31 / 63 (**0.00**) |
| `tchg1e01` (teacher gamma 1.0, expectile 0.01) | 3 | 70.7 | **65.3** | 65.3 / 65.3 / 64.7 | 76.7 | 26 / 49 (0.01) |

**`rp0` is the winner and it is monotone across seeds**: 73.3 / 72.7 / 68.0
against the matched baselines 72.7 / 70.7 / 67.3 -- +0.6 / +2.0 / +0.7, three
of three positive, median +2.0. This reproduces the August K=8 finding
(replay_prob 0 was +5 on the old recipe) on the current recipe, and E29 gives
the mechanism: replay samples hand the actor an IMAGINED start window with
zeroed action history, which is not what the deployed replan ever sees. Note
the per-task ledger is not significant (p = 0.60) -- the gain is a consistent
small shift, not a clean set of fixes -- so it needs the remaining three seeds
before it becomes the recipe.

**Both teacher arms are significantly negative, and that is the sharpest
result of the wave.** `tch24k` (double the teacher's optimisation, nothing
else) lands at 63.3 vs 70.7 with a paired ledger of 31 fixes to 63 breaks,
p = 0.00 -- the most significant per-task effect this campaign has measured on
any arm. `tchg1e01` (the August offline-value corner) is -5.4, p = 0.01.
In both arms the teacher itself stays healthy AS A RANKING OBJECTIVE
(cem_value 75.3-76.7, cem_tdvalue 76.0), so a teacher that is equally good for
sampling is measurably worse for descent. That is E24's inversion again, now
with a knob attached: **optimising the critic further makes it a worse
gradient field while leaving its ranking intact.** Practical consequence for
the recipe: the teacher's 12k steps is not a budget compromise to be relaxed,
it is near a ceiling, and E27/E28's "the teacher is the data-limited learner"
should be read strictly as DATA, not optimisation.

`md6pc01` completed at three seeds and is a MEDIAN NULL (70.7 vs 70.7) though
its mean is +1.3 and its best seed is the campaign's highest single-seed value
under standard protocol (74.0 = 66/88/68 at seed 0, draw 43 = 88 the highest
report-draw cell any RLP actor has produced). `ab512` completed negative
(68.0 vs 70.7). So the actor-side axes split cleanly: what the actor is
trained ON (replay distribution, goal horizon) moves the number; how much of
it (batch, teacher steps) does not.

Wave 3 (launched on this read): `rp0` at seeds 3-5 for the six-seed median,
and `rp0md6` (replay off + the eval-matched goal horizon) at seeds 0-2.
Wave 4, both on top of replay-off: `rp0ac0` drops the anti-constancy
regulariser -- `docs/campaigns/2026-08-27/RESULTS.md` falsified the TwoRoom
constancy basin on PushT (probe constancy <= 0.13 vs 0.77+ on TwoRoom, all
checkpoints), so acr 0.5 is a regulariser against a pathology this
environment does not have -- and `rp0md10` retests the milder goal horizon
unbundled from `p_cross`.

### Wave-1/2 complete: one winner, one null, four negatives

All six wave-1/2 arms are now at three seeds. Only `rp0` is positive
(72.7 vs 70.7, three of three seeds up). `md6pc01` is a median null,
`md10` is -2.0, `ab512` -2.7, `tchg1e01` -5.4 (p = 0.01) and `tch24k`
-7.4 (p = 0.00). **The goal-horizon axis is closed**: 6 blocks (bundled with
p_cross 0.1) is a null and 10 blocks is negative, so matching the actor's
training goals to the eval's 5-block horizon does not help -- the wide
1-20 band is not a handicap. `rp0md10` (wave 4) was cancelled on this read
since its motivating single-seed value did not survive.

Wave 5, launched on the tch24k finding: the teacher-optimisation ladder
DOWNWARD on top of replay-off, `value.steps` 6000 and 3000 against the
recipe's 12000. The prediction is explicit and falsifiable -- if a less
converged teacher presents a better gradient field, 6k/3k should be >= 12k;
if instead they also lose, 12k is a genuine optimum and tch24k's loss is
specifically about over-sharpening.

## E35 -- audit of the whole record: no earlier RLP PushT number survives the protocol (2026-09-14)

Prompted by "didn't we get higher for some RLP version?". Six blind searches
(campaign docs, the persistent memory notes, git history including deleted and
revised files, the SkyPilot job history, session scratchpads and sibling
repos, plus the local session transcripts found by a completeness critic)
collected every PushT number attributable to the refiner. **38 distinct
candidates at or above 69.0 were each audited by two independent agents, one
checking protocol compliance and one trying to refute it from the primary
source. None survived.** Seventy-seven agents, zero errors on the resumed run.

The disqualification classes, with the best example of each:

| class | best value | why it is not reportable |
|---|---|---|
| single DRAW of a single seed | 90 (actor_phases=5, s0, draw 43) | one 50-episode cell; that arm's own median is 70.0 |
| deploy-time restarts + argmin-V | 88 (LIP v2, R=8, draw 42) | restarts are vetoed; R=32 reaches CEM's compute |
| oracle union of two planners | 86.2 | not a planner |
| old stack, no held-out split | 84.7 (LIP + 8 restarts) | trained on the episodes it was scored on |
| **one-shot K=8, but selected on a report draw** | **80.7** (LIP v2, deep teacher tau 0.01/n1) | actor chosen by its own s42 score; its teacher won a 46-config sweep on the same draws; osmesa render |
| selection draws 50/51 | 78 (K=24 counterstrike winner) | same actor = 70.0 on report draws |
| K != 8 | 78 (K=24), 75 (K=32) | three to four times the shipped refinement budget |
| CEM-initialised hybrid | 77.3 (`rlp_ceminit_k8`) | a 9,000-rollout CEM search picks the plan; the refiner only polishes it, and the K=0 control scores the same |
| single seed, otherwise clean | 73.3 / 72.7 (W1NEAR_FRZ s0) | best of six; the n=6 median is 69.0 |

Two arms surfaced that appear in **no repo document, results table or launch
script** -- `pusht-rs8-w4-s0` (74.7) and `pusht-rs32-w4-s0` (74.0) -- recovered
only from session transcripts. Both are restart arms, so both are vetoed, but
they are a reminder that the cluster's raw output lives in transcripts and job
logs, not in the repo.

**Closest honest miss: 80.7.** The July LIP v2 one-shot (K=8, no restarts) is
the only historical number that is simultaneously above 69 and free of
restarts. It fails on selection contamination (the actor and its teacher were
both chosen on report draw 42) and predates the held-out split. It is not
comparable to anything in this campaign, but it is the reason the one-shot
ceiling was once believed to be ~80.

Ops note from the critic: `logs/`, the in-repo `scratchpad/`, the Synology
version store, and W&B (entity armin-sommer, projects scanned incl. 400 newest
RLP runs) contain NO PushT eval output. Evals never logged to W&B. The only
complete local record of cluster output is the session transcripts under
`~/.claude/projects/-Users-arminsommer-SynologyDrive-1privat-RLP-original/*.jsonl`
(463 `RESULT rlp` blocks at the time of the audit).

### E34 HEADLINE -- `replay_prob=0` is a real +3.3 at the full six seeds

| seed | baseline W1NEAR_FRZ | + replay off | delta | per-draw 42/43/44 |
|---|---|---|---|---|
| 0 | 72.7 | 73.3 | +0.6 | 74 / 84 / 62 |
| 1 | 70.7 | 72.7 | +2.0 | 74 / 82 / 62 |
| 2 | 67.3 | 68.0 | +0.7 | 74 / 76 / 54 |
| 3 | 66.0 | 70.0 | +4.0 | 78 / 78 / 54 |
| 4 | 68.0 | 72.0 | +4.0 | 74 / 82 / 60 |
| 5 | 70.0 | 75.3 | +5.3 | 80 / 84 / 62 |
| **median** | **69.0** | **72.3** | **+3.3** | |
| mean | 69.1 | 71.9 | +2.8 | |

**Six of six seeds positive; paired per-task ledger 73 fixes / 48 breaks,
sign test p = 0.03.** This is the first arm in the campaign to move the n=6
median at all, and the first with a significant paired ledger. The teacher is
untouched (cem_value 77.3, cem_tdvalue 77.0 -- unchanged from the baseline
recipe), so the gain is entirely on the actor side.

**Gap to latent-CEM: -10.3 -> -7.0.** New standing numbers for PushT:
RLP 72.3 (n=6) vs latent-CEM 79.3.

**Mechanism** (August found the same effect at K=8 on the old recipe, +5, and
it was dropped when the recipe changed): with `replay_prob=0.5` half of every
actor batch starts from an IMAGINED window -- the previous rollout's last
three imagined latents -- with the action history zeroed. E29 established that
the deployed replan never sees that input: it sees real frames, and (since
E29) real action history. Replay was training the actor on a state
distribution that does not occur at deploy, and on PushT, where imagined
latents diverge fastest at contact, that distribution is exactly where the
world model's derivative is least trustworthy.

**Scope caveat:** `replay_prob=0.5` is the shared config-B default across
environments (TwoRoom already uses 0). This result is PushT-only; whether
cube and reacher also prefer 0 is untested and is a cheap follow-up, but the
global default is NOT changed on the strength of one environment.

### Wave 4/5 on top of replay-off (partial)

| arm | seeds | base (matched) | rlp | rp0 alone, same seeds | read |
|---|---|---|---|---|---|
| `rp0ac0` (drop anti-constancy, ac_weight 0) | 2 | 71.7 | **65.7** | 73.0 | **negative, p = 0.01** |
| `rp0tch6k` (teacher 6000 steps) | 1 | 70.7 | 73.3 | 72.7 | +0.6 vs rp0, direction as predicted |

**`ac_weight=0.5` is load-bearing on PushT and my rationale for dropping it was
wrong.** The argument was that `RESULTS.md` falsified the TwoRoom constancy
basin here (probe constancy <= 0.13 vs 0.77+ on TwoRoom), so the regulariser
guards against a pathology this environment does not have. The ablation says
otherwise: removing it costs ~7 points against replay-off at the same seeds,
12 fixes to 30 breaks, p = 0.01. Correct reading of the probe: constancy stays
low *because* the regulariser is holding it there. Keep acr 0.5.

### The teacher-budget ladder, on top of replay-off

| `value.steps` | rlp (median) | seeds | teacher as CEM objective | read |
|---|---|---|---|---|
| 3,000 | **63.3** | 3 | 74.0 (degraded) | -7.4, p = 0.03; all three seeds land on 63.3 |
| 6,000 | 73.3 | 1 | 75.3 | +0.6 over replay-off at the same seed |
| **12,000 (recipe)** | **72.7** | 6 | 77.3 | the shipped value |
| 24,000 | 63.3 | 3 | 75.3 (intact) | -7.4, p = 0.00 (measured without replay-off) |

**An inverted U with a flat top at 6k-12k and both ends falling by the same
7.4 points -- but the two ends fail for opposite reasons.** At 3k the teacher
is simply undertrained and its own ranking score drops with it (74.0 vs 77.3).
At 24k the teacher's ranking score is untouched (75.3) and only the refiner
loses. So the low end is a capacity failure that any planner would feel, and
the high end is E24's inversion: further optimisation of the critic degrades
the gradient field it presents while leaving the values it ranks with intact.
The recipe's 12,000 is a genuine optimum, not a budget compromise.
