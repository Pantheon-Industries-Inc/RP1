# Why Dyna is not at 100 on Cube under the shared config (2026-09-03)

Scope: the Dyna row of `HANDOFF_CONFIGS_20260903.md` (cube only; TwoRoom
Dyna WMs were lost with the July pods, PushT has no Dyna). Under config B
the Dyna WMs score LeWM 94.0 / 82.0 and PLDM 92.0 / 85.3 (h25 / h100,
medians of 3 seeds). Everything below is held-out (eval draws 8000:10000).

## 1. What was checked and cleared

- **No cache/WM mismatch.** The Dyna jobs (`rlp-cu-v2dyna[-p]-20260902`)
  built their own latent caches (`cu-v2dyna[-p]-n1s2-v1`), so critic and
  actor trained on the Dyna WM's latents and planned through the same WM.
  (The harness keys the cache on `CACHE_VERSION` only, not on the WM, so
  this has to be checked per launch.)
- **Early stopping behaved.** All six Dyna actors deployed a snapshot
  (step 2000/4000) chosen on val draws 48-51.
- **Per-seed numbers** (each cell = draw 42/43/44):

| arm | seed | h25 | h100 |
|---|---|---|---|
| Dyna LeWM | a | 96 / 92 / 94 = 94.0 | 82 / 78 / 86 = 82.0 |
| Dyna LeWM | b | 98 / 94 / 98 = 96.7 | 88 / 76 / 88 = 84.0 |
| Dyna LeWM | c | 98 / 88 / 96 = 94.0 | 82 / 80 / 84 = 82.0 |
| base LeWM | a | 96 / 92 / 82 = 90.0 | 86 / 74 / 90 = 83.3 |
| base LeWM | b | 96 / 92 / 90 = 92.7 | 88 / 82 / 88 = 86.0 |
| base LeWM | c | 92 / 90 / 84 = 88.7 | 88 / 86 / 86 = 86.7 |
| Dyna PLDM | a | 92 / 92 / 88 = 90.7 | 90 / 86 / 86 = 87.3 |
| Dyna PLDM | b | 92 / 90 / 94 = 92.0 | 84 / 80 / 88 = 84.0 |
| Dyna PLDM | c | 92 / 94 / 90 = 92.0 | 90 / 80 / 86 = 85.3 |

  Dyna's h25 gain is concentrated on draw 44 (base 82/90/84 -> 94/98/96);
  draw 43 barely moves (92/92/90 -> 92/94/88). Draw 43 is the hard draw at
  both horizons for every actor family.

## 2. The frozen-WM hard core, by task index

Per-task success arrays from the failure-atlas re-evals
(`rlp-atlas-cu-{d2,dc}[-p]-20260823/rawlogs`, 11-12 actors per cell: six
seeds x two capacities, both bases). Task draws depend only on the draw
seed and `ep_range`, so indices are comparable across every cube actor.

| horizon | draw 42 | draw 43 | draw 44 | total | ceiling |
|---|---|---|---|---|---|
| h25 | {8, 30} | {2, 8, 9} (+1 LeWM-only) | {4, 8, 9, 33} (+29 PLDM-only) | 9 / 150 | **94.0** |
| h100 | {12, 25, 34, 35, 41} | {0, 1, 2, 10, 28, 38, 47} (+26 LeWM / +29 PLDM) | {4, 8, 12, 23, 43} | 17 / 150 | **88.7** |

These are the tasks no frozen-WM actor solves. The h25 and h100 cores are
disjoint (different segments of the same episodes at different offsets).

## 3. Dyna already breaks the "task-intrinsic" core

- The July Dyna clips (`Dyna/videos_pre_post_20260730`, same draws 42/44,
  July recipe, h25-collected WM) cured h25 core tasks **8 (draw 42)** and
  **9 (draw 44)**; they cured none of the h100 core (34/35/41 and 8/12/43
  still fail both arms).
- Dyna-under-B LeWM h25 mean is 94.9, above the frozen-WM ceiling of 94.0.

So the atlas verdict "Cube = task-intrinsic hard core, not a method
deficiency" holds only for a frozen world model. The core is
**WM-intrinsic**: tasks the base WM mis-models around contact, which one
Dyna iteration on the planner's own h25 rollouts partly repairs.

## 4. What the shipped Dyna WMs actually are (and why they stop at ~95 / ~84)

`assets/core/world_model/{lewm,pldm}_cube_dyna` = wandb artifacts
`dyna_wm_{lewm,pldm}_olr_a1.6_m0.3_l1e-4`, built by
`tworoom_matrix_20260727/sky/dyna_lewm.yaml` (branch eval-sweep):

1. **One iteration.** Collect -> fine-tune -> stop. No second round on the
   post-Dyna planner's failures.
2. **Collected by a different planner than the one deployed.** PRE actors
   were the 2026-08-05 open-loop-sweep actors (amax 1.6, md 12, no acr,
   depth 2). Config B plans with amax 2.5, md 20, acr 0.5. The WM was
   repaired where the *old* planner failed; config B fails elsewhere.
3. **h25 only.** 12 calls x 3 actors x 50 episodes = 1800 episodes at
   goal_offset 25 / budget 50. No long-horizon rollouts. The July 2x2 that
   would have tested "collect at h100" (`dyna_h100.sh`) reached the
   fine-tune stage and was never evaluated (no `h1post_*` rows in the
   rescued CSV), so the handoff's "h25-only collection forgets long-horizon
   dynamics" is a hypothesis, not a measurement. The July h100 numbers are
   contradictory across recipes (LeWM +3.6, PLDM -2.9 at n=3), i.e. inside
   noise.
4. **Uniform 50/50 mix, epoch 1 of 2, lr 1e-5, encoder unfrozen, no anchor,
   no rollout loss.** Fine-tune loss is 1-step latent MSE + SIGReg on the
   mixture (`lewm_expert.py`); nothing targets the failure states
   specifically. July nulls: failure fraction 0.08 -> 0.13 (-0.4), pool
   distinctness (-0.2), so more of the *same* data does nothing.
5. **PLDM's Dyna WM is a LeWM-class model** (fine-tuned via
   `lewm_expert.py` with the PLDM encoder renamed; pre-registered caveat).
6. **Eval-time headroom is the planner's, not the WM's, at h25 draw 43.**
   Dyna moved draw 44 by +10 and draw 43 by ~0; the base actors already
   fail 3-6 tasks on draw 43 that Dyna does not touch.

## 5. What "100" needs (arithmetic)

At h25, 100 = solving the residual ~6-8 tasks per 150 that Dyna-B still
misses. Under the frozen WM these were {30}, {2, 8, 9}, {4, 8, 33} plus
seed-dependent stragglers; the per-task re-eval below says which of them
Dyna still fails. At h100, 100 means solving 17 core tasks plus a
~10-task contested band; the h25-collected WM has never been shown to
move any h100 core task.

## 6. Per-task re-evaluation (2026-09-03)

Tags `rlp-cu-pertask-{dyna,dyna-p,base,base-p}-20260903`: the persisted
final actors from the four 2026-09-02 jobs re-evaluated with
`EVAL_RAWDIR=volume`, so per-task success arrays exist.

## 7. Results of the per-task cross

The re-evals reproduced every original `[summary]` cell to the digit
(deterministic eval), so the arrays below are the shipped numbers.
"Fail-cells" = failing (task, seed) pairs out of 3 seeds x 150 tasks = 450.

| base | horizon | base fail-cells | Dyna fail-cells | frozen-core tasks fully cured by Dyna | frozen-core tasks still failing all 3 Dyna seeds |
|---|---|---|---|---|---|
| LeWM | h25 | 43 | **23** | 6 of 9: d42 {8, 30}, d43 {1}, d44 {4, 9, 33} | 3: d43 {8, 9}, d44 {8} |
| LeWM | h100 | 67 | **78** | 0 of 17 | 17 (all) + 2 new: d43 {44}, d44 {31} |
| PLDM | h25 | 57 | **38** | 2 of 9: d43 {1}, d44 {9} | 7: d42 {8}, d43 {2, 8, 9}, d44 {8, 29, 33} |
| PLDM | h100 | 66 | 65 | 0 of 16 | 16 (all) |

### Where LeWM-Dyna's remaining 23 h25 fail-cells sit

| draw | task: seeds failing (of 3) |
|---|---|
| 42 | 32:1, 34:1, 35:1, 46:1 |
| 43 | **8:3, 9:3**, 2:2, 11:2, 44:2, 5:1 |
| 44 | **8:3**, 29:2, 45:1 |

- **9 of 23 cells are three tasks that fail every seed** (draw 43 tasks 8
  and 9, draw 44 task 8). These are frozen-core tasks the one-iteration,
  old-planner, h25-only WM did not repair. Solving them is worth +2.0.
- **14 of 23 cells are seed-dependent** (11 distinct tasks, each failing
  1-2 of 3 seeds). That is a planner/critic lottery on a WM that, for
  some seed, does support the task; it is worth +3.1 and is not a
  world-model problem. Two of these tasks (draw 43: 11, 44) are *new*
  Dyna-induced failures the base actors rarely had.
- Ceiling arithmetic for Dyna-B LeWM at h25: 100 - 3 always-fail tasks/1.5
  = **98.0** if the lottery were eliminated; the measured 94.9 is the
  lottery's cost.

### h100: the h25-collected WM moves nothing in the core

Zero of the 17 (LeWM) / 16 (PLDM) frozen-core h100 tasks were cured by
either Dyna WM; LeWM-Dyna adds 11 fail-cells over base, including two new
always-fail tasks. The h25-collected fine-tune is horizon-specific in the
direction that matters: it repairs short-horizon contact segments and
leaves (or slightly damages) the long ones. This is now measured at the
task level; the July "collect at h100" 2x2 that would show whether
h100-collected data cures the h100 core was never evaluated.

### Base-dependence

On PLDM the same Dyna WM (a LeWM-class fine-tune of the renamed PLDM
encoder) cures only 2 of 9 h25 core tasks and leaves 7, versus LeWM's 6/3.
Task 8 in draws 42/43/44 (three different episodes) survives on PLDM in all
three draws.

## 8. Answer: why Dyna is not at 100, and what 100 would take

Why not (in order of size, cube h25, LeWM):

1. **Seed lottery on the planner side (~3 pts, 14/23 residual cells).**
   Not a WM problem: for each of these tasks some seed succeeds on the
   same Dyna WM. Candidate levers are planner-side (selection across seeds
   on val draws, deploy-time restarts as in PushT, heavier ES), not more
   Dyna.
2. **Three h25 tasks the Dyna WM still mis-models (~2 pts).** The shipped
   fine-tune saw rollouts of the *old* open-loop planner (amax 1.6, md 12,
   depth 2) at h25 only, one iteration, uniform 50/50 mix, epoch 1, no
   anchor. Those three tasks were never in its failure data with the
   current planner's behaviour.
3. **At h100 Dyna is currently a net negative (-11 fail-cells) and cures
   none of the 17-task core.** Collection horizon is the obvious cause and
   is untested; the h100 core is the pick-and-place / multi-grasp funnel
   family (RESULTS.md, h200 probe).

What a "Dyna to 100" attempt has to include (none of these were in the
shipped WMs): (a) collect with the *deployed* config-B planner, not the
2026-08-05 actors; (b) collect at both offsets (25 and 100) so the
fine-tune sees long-horizon contact sequences; (c) iterate: re-collect
with the post-Dyna planner and fine-tune again, since one iteration left a
specific 3-task residue; (d) anchor the fine-tune (frozen encoder or a
distillation term to the base WM) so h100 does not regress while h25
improves; (e) keep the collect 0:8000 / eval 8000:10000 split. The
planner-side lottery (item 1) is independent of all of this and caps any
single-seed report near 98 unless addressed separately.

Data: `rlp-cu-pertask-{dyna,dyna-p,base,base-p}-20260903/rawlogs`,
parsed by the scratchpad `parse_pertask.yaml`; frozen core from
`rlp-atlas-cu-*-20260823/rawlogs`.

## 9. Seed-lottery levers, measured on the existing actors (2026-09-04)

**Cross-seed selection on val draws 48-51** (`scripts/sky/dyna2/select_seed.yaml`,
job dyna2-select-base). Val-selected seed vs median-of-seeds:

| arm | val scores (s0/s1/s2) | h25 median -> selected | h100 median -> selected |
|---|---|---|---|
| base LeWM | 82.3 / 82.8 / 83.8 | 90.0 -> 90.0 | 86.0 -> 83.3 |
| base LeWM s3-5 | 81.0 / 84.3 / 82.5 | 90.0 -> 90.7 | 86.7 -> 87.3 |
| Dyna LeWM | 88.0 / 84.8 / 87.5 | 94.0 -> **96.7** | 82.0 -> 84.0 |
| base PLDM | 83.0 / 83.3 / 81.0 | 87.3 -> 87.3 | 85.3 -> 84.7 |
| Dyna PLDM | 86.5 / 86.3 / 85.8 | 92.0 -> 90.7 | 85.3 -> 87.3 |

Val scores sit within ~3 points of each other (200 val tasks, one h25/h100
blend), so selection is at the noise floor: it gains +2.7 on Dyna-LeWM h25 and
loses 1.3 on Dyna-PLDM h25. Cross-seed selection with four val draws is
**not** a reliable lever; it would need a much larger val set (and per-horizon
selection) before it can be reported. Deploy-time restarts were ruled out by directive (2026-09-04): the planner
runs its K=8 refinement once, always; the R=8 evals were cancelled unrun.


## 10. Dyna v2, iteration 1 (2026-09-05): anchored fine-tune, config-B collection at both offsets

Tags `rlp-cu-dyna2t-{lewm,pldm}-it1-20260903` (WM from `rlp-cu-dyna2-*-it1`:
3600 on-policy episodes per base from the config-B actors, 1800 at h25 + 1800
at h100, 50/50 mix, latent anchor weight 1.0, native PLDM objective for PLDM,
1 epoch lr 1e-5). Same eval protocol and per-task cross as section 7.

| base | arm | h25 median (seeds) | h25 fail-cells /450 | h25 core cured / 9 | h100 median (seeds) | h100 fail-cells /450 | h100 core cured / still-always |
|---|---|---|---|---|---|---|---|
| LEWM | base | 90.0 (88.7/90.0/92.7) | 43 | 0 / 9 | 86.0 (82.7/86.0/86.7) | 67 | 0 / 17 |
| LEWM | shipped Dyna | 94.0 (94.0/94.0/96.7) | 23 | 6 / 9 | 82.0 (82.0/82.0/84.0) | 78 | 0 / 17 |
| LEWM | Dyna v2 it1 | 93.3 (92.0/93.3/97.3) | 26 | 1 / 9 | 82.0 (81.3/82.0/86.0) | 76 | 0 / 14 |
| PLDM | base | 87.3 (86.7/87.3/88.0) | 57 | 0 / 9 | 85.3 (84.7/85.3/86.0) | 66 | 0 / 19 |
| PLDM | shipped Dyna | 92.0 (90.7/92.0/92.0) | 38 | 2 / 9 | 85.3 (84.0/85.3/87.3) | 65 | 0 / 16 |
| PLDM | Dyna v2 it1 | 92.0 (91.3/92.0/93.3) | 35 | 2 / 9 | 85.3 (84.7/85.3/88.0) | 63 | 0 / 15 |

Reading:
- **h25: v2 it1 lands where the shipped Dyna WM landed** (LeWM 93.3 vs 94.0,
  PLDM 92.0 vs 92.0), but by a different route: it cures only 1 (LeWM) / 2
  (PLDM) of the 9 frozen-core tasks where the shipped WM cured 6 / 2, and
  instead removes seed-dependent failures. The LeWM residual is again three
  always-fail tasks (draw 43 tasks 2 and 8, draw 44 task 8) plus stragglers.
- **h100: the anchor did not remove LeWM's regression** (82.0 vs base 86.0;
  PLDM flat at 85.3). No h100 core task is fully cured on either base, but
  the always-fail set shrinks from 17 to 14 (LeWM) and 19 to 15 (PLDM) — the
  h100 rollouts made 3-4 core tasks solvable for some seeds, which the
  h25-only shipped WM never did (it made the set larger: 19).
- Val-selected seeds (48-51): LeWM s1 97.3 / 86.0 (val 86.5 vs 83.3 for the
  other two), PLDM s2 92.0 / 85.3. Same noise-floor caveat as section 9;
  reported numbers stay median-of-seeds.
- Interpretation: anchor weight 1.0 keeps the WM within 0.3% (LeWM) / 3.3%
  (PLDM) relative L2 of the base, which protects the base's competence but
  also blocks the larger correction the unanchored shipped WM made on the h25
  core tasks. Iteration 2 (running, anchored to the it1 WM, collected with the
  it1 actors) tests whether the correction compounds; an anchor ablation
  (weight 0.1; frozen encoder) on the same it1 collection is queued to
  separate "anchor too strong" from "data insufficient".
