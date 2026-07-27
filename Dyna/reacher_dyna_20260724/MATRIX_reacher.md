# Reacher planner × cost matrix (canonical LeWM dataset)

**Status: IN PROGRESS.** Numbers below are live as of 2026-07-27 00:15 UTC and
will be refreshed when the sweep completes. Cells marked with `n<6` are partial.

## What this measures and why

Separates three things that were previously entangled in the reacher results:

1. **Does the learned TD value beat the WM's native latent cost?** Same solver,
   same WM, only the terminal cost changes (latent MSE vs learned quasimetric).
2. **How much of LIP's advantage is the learned *planner*, versus just the
   learned *value*?** LIP optimises through the value gradient; TD+CEM uses the
   same value with sampling search. The gap between them is what the learned
   planner contributes on top of the learned value.
3. **Does Dyna's gain survive against every baseline**, not only CEM?

## Protocol (identical for every cell)

- Data: **canonical** `quentinll/lewm-reacher` → `reacher.h5` (10,000 episodes
  × 201 steps). This is the authors' own file, named by
  `scripts/train/config/data/dmc.yaml`. Earlier reacher numbers used our
  re-collection, which is content-identical (our row *t* = canonical row *t+1*,
  actions matching to 3e-08) but gave a different eval task draw, worth ~+5.
- Env/task: `swm/ReacherDMControl-v0`, `qpos_match` (success = both joint angles
  within 0.05 rad). Repo-default plan config: horizon 5, receding 5,
  action_block 5.
- Card = {h25 (goal offset 25, budget 50), h50 (offset 50, budget 100)} × eval
  seeds {42, 43, 44}, n=50 tasks/cell → 6 cells, reported as their mean.
- **Seed discipline**: every *learned* component gets 3 independent training
  seeds, each evaluated on all 3 task draws (18 cells → one number). TD values
  and LIP actors both. Costs with no learned component (latent) have 6 cells.
  Single cells carry ±6–7 points of task-draw noise plus ~±2 of GPU
  nondeterminism, so only multi-seed means are compared.
- Solvers: CEM (300×30), MPPI (300×30, temperature 0.5, **untuned**), Adam
  (`GradientSolver`, AdamW lr 0.1, 30 steps).

## Pre-Dyna (author bases)

| cost × solver | LeWM (lejepa) | PLDM |
|---|---|---|
| Latent + CEM | 83.7 | 85.0 |
| Latent + MPPI | 54.7 *(n=3)* | 44.3 |
| Latent + Adam | pending | pending |
| **TD + CEM** (3 seeds) | **85.1** *(n=17)* | **90.1** |
| TD + MPPI (3 seeds) | 51.2 | 57.2 |
| TD + Adam (3 seeds) | pending | pending |
| **LIP v4 tandem** (3 seeds) | **76.9** | **78.8** |

Per-seed spread (shows why single seeds are not comparable):
- LeWM TD+CEM 86.0 / 87.7 / 81.7 · LIP 72.7 / 77.7 / 80.3
- PLDM TD+CEM 91.7 / 90.0 / 88.7 · LIP 70.3 / 83.0 / 83.0

### Reading so far

- **The learned value helps, with CEM.** TD+CEM > Latent+CEM on both bases
  (+1.4 LeWM, +5.1 PLDM). PLDM's value is the stronger of the two, consistent
  with it having the smallest predictor-vs-encoder offset of the three bases.
- **The learned value does NOT rescue MPPI.** TD+MPPI ≈ Latent+MPPI (51 vs 55;
  57 vs 44) and both sit ~30 points below CEM. MPPI is untuned at temperature
  0.5 — treat these rows as a floor, not as a considered baseline.
- **Pre-Dyna LIP (77–79) is below both CEM variants.** This is the gap Dyna
  closes: post-Dyna LIP reached 93.1 (LeWM) / 89.6 (PLDM) on the earlier eval
  draw, and nearly all of that came from retraining the actor on the fine-tuned
  WM rather than from the WM helping the old actor (85.0 → 93.1).
- Pre-Dyna LIP is *higher* here than the 63.8 we reported for LeWM on our own
  eval draw — the canonical draw lifted LIP (~+13) more than it lifted CEM (+5).

## Post-Dyna

Only the **LIP** row is valid on the existing fine-tuned WMs: their round-1
fine-tune consumed **LIP rollouts**, so that WM is adapted to LIP's state
distribution and scoring CEM/MPPI/Adam on it would be confounded. The sampling
baselines therefore each get their **own** Dyna loop
(`reacher_dyna_perplanner.sh`): collect with that planner → build its own 80/20
mix → fine-tune → retrain TD on the result if it uses the TD cost → card. Not
yet run; ~26 h on 2 GPUs for all six, or ~9 h for the two CEM arms, which carry
most of the scientific weight.

**Caveat to state in any writeup:** the fine-tuned WMs and their fresh actors
were trained during the own-collection phase, so the post-Dyna rows use assets
built on our re-collected data while being *evaluated* on canonical. The two
datasets are content-identical, so this is defensible, but it is not the same as
redoing the Dyna loop end-to-end on canonical data.

## Bugs found while building this

- `scripts/plan/config/solver/mppi.yaml` did not exist — `MPPISolver` shipped
  with no Hydra config, so MPPI had never been runnable. Created.
- `solver/gd.py`: `init_action` only moved actions to the device inside the
  zero-padding branch, so a warm-started full-horizon plan stayed on CPU and
  every second replan died with a cuda/cpu mismatch. **All Adam cells failed
  until this was fixed** (52 FAIL rows, cleared and re-queued).
- Both fixes had been written locally but never reached the pod: they were in a
  single `scp a && scp b && scp c` chain that aborted on an earlier missing
  file, silently dropping everything after it. Both are now committed to the
  `reacher` branch of `stable-worldmodel` so they cannot be lost again.

## Reproduce

```bash
# pre-Dyna full matrix + the one valid post-Dyna LIP row, per base, one GPU each
scripts/plan/reacher_matrix.sh lejepa 0 "pre postlip"
scripts/plan/reacher_matrix.sh pldm   1 "pre postlip"
# 3 TD training seeds (prerequisite for the TD rows)
scripts/plan/train_td_seeds.sh
# per-planner Dyna for a sampling baseline
scripts/plan/reacher_dyna_perplanner.sh lejepa latent_cem 0
```

Rows cache per cell in `/workspace/results/summary_matrix_<base>.csv`, so the
drivers are restartable and skip completed work.
