# Adam (GradientSolver) hyperparameter sweep — 2026-08-15/16

Sweep of the paper's Adam gradient-planner baseline (`core/solver=adam`,
`GradientSolver` with AdamW) over **lr × n_steps** at the paper's fixed
100-sample budget, latent objective, on all three environments × both bases.

- Grid: lr ∈ {0.03, 0.1, 0.3, 1.0} × n_steps ∈ {15, 30, 60}; paper default is
  (0.1, 30).
- Protocol: **selection seeds 50/51 only** (never report these), 50 episodes,
  h25 (`goal_offset_steps=25 budget=50`), held-out episode range 8000:10000 on
  every environment. Reacher reports first-hit at tau 0.1 and 0.05.
- Harness: `scripts/sky/adam_sweep.yaml` (commit 1a48a44), jobs 7112/7114/
  7115/7116/7117/7118, tag `adam-sweep-20260815`. Per-cell result files at
  `/checkpoints/armin@pantheon.inc/adam-sweep-20260815/<env>_<base>/`.

All numbers below are means over the two selection draws.

## Cube (OGBench) — LeWM | PLDM

| lr \ steps | 15 | 30 | 60 |
|---|---|---|---|
| 0.03 | 63.0 \| 52.0 | 62.0 \| 49.0 | 63.0 \| 49.0 |
| 0.1  | 60.0 \| 52.0 | 60.0 \| 50.0 | **68.0** \| 52.0 |
| 0.3  | 51.0 \| 44.0 | 59.0 \| 47.0 | 64.0 \| 48.0 |
| 1.0  | 46.0 \| 40.0 | 47.0 \| 41.0 | 45.0 \| 46.0 |

LeWM winner lr 0.1 / 60 steps (68.0 vs default 60.0). PLDM is flat for
lr ≤ 0.1 (52.0 three-way tie); no meaningful win over the default.

## TwoRoom — LeJEPA | PLDM

| lr \ steps | 15 | 30 | 60 |
|---|---|---|---|
| 0.03 | **94.0** \| 94.0 | 90.0 \| 90.0 | 88.0 \| 89.0 |
| 0.1  | 92.0 \| **97.0** | 91.0 \| 95.0 | 86.0 \| 86.0 |
| 0.3  | 58.0 \| 77.0 | 60.0 \| 76.0 | 54.0 \| 70.0 |
| 1.0  | 32.0 \| 39.0 | 33.0 \| 37.0 | 25.0 \| 36.0 |

TwoRoom prefers *fewer* steps and low lr; more optimization at high lr
degrades monotonically (over-optimization against the latent objective).
Winners are 1–3 pts over the default — inside selection noise.

## Reacher (tau 0.1 / tau 0.05) — LeJEPA

| lr \ steps | 15 | 30 | 60 |
|---|---|---|---|
| 0.03 | 94 / 70 | 93 / 80 | 90 / 69 |
| 0.1  | 91 / 79 | **95 / 80** | 94 / 68 |
| 0.3  | 80 / 56 | 90 / 72 | 96 / 79 |
| 1.0  | 12 / 6 | 17 / 7 | 16 / 8 |

## Reacher (tau 0.1 / tau 0.05) — PLDM

| lr \ steps | 15 | 30 | 60 |
|---|---|---|---|
| 0.03 | 91 / 74 | 96 / 74 | **97 / 86** |
| 0.1  | 90 / 70 | 91 / 70 | 99 / 83 |
| 0.3  | 88 / 64 | 93 / 69 | 92 / 77 |
| 1.0  | 13 / 8 | 9 / 4 | 16 / 10 |

LeJEPA: the paper default is already optimal (95/80). PLDM: the 60-step
column dominates — lr 0.03/n60 gains ~13 pts at tau 0.05 over the default
(86 vs 70).

## Takeaways

1. **The paper's default (lr 0.1, 30 steps) is a fair baseline** — at or near
   the grid optimum on 4 of 6 cells. The only material win is Reacher-PLDM at
   60 steps (+13 at tau 0.05) and a possible +8 on Cube-LeWM at 60 steps.
2. **lr 1.0 is catastrophic everywhere** (Reacher collapses to ~10, TwoRoom
   to ~30); lr 0.3 already costs 20–35 pts on TwoRoom. The usable range is
   lr ≤ 0.1.
3. **Step count interacts with the base**: TwoRoom degrades with more steps,
   Reacher-PLDM improves — consistent with over-optimization against latent
   distance being base/env-dependent.
4. These are selection-seed numbers (50/51). Any quoted number needs a report
   pass on 42/43/44 for the chosen cell.
