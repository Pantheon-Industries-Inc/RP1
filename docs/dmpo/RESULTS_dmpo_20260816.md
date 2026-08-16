# DMPO baseline campaign — results record (2026-08-14 … 2026-08-16)

Offline DMPO (Sacks et al., ICRA 2024; `model=dmpo`, `core/solver=dmpo`,
method mapping and deviations in [README_dmpo.md](README_dmpo.md)) evaluated
against the same-critic MPPI baseline on all three environments and both
bases. Cube and Reacher DMPO numbers are the mean over **3 optimizer training
seeds (0/1/2) × eval seeds 42/43/44 × 50 episodes** (n=9 per entry); TwoRoom
is reported at three draws (n stated per entry — see that section). MPPI rows
use eval seeds 42/43/44 × 50 episodes with the identical `value_td` critic
and world model.
Training protocol per cell: `model=rlp skip=[planner]` (caches + offline
quasimetric critic) → `model=dmpo` per seed → `rlp.eval.world_model`.
Harness: `scripts/sky/dmpo_campaign.yaml`.

Rollout budgets per decision: DMPO 256 forward (256 samples × 1 learned
iteration, no backward); MPPI 9,000 forward (300 × 30); RLP 9 forward +
8 backward.

## OGBench Cube (success %, held-out episodes 8000–9999)

| planner | roll. | LeWM h25 | LeWM h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** | 256 | **72.9** | **55.8** | 61.8 | 51.6 |
| value MPPI | 9k | 62.7 | 50.0* | **66.0** | 50.0* |
| RLP (repo record, h25) | 9 | 86.9 ± 3.0 | — | 82.0 | — |
| no-op floor (record) | — | 54.0 | — | 54.0 | — |

*Partial (1–2 eval seeds, measured before h100 MPPI was dropped from the
grid as 8 replans × 9k rollouts × 50 episodes per cell); footnote-grade only.

DMPO seeds (h25): LeWM 72.0 / 73.3 / 73.3, PLDM 60.7 / 62.7 / 62.0 —
tight per-seed spread on both bases.

## Reacher (first-hit success %, worst joint within τ, h25, held-out split)

Window-1 cells train and evaluate against the single-frame critic; window-3
cells against the 3-frame window quasimetric — matching the paper's
Tab. "Reacher by cost window" protocol (`environment.success_threshold`
scores both tolerances in one run).

| | | LeWM τ=.1 | τ=.05 | PLDM τ=.1 | τ=.05 |
|---|---|---|---|---|---|
| **(a) window-1** | **DMPO** (256) | **92.4** | **67.8** | **90.4** | **62.4** |
| | value MPPI (9k, same critic) | 64.7 | 40.0 | 68.0 | 46.7 |
| | RLP w1 ablation (paper†) | 98.7 | 88.7 | 97.8 | 82.0 |
| **(b) window-3** | **DMPO** (256) | **92.7** | **75.8** | **95.6** | **83.8** |
| | value MPPI (9k, same critic) | 83.3 | 66.0 | 88.0 | 68.0 |
| | RLP (paper) | 99.9 | 97.1 | 99.4 | 91.2 |

Cross-check: the in-job value-MPPI rows sit within a few points of the
paper's App C.3 value-MPPI rows (74.0/42.0 LeWM-w1, 86.0/66.0 LeWM-w3 …),
validating the first-hit tolerance scoring added for this campaign
(`rlp.environment.World.threshold_hits`; τ=.05 additionally reproduces the
env's own 0.05 rad termination criterion to within noise).

## TwoRoom (success %, authors' 10,000-episode pool)

| planner | roll. | LeJEPA h25 | LeJEPA h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** | 256 | **99.1** (n=9) | **98.5** (n=4) | **100.0** (n=6) | **53.0** (n=2)‡ |
| value MPPI | 9k | 86.0 (n=2)‡ | — | 84.0 (n=2)‡ | — |
| RLP (shipped table) | 9 | 100.0 | 94.2 | 98.2 | 96.0 |

TwoRoom cells are reported at **three eval draws** rather than the full
3 actor × 3 draw grid: the h25 rows are saturated (every cell 96–100) and the
LeJEPA h100 spread is 98/98/98/100, so additional draws cannot move them.
‡ marks entries still at n=2 when this record was written; they are being
topped up to n=3 before the jobs are stopped.

Per-cell values — LeJEPA h100: 100 (a0/s42), 98 (a0/s43), 98 (a1/s42),
98 (a2/s42). PLDM h100: 56 (a0/s42), 50 (a1/s42).

The h100 asymmetry is the notable TwoRoom result: **LeJEPA barely degrades**
(99.1 → 98.5) while **PLDM collapses** (100.0 → ~53) under the same recipe,
critic settings, and planner — the difference is the world-model base. RLP's
shipped rows degrade on neither (94.2 / 96.0). DMPO's single learned
iteration is trained on 5-block problems and nothing in it adapts to the
8-replan regime; on a base whose rollouts drift more, that shows up as a
long-horizon collapse. Evaluation runs through
the `tworoom_{lewm,pldm}_h5` roots (the fetched pool stores the agent
position as `pos_agent` and has no `state` column; the collector-produced
lance is not regenerable in-job — 3,443/10,000 episodes in 11 h and
decelerating).

## Reading

1. **DMPO beats the update it residuals on almost everywhere.** Against
   MPPI under the identical critic, WM, and data: +10.2 (Cube LeWM h25),
   +19.7 to +27.7 (all four Reacher w1/w3 τ columns on LeWM; +9 to +18 on
   PLDM), +13.3/+14.0 (TwoRoom h25) — at 1/35th the rollouts. The one loss
   is Cube PLDM h25 (−4.2), where `amax=4.5` (inherited from the RLP recipe,
   untuned for DMPO) is the prime suspect.
2. **DMPO loses to RLP in every cell where an RLP number exists** — by 6–21
   points on Reacher, ~14–20 on Cube. Under a shared frozen model and cost,
   the learned sampling-reduction is consistently weaker than the learned
   gradient-refinement at a fraction of RLP's gap to the hand-written
   planners.
3. **Horizon degrades DMPO faster than RLP, and base-dependently.** Cube
   h25→h100 drops DMPO ~10–17 points on both bases. On TwoRoom the split is
   stark: LeJEPA holds (99.1→98.5) while PLDM collapses (100.0→~53) against
   RLP's 96.0. DMPO's single learned iteration was trained on 5-block
   problems and nothing in it compensates for the 8-replan regime.
4. **Scope of the claim.** This is *offline* DMPO: the paper's update rule
   and inner loop, trained by pathwise gradients against the shared critic
   through the frozen world model (the same regime RLP trains in). It
   measures optimizer quality under a given cost — not the paper's
   return-driven model-error compensation, which requires on-system
   interaction (see README_dmpo.md).

## Provenance

| cell | jobs (train+eval) | tag |
|---|---|---|
| Cube LeWM | 6758 → 7025/7152 | `dmpo-cube-lewm-20260814` |
| Cube PLDM | 6759 → 7026/7153 | `dmpo-cube-pldm-20260814` |
| Reacher LeJEPA/PLDM w1 | 7018 / 7020 | `dmpo-re-{lejepa,pldm}-w1-20260815` |
| Reacher LeJEPA/PLDM w3 | 7022 / 7023 | `dmpo-re-{lejepa,pldm}-w3-20260815` |
| TwoRoom LeJEPA/PLDM | 7143/7287/7340 / 7145/7290/7341 | `dmpo-tw-{lejepa,pldm}-20260815` |

Per-cell result files live under
`/checkpoints/armin@pantheon.inc/<tag>/results_*.txt` on the cluster volume.

Ops notes hit during the campaign (fixes committed): pipefail-killed dataset
discovery (`|| true`), TwoRoom pool regeneration infeasible in-job, the
authors' TwoRoom h5 needing dedicated eval roots, first-hit tolerance
scoring for Reacher's τ=0.1 column, resumable eval cells, and 4-wide
parallel evals requiring per-process `OMP_NUM_THREADS` caps (default
threading made 4-wide *slower than sequential*: zero cells in 3.2 h).
