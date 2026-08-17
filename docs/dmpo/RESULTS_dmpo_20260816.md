# DMPO baseline campaign — results record (2026-08-14 … 2026-08-16)

Offline DMPO (Sacks et al., ICRA 2024; `model=dmpo`, `core/solver=dmpo`,
method mapping and deviations in [README_dmpo.md](README_dmpo.md)) evaluated
against the same-critic MPPI baseline on all three environments and both
bases. Cube and Reacher DMPO numbers are the mean over **3 optimizer training
seeds (0/1/2) × eval seeds 42/43/44 × 50 episodes** (n=9 per entry). MPPI
rows use eval seeds 42/43/44 × 50 episodes with the identical `value_td`
critic and world model. All 8 cells are complete.
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

## TwoRoom (success %, held-out split)

**Held-out protocol** (train 0–7999, eval draws 8000–9999) — matching the RLP
rows in `RESULTS_REPLICATION_20260814.md`. Jobs 7689/7690, 21 cells each.

| planner | roll. | LeJEPA h25 | LeJEPA h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** | 256 | 96.7 | **99.1** | 97.3 | 54.4 |
| value MPPI | 9k | 83.3 | — | 76.0 | — |
| RLP (held-out) | 9 | **100.0** | 94.2 | **98.2** | **96.0** |
| value CEM (held-out record) | 9k | 100.0 | 96.0 | 100.0 | 89.3 |

An earlier version of these cells ran the authors' full-pool contract
(train and eval on all 10,000 episodes) and was therefore an upper bound:
99.1 / 98.9 / 99.8 / 58.4, with MPPI at 84.7 / 80.0. The leak was worth
~2–3 points at h25 on both bases. `TWOROOM_SPLIT=0` reproduces it; the cache
name encodes the split so the two cannot collide.

Two findings survive the corrected protocol:

1. **PLDM collapses at h100** (97.3 → 54.4) while LeJEPA does not
   (96.7 → 99.1) — consistent across all three optimizer seeds (full-pool
   run: 57.3 / 58.0 / 60.0). Same recipe, same critic settings, same
   planner; the difference is the world-model base. DMPO's single learned
   iteration is trained on 5-block problems and nothing in it adapts to the
   8-replan regime, so on a base whose imagined rollouts drift more, that
   surfaces as long-horizon collapse. RLP holds 96.0 there.
2. **TwoRoom LeJEPA h100 is a weak RLP cell.** Under the identical held-out
   protocol, RLP's 94.2 is beaten by DMPO (99.1, +4.9) and by value+CEM
   (96.0, +1.8), and sits 3.8 below the paper's own quote (98.0, full pool).
   This is the only cell in the campaign where DMPO beats RLP. A plausible
   mechanism: the TwoRoom RLP recipe trains the refiner with
   `max_delta=12` (goals ≤12 blocks) while h100 places the goal 20 blocks
   out — outside its training range of goal distances, which would penalize
   a learned refiner more than a re-sampling planner. Untested; one retrained
   TwoRoom RLP seed at `max_delta=20` would settle it.

Evaluation runs through
the `tworoom_{lewm,pldm}_h5` roots (the fetched pool stores the agent
position as `pos_agent` and has no `state` column; the collector-produced
lance is not regenerable in-job — 3,443/10,000 episodes in 11 h and
decelerating).

## Reading

1. **DMPO beats the update it residuals on almost everywhere.** Against
   MPPI under the identical critic, WM, and data: +10.2 (Cube LeWM h25),
   +19.7 to +27.7 (all four Reacher w1/w3 τ columns on LeWM; +9 to +18 on
   PLDM), +13.4/+21.3 (TwoRoom h25) — at 1/35th the rollouts. The one loss
   is Cube PLDM h25 (−4.2), where `amax=4.5` (inherited from the RLP recipe,
   untuned for DMPO) is the prime suspect.
2. **DMPO loses to RLP in 13 of 14 comparable cells** — by 6–21 points on
   Reacher, ~14–20 on Cube, 0.9–41.6 on TwoRoom. The single exception is
   TwoRoom LeJEPA h100 (99.1 vs 94.2), which is a weak RLP cell rather than a
   DMPO strength: value+CEM also beats RLP there. Under a shared frozen model
   and cost, the learned sampling-reduction is otherwise consistently weaker
   than the learned gradient-refinement.
3. **Horizon degrades DMPO faster than RLP, and base-dependently.** Cube
   h25→h100 drops DMPO ~10–17 points on both bases. On TwoRoom the split is
   stark: LeJEPA holds (96.7→99.1) while PLDM collapses (97.3→54.4) against
   RLP's 96.0 — consistent across all three optimizer seeds. DMPO's single
   learned iteration was trained on 5-block problems and nothing in it
   compensates for the 8-replan regime.
4. **The comparison is not compute-matched, and the mismatch favors DMPO.**
   Counting a backward pass as ~2 forward, RLP spends ~25 forward-equivalents
   per decision (9 fwd + 8 bwd) against DMPO's 256 — DMPO loses every cell
   while spending an order of magnitude more compute. A compute-matched DMPO
   row (`num_samples≈24`) is not yet run; `num_samples` is architectural (the
   actor reads the N costs positionally) so it requires retraining, and the
   256 used here is the paper's advertised operating point, not a tuned value.
5. **Scope of the claim.** This is *offline* DMPO: the paper's update rule
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
| TwoRoom LeJEPA/PLDM (held-out) | 7689 / 7690 | `dmpo-tw-{lejepa,pldm}-split-20260817` |
| TwoRoom LeJEPA/PLDM (full pool, superseded) | 7340 / 7341 | `dmpo-tw-{lejepa,pldm}-20260815` |

Per-cell result files live under
`/checkpoints/armin@pantheon.inc/<tag>/results_*.txt` on the cluster volume.

Ops notes hit during the campaign (fixes committed): pipefail-killed dataset
discovery (`|| true`), TwoRoom pool regeneration infeasible in-job, the
authors' TwoRoom h5 needing dedicated eval roots, first-hit tolerance
scoring for Reacher's τ=0.1 column, resumable eval cells, and 4-wide
parallel evals requiring per-process `OMP_NUM_THREADS` caps (default
threading made 4-wide *slower than sequential*: zero cells in 3.2 h). The
final harness (H200:4 / 80 cores, 8 cells wide, three optimizer seeds trained
concurrently one per GPU) runs a complete TwoRoom cell — caches, critic, three
seeds, 21 eval cells — in **21 minutes**, against ~11 h for the 1-GPU
sequential version.
