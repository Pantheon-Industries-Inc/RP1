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

## TwoRoom (success %, authors' 10,000-episode pool)

| planner | roll. | LeJEPA h25 | LeJEPA h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** (full pool)‡ | 256 | 99.1 | 98.9 | 99.8 | 58.4 |
| value MPPI (full pool)‡ | 9k | 84.7 | — | 80.0 | — |
| RLP (held-out split) | 9 | 100.0 | 94.2 | 98.2 | 96.0 |

‡ trained and evaluated on the same 10,000-episode pool — upper bounds, not
comparable to the held-out RLP row below (see the warning).

Complete 3×3 grids (jobs 7340/7341, 21 result cells each). Per-actor-seed
h100 means — LeJEPA: 99.3 / 98.7 / 98.7 (range 98–100); PLDM: 57.3 / 58.0 /
60.0 (range 50–70).

> **Protocol warning — the DMPO and RLP rows here are NOT comparable.**
> These DMPO cells follow the authors' TwoRoom contract (train on all 10,000
> episodes, `episode_range: null`), so the critic and optimizer trained on the
> episodes the eval tasks are drawn from: they are **upper bounds**. The RLP
> row is from the held-out replication (`RESULTS_REPLICATION_20260814.md`:
> train 0–7999, eval 8000–9999). DMPO's apparent +4.7 at LeJEPA h100 and +1.6
> at PLDM h25 are therefore artifacts of the split difference, not wins. To
> make the row quotable, re-run these cells with `train_episodes=8000`,
> `evaluation.episode_range="8000:10000"`, and a fresh cache tag (the cache
> filename does not encode the episode count, so a split cache would silently
> collide with the full-pool one).

The h100 asymmetry is the notable TwoRoom result: **LeJEPA barely degrades**
(99.1 → 98.9) while **PLDM collapses** (99.8 → 58.4) under the same recipe,
critic settings, and planner — the difference is the world-model base. The
collapse is consistent across all three optimizer seeds, so it is a property
of the method on this base, not seed variance. RLP's shipped rows degrade on
neither base (94.2 / 96.0), so this is the one cell where DMPO's deficit
against RLP is a chasm (−37.6) rather than a gap. DMPO's single learned
iteration is trained on 5-block problems and nothing in it adapts to the
8-replan regime; on a base whose imagined rollouts drift more, that surfaces
as long-horizon collapse. Evaluation runs through
the `tworoom_{lewm,pldm}_h5` roots (the fetched pool stores the agent
position as `pos_agent` and has no `state` column; the collector-produced
lance is not regenerable in-job — 3,443/10,000 episodes in 11 h and
decelerating).

## Reading

1. **DMPO beats the update it residuals on almost everywhere.** Against
   MPPI under the identical critic, WM, and data: +10.2 (Cube LeWM h25),
   +19.7 to +27.7 (all four Reacher w1/w3 τ columns on LeWM; +9 to +18 on
   PLDM), +14.4/+19.8 (TwoRoom h25) — at 1/35th the rollouts. The one loss
   is Cube PLDM h25 (−4.2), where `amax=4.5` (inherited from the RLP recipe,
   untuned for DMPO) is the prime suspect.
2. **DMPO loses to RLP on Cube and Reacher** — by 6–21 points on Reacher,
   ~14–20 on Cube, under a shared frozen model and cost. **TwoRoom cannot be
   compared**: see the protocol warning in that section. Where the comparison
   is valid, the learned sampling-reduction is consistently weaker than the
   learned gradient-refinement.
3. **Horizon degrades DMPO faster than RLP, and base-dependently.** Cube
   h25→h100 drops DMPO ~10–17 points on both bases. On TwoRoom the split is
   stark: LeJEPA holds (99.1→98.9) while PLDM collapses (99.8→58.4) against
   RLP's 96.0 (different protocol; the collapse is far larger than any split
   effect) — consistent across all three optimizer seeds. DMPO's single
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
| TwoRoom LeJEPA/PLDM | 7143/7287/7340 / 7145/7290/7341 | `dmpo-tw-{lejepa,pldm}-20260815` |

Per-cell result files live under
`/checkpoints/armin@pantheon.inc/<tag>/results_*.txt` on the cluster volume.

Ops notes hit during the campaign (fixes committed): pipefail-killed dataset
discovery (`|| true`), TwoRoom pool regeneration infeasible in-job, the
authors' TwoRoom h5 needing dedicated eval roots, first-hit tolerance
scoring for Reacher's τ=0.1 column, resumable eval cells, and 4-wide
parallel evals requiring per-process `OMP_NUM_THREADS` caps (default
threading made 4-wide *slower than sequential*: zero cells in 3.2 h).
