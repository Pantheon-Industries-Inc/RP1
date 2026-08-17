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

Rollout budgets per decision: **DMPO 256 forward** (256 samples × 1 learned
iteration, no backward); **MPPI 9,000 forward** (300 × 30); **RLP 9 forward**
— 8 refinement iterations, each reusing its single unroll for both the value
gradient and the actor's features, plus one selection unroll — with 8 backward
passes over those same graphs.

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
| **DMPO** (env action bounds) | 256 | 96.9 | **100.0** | 97.8 | 92.7 |
| *DMPO (amax=2.5, superseded)* | *256* | *96.7* | *99.1* | *97.3* | *54.4* |
| value MPPI | 9k | 83.3 | — | 76.0 | — |
| RLP (held-out) | 9 | **100.0** | 94.2 | **98.2** | **96.0** |
| value CEM (remeasured 2026-08-17) | 9k | 100.0 | 94.7 | 100.0 | 88.7 |

An earlier version of these cells ran the authors' full-pool contract
(train and eval on all 10,000 episodes) and was therefore an upper bound:
99.1 / 98.9 / 99.8 / 58.4, with MPPI at 84.7 / 80.0. The leak was worth
~2–3 points at h25 on both bases. `TWOROOM_SPLIT=0` reproduces it; the cache
name encodes the split so the two cannot collide.

> **RETRACTED (2026-08-17): the "PLDM collapses at h100" finding was a
> clipping artifact, not a property of the base.** Those cells clipped plans
> to a symmetric `amax=2.5` inherited from the RLP recipe, while TwoRoom's real
> action range in z-scored units is ±1.4. DMPO was therefore proposing actions
> the environment cannot execute *and that the world model never saw in
> training*; the resulting out-of-distribution rollout error compounded across
> the 8 replans of h100. Clipping to the environment's own per-dimension bounds
> (`action_range=1.0`, jobs 7730/7731) recovers PLDM h100 from **54.4 to 92.7**
> (+38.3) and leaves every other TwoRoom cell within ~1 point. Reacher moved
> ≤1 point on all four τ cells because its inherited `amax=1.8` was already
> close to its true bound. Lesson: for a *sampling* planner the clip is part of
> the dynamics contract, not a regularizer — RLP's `amax` is a tuned residual
> trust region and must not be reused as a bound.

One finding survives the corrected protocol:
2. **TwoRoom LeJEPA h100 is a weak RLP cell, and DMPO is the evidence.**
   Under the identical held-out protocol RLP scores 94.2 against DMPO's 99.1
   (+4.9) — the only cell in this campaign where DMPO beats RLP. value+CEM
   *ties* RLP there (94.7 remeasured over six shards; the banked 96.0 did not
   reproduce), so DMPO's margin is the only surviving evidence that the cell
   has headroom (see
   [../campaigns/2026-08-17/RESULTS_tworoom_improvement.md](../campaigns/2026-08-17/RESULTS_tworoom_improvement.md)).

   The mechanism established there is a value-scale sensitivity specific to
   RLP's architecture, not a data-range problem: the LIPv4 actor consumes the
   raw critic value `E` as its only goal-distance channel, unnormalised, and
   with `gamma=1.0` h100 presents an `E` ~4x outside its training band, driving
   plan entries onto the clip boundary. **DMPO is structurally immune** because
   it z-scores the cost vector before the actor sees it (and MPPI only ranks
   it) — which is why DMPO holds h100 while training at the *same*
   `max_delta=12`. My earlier "trained on nearer goals" hypothesis was the
   rival explanation and is ruled out by exactly that fact.

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
3. **Horizon degradation is mostly an artifact of the action clip.** With the
   environment's own bounds, TwoRoom h25→h100 costs DMPO nothing on LeJEPA
   (96.9→100.0) and ~5 points on PLDM (97.8→92.7) — against 54.4 under the
   inherited symmetric clip. The Cube h100 numbers in this record still use
   the old clip and are being re-measured (jobs 7728/7729); expect them to
   move up, since Cube LeWM's `amax=1.6` was *narrower* than its true bound
   while PLDM's 4.5 was near it.
4. **DMPO cannot be run at RLP's rollout budget, and loses at 28x it.**
   RLP spends 9 forward unrolls per decision; DMPO spends 256 (its reference
   operating point) and loses in 13 of 14 comparable cells. A rollout-matched
   DMPO row is not merely unrun but ill-posed: 9 samples in a 125-dimensional
   plan space (Cube) is fewer samples than dimensions, below what a Gaussian
   sampler can estimate at all. That asymmetry — a gradient refiner needs
   O(10) unrolls where a sampler needs O(100+) — is a structural property, not
   a tuning gap. (Counting RLP's 8 backward passes at the textbook ~2x forward
   would put it near ~25 forward-equivalents; wall-clock measurement is the
   honest arbiter and is pending, job 7737.)
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
