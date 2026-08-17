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

## Per-environment records

The tables live in one file per environment; this file keeps the shared
protocol, the wall-clock benchmark, and the campaign provenance.

| environment | record | status |
|---|---|---|
| TwoRoom | [RESULTS_dmpo_tworoom.md](RESULTS_dmpo_tworoom.md) | current — held-out, environment action bounds, full 3x3 |
| Reacher | [RESULTS_dmpo_reacher.md](RESULTS_dmpo_reacher.md) | PLDM current; LeJEPA still on the superseded symmetric clip |
| OGBench Cube | [RESULTS_dmpo_ogbench_cube.md](RESULTS_dmpo_ogbench_cube.md) | superseded clip; reruns pending |

Headline across the campaign: **DMPO beats the same-critic MPPI update it
learns a residual on in 13 of 14 comparable cells** (the exception is Cube
PLDM, itself a clip confound) at 1/35th the rollouts, and **loses to RLP in 13
of 14** (the exception is TwoRoom LeJEPA h100, a weak RLP cell that value+CEM
also ties). Two documented findings were retracted during the campaign — a
TwoRoom split leak and a clipping artifact — both recorded in the
per-environment files rather than quietly corrected.

## Wall-clock per decision (measured 2026-08-17, job 7868)

OGBench Cube LeWM on one H200, `evaluation.num_episodes=50` (the protocol's
batch), h100 budget 500 so each arm yields 20 decisions. **Steady median**
excludes the first solve, which carries warmup and — in graphed mode — the
multi-second CUDA-graph capture; a mean over 8 solves is dominated by that
outlier and is how an earlier version of this table wrongly showed graphing as
a 3.9x regression.

| planner | mode | first solve | steady median | ms/env | rollouts/decision |
|---|---|---|---|---|---|
| **RLP / LIP** | graphed | 2182 | **80.1** | **1.60** | 9 fwd (+8 bwd) |
| RLP / LIP | eager | 747 | 319.5 | 6.39 | 9 fwd (+8 bwd) |
| **DMPO** | graphed | 2364 | 151.5 | 3.03 | 256 fwd |
| DMPO | eager | 651 | 159.1 | 3.18 | 256 fwd |
| CEM | eager, `batch_size=50` | 10342 | 1976.9 | 39.54 | 9,000 fwd |
| MPPI | eager, `batch_size=50` | 10390 | 4766.8 | 95.34 | 9,000 fwd |

Reading:

1. **Graph capture buys RLP 4.0x and DMPO 1.05x.** RLP's decision is 8
   sequential iterations of a 5-step unroll plus backward — launch-bound, which
   is what capture removes. DMPO is one wide forward batch of `B*N` plans (5
   dependent steps), so there is almost no launch overhead to recover. Graphed
   DMPO is bit-exact against eager (`max_difference = 0.000e+00` on every
   `graphed=verify` call).
2. **Best-config latency: RLP 1.60 vs DMPO 3.03 ms/env** — RLP 1.9x faster,
   on top of using 28x fewer rollouts. Eager-only, the ordering *reverses*
   (DMPO 3.18 vs RLP 6.39), so any latency claim must state the mode.
3. **Both learned planners are an order of magnitude faster than the sampling
   baselines**: 13-25x versus CEM, 31-60x versus MPPI.
4. The shipped `configs/core/solver/{cem,mppi}.yaml` set `batch_size: 1`, i.e.
   50 sequential single-env solves; the rows above use `batch_size=50`. As
   shipped, CEM costs 144 ms/env and MPPI 222 ms/env (job 7761) — a ~2.3x
   config penalty that applies to every baseline timing in this repository, not
   just this table.

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
