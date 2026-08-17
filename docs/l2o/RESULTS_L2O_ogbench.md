# L2O-MPC on OGBench Cube — results

Method and port: [README_l2o.md](README_l2o.md). Campaign index:
[RESULTS_L2O_20260816.md](RESULTS_L2O_20260816.md).

**Protocol.** DAgger imitation (beta = 0.8^k over 20 rounds) of an MPPI expert
(N = 512) through the frozen world model against the frozen `value_td` critic;
learner M = 64 samples x K = 4 iterations = 256 forward unrolls per decision.
Three optimizer seeds, report draws 42/43/44, 50 episodes per cell, held-out
task range (value/planner train on episodes 0–7999, eval draws 8000–9999).
`amax` matched to the other planner rows per base (LeWM 1.6, PLDM 4.5).
Horizons: h25 = `goal_offset_steps=25 budget=50`, h100 = 100/200.

## Success rate (%, h25 / h100)

| planner | LeWM | PLDM |
|---|---|---|
| **L2O-MPC** | **64.4 ± 6.0 / 50.0 ± 3.3** | **58.9 ± 0.8 / 45.3 ± 0.5** |
| in-cell value-MPPI (h25, draw 42 only) | 64.0 / – | 68.0 / – |
| latent CEM (replication table) | 78.0 / – | 62.7 / – |
| RLP | 86.9 / – | 82.0 / – |
| no-op floor | 54.0 | 54.0 |

Per-seed h25 means — LeWM 68.7 / 56.0 / 68.7, PLDM 58.7 / 60.0 / 58.0.

**Cube is L2O-MPC's failure mode.** It reaches parity with the MPPI it imitates
on LeWM (64.4 vs 64.0) and falls *below* it on PLDM (58.9 vs 68.0), with both
bases under latent CEM and ~10 points over the no-op floor. The training signal
says why: cube's DAgger imitation loss *rose* as beta decayed (0.57 → 0.87 on
seed 0), i.e. the learned update never fit the learner's own iterate
distribution in cube's 125-dimensional plan space at the paper's hyperparameters.
LeWM seed variance (±6.0) is the largest in the campaign, consistent with a
partially-fit optimizer. The paper's own escalation for harder tasks — hidden
2048/4096 instead of 1024 — is the natural follow-up arm and has not been run.

## Wall-clock speed (LeWM, 1×H200, fp32, whole decision, mean ms)

Measured in the real deployment path: each solver times its own `solve()`
(encode + plan + select), planners run sequentially on one node, warmup
decisions dropped per process. Harness
[scripts/sky/planner_speed.yaml](../../scripts/sky/planner_speed.yaml).

| planner | rollouts/decision | mode | B=1 | B=50 | B=50 per env |
|---|---|---|---|---|---|
| **L2O-MPC** | 256 fwd | graphed | 45.3 ¹ | 188.3 | 3.77 |
| L2O-MPC | 256 fwd | eager | 88.0 | 232.5 | 4.65 |
| **RLP / LIP** | 9 fwd + 8 bwd | graphed | ~30 ² | ~79 ³ | 1.58 |
| RLP / LIP | 9 fwd + 8 bwd | eager | 304.7 | 333.3 | 6.67 |
| **DMPO** | 256 fwd | eager (its best) | 83.3 ⁴ | 236.3 ⁴ | 4.73 |
| DMPO | 256 fwd | graphed | 126.4 ⁴ ⚠ | 916.6 ⁴ ⚠ | 18.33 |
| CEM | 9,000 fwd | eager | 390.6 ⁵ | 5,308.2 ⁵ | 106.2 |
| MPPI | 9,000 fwd | eager | 420.3 ⁵ | 10,020.3 ⁵ | 200.4 |
| Adam | 3,000 fwd + 3,000 bwd | eager | 959.3 ⁵ | 17,129.8 ⁵ | 342.6 |

¹ Median 30.9; the mean-vs-median gap is **encode** variance, not planning —
L2O's plan phase measured 21.0–21.2 ms across every decision (job 7771).
² From the 2026-08-12 audit (job 4840); this campaign's own RLP graphed B=1
samples were capture-contaminated (RLP terminates early at B=1, yielding 5–6
decisions, so one capture dominates) and are not used.
³ Steady state (median); the measured mean was 374.4 because a shrinking eval
batch re-captures.
⁴ Measured by the DMPO campaign, not here. ⚠ DMPO's graph capture *regresses*
its wall-clock, unlike L2O's.
⁵ Under the **learned-value objective** (`core/value=metric`), so each sampled
rollout also pays a critic forward; the audit's latent-objective CEM was 241.9
ms eager / 218.6 graphed. The samplers have no graphed path in this tree.

Calibration: the repo's published benchmark table lists 1-env wall-clock of
0.42 s (CEM) and 0.38 s (MPPI) against 0.391 / 0.420 measured here, so this is
timing the same quantity those numbers did.

**Reading.** RLP is fastest at both batch sizes once graphed (~30 / ~79 ms).
L2O's 256 rollouts cost only ~1.5× RLP's 17 at B=1 because rollout *count* is
the wrong currency: L2O's 64 plans are scored as one batch, so a decision is
~25 sequential world-model calls against RLP's ~85 (8 iterations of forward
*and* backward), and a call on 64 rows costs only ~1.6× a call on one row at
this model size. RLP scales better with batch (2.6× from B=1 to B=50 vs L2O's
4.2×) because its work stays one row deep while L2O's rows multiply to 3,200.

**CUDA-graph capture** (`core.solver.graphed=true`) for L2O is a plain
`CUDAGraph` replay, not LIP's `make_graphed_callables`: the decision is a single
fixed shape, forward-only, no gradient. `graphed=verify` measured
**max_deviation 0.000e+00** at B=1 and B=50 — bit-exact, where LIP's capture
(which must also record a backward) lands at 5.96e-8. Capture costs seconds per
distinct row count and a shrinking eval batch triggers fresh captures, so
graphed *means* over a full eval run are inflated relative to steady state.

## Provenance

Accuracy: jobs 7035/7037 (+ 7146/7147, shards 7285/7286), tag
`l2o-cube-{lewm,pldm}-20260815` on `/checkpoints/armin@pantheon.inc/`
(`value_td`, `l2o_s{0,1,2}.pt`, `results_*.txt`).
Speed: jobs 7739 / 7765 / 7766 / 7768 / 7769 / 7771.
Code: de4c165 (port), cb67c17 (`GraphedCost` + `core.solver.graphed` for L2O),
be32849 / c51e08b / 38d7c03 / 97e388d / 8f4316c (speed harness).
