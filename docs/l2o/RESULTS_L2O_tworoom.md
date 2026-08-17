# L2O-MPC on TwoRoom — results

Method and port: [README_l2o.md](README_l2o.md). Campaign index:
[RESULTS_L2O_20260816.md](RESULTS_L2O_20260816.md).

**Protocol.** DAgger imitation (beta = 0.8^k over 20 rounds) of an MPPI expert
(N = 512) through the frozen world model against the frozen `value_td` critic;
learner M = 64 samples x K = 4 iterations = 256 forward unrolls per decision.
Three optimizer seeds, report draws 42/43/44, 50 episodes per cell, `amax` 2.5
on both bases. Horizons: h25 = `goal_offset_steps=25 budget=50`, h100 = 100/200.

Evaluation uses the authors' fetched 10,000-episode pool
(`tool=fetch_dataset dataset=tworoom`) through the `tworoom_*_h5` eval roots.
Per the shipped TwoRoom contract, training and evaluation share that pool (no
episode split), so these rows — like every other TwoRoom row in this
repository — are formally upper bounds; the within-table comparison is
unaffected because all planners draw tasks from the same pool.

## Success rate (%, h25 / h100)

| planner | LeJEPA | PLDM |
|---|---|---|
| **L2O-MPC** | **95.8 ± 1.4 / 90.9 ± 2.5** | **95.1 ± 0.8 / 48.2 ± 1.3** |
| in-cell value-MPPI (h25, draw 42 only) | 90.0 / – | 86.0 / – |
| value MPPI (replication table) | 87.3 / 64.0 | 78.7 / 58.7 |
| value CEM (replication table) | 100.0 / 96.0 | 100.0 / 89.3 |
| latent Adam (replication table) | 94.7 / 24.0 | 90.7 / 42.0 |
| RLP | 100.0 / 94.2 | 98.2 / 96.0 |

Per-seed means — LeJEPA h25 94.0/97.3/96.0, h100 88.0/90.7/94.0;
PLDM h25 95.0/92.0/92.0, h100 47.3/47.3/50.0.

## Reading

**LeJEPA h100 is the campaign's headline cell: 90.9 against the hand-written
value-MPPI's 64.0** — +26.9 points at 35× fewer rollouts (256 vs 9,000), and
within ~3 points of RLP's 94.2. This is the clearest case of the paper's claim
transferring: a learned update rule, imitated offline from a many-sample
expert, retains that expert's quality at a fraction of the search budget. The
h25 cells confirm it more modestly (+5.8 LeJEPA, +9.1 PLDM over the in-cell
control), where the ceiling leaves little room.

**PLDM h100 collapses to 48.2 while LeJEPA holds at 90.9.** The critic is not
the cause: value-CEM reaches 89.3 and RLP 96.0 on that same base with that same
critic family. What differs is the search budget — 256 samples versus CEM's
9,000 — and PLDM's long-horizon landscape punishes it specifically, which its
latent-objective rows (33–52 at h100) independently show. So the failure is the
sample budget interacting with that base's geometry, not the objective and not
the imitation.

L2O-MPC never overtakes value CEM on TwoRoom (100.0 / 96.0 LeJEPA), so the
ordering here is RLP ≈ CEM > L2O > MPPI.

## Provenance

Jobs 7036 / 7038 (+ 7148 / 7149 and per-(draw, seed) shards 7236–7250), tags
`l2o-tworoom-{lejepa,pldm}-20260815` on `/checkpoints/armin@pantheon.inc/`
(`value_td`, `l2o_s{0,1,2}.pt` with value siblings, `results_*.txt` per cell).
Code: de4c165 (port), 62724d2 (resume-safe evals, which made the h100 tail
shardable).

Ops note for anyone rerunning this: h100 cells cost 10–20× h25 because failing
episodes burn the full 200-step budget while successful h25 episodes stop
early, and eval nodes on this cluster varied ~6× in env-stepping speed
(score-neutral, wall-clock fatal). Sharding one job per (base, draw, seed) cut
a ~30 h serial tail to ~5 h.
