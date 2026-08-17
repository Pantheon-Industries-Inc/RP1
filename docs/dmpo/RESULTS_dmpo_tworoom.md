# DMPO on TwoRoom — results (2026-08-17)

Offline DMPO (`model=dmpo`, `core/solver=dmpo`; method and deviations in
[README_dmpo.md](README_dmpo.md)) against the same-critic MPPI baseline on both
TwoRoom bases. **Complete and current**: held-out split, environment action
bounds, full 3x3 grid.

## Protocol

| | |
|---|---|
| data | authors' fetched pool (`tool=fetch_dataset dataset=tworoom`, 10,000 episodes, h5) |
| split | held-out — value/optimizer train on episodes 0–7999, eval draws 8000–9999 |
| seeds | 3 optimizer seeds (0/1/2) x eval draws 42/43/44 x 50 episodes = n=9 per cell |
| horizons | h25 (`goal_offset_steps=25 budget=50`), h100 (`=100`/`=500`) |
| action bounds | the environment's own per-dimension limits, ±1.4 in z-scored units |
| eval roots | `tworoom_{lewm,pldm}_h5` — the fetched pool stores the agent position as `pos_agent` and has no `state` column |
| jobs | 7730 / 7731 (bounds), 7689 / 7690 (split), 21 result cells each |

## Results (success %)

| planner | roll./decision | LeJEPA h25 | LeJEPA h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** | 256 | 96.9 | **100.0** | 97.8 | 92.7 |
| value MPPI (same critic) | 9,000 | 83.3 | — | 76.0 | — |
| RLP (held-out record) | 9 | **100.0** | 94.2 | **98.2** | **96.0** |
| value CEM (remeasured 2026-08-17) | 9,000 | 100.0 | 94.7 | 100.0 | 88.7 |

## Reading

1. **DMPO beats the update it residuals on by +13.6 / +21.8** at h25 (96.9 vs
   83.3; 97.8 vs 76.0) using 1/35th the rollouts.
2. **RLP leads on three of four cells** — h25 on both bases, and PLDM h100 by
   3.3.
3. **TwoRoom LeJEPA h100 is a weak RLP cell**: RLP 94.2 against DMPO's 100.0.
   value+CEM *ties* RLP there (94.7 remeasured over six shards; the previously
   banked 96.0 did not reproduce), so DMPO's +5.8 is the only clear evidence
   that the cell has headroom. The mechanism is established in
   [../campaigns/2026-08-17/RESULTS_tworoom_improvement.md](../campaigns/2026-08-17/RESULTS_tworoom_improvement.md):
   the LIPv4 actor consumes the raw critic value `E` as its only goal-distance
   channel, unnormalised, and with `gamma=1.0` h100 presents an `E` ~4x outside
   its training band. **DMPO is structurally immune** because it z-scores the
   cost vector before the actor sees it, which is why it holds h100 while
   training at the same `max_delta=12`.

## Retracted finding

An earlier version of this record reported **PLDM h100 = 54.4** and framed it as
a base-specific long-horizon collapse. That was a **clipping artifact**: those
cells clipped plans to a symmetric `amax=2.5` inherited from the RLP recipe,
while TwoRoom's real action range is ±1.4. DMPO was proposing actions the
environment cannot execute *and that the world model never saw in training*, and
the out-of-distribution rollout error compounded over h100's 8 replans.
Clipping to the environment's bounds recovers **54.4 → 92.7** (+38.3) and moves
every other TwoRoom cell by ≤1 point.

Superseded numbers, for the record: 96.7 / 99.1 (LeJEPA h25/h100), 97.3 / 54.4
(PLDM), MPPI 84.7 / 80.0. `TWOROOM_SPLIT=0` reproduces the authors' full-pool
contract, whose numbers are upper bounds (the leak was worth ~2–3 points at
h25) and are **not** comparable to the held-out RLP rows.

## Lesson for baseline ports

For a *sampling* planner the action clip is part of the dynamics contract, not a
regularizer: outside the environment's range the world model is off-distribution
and multi-replan horizons amplify it. RLP's `amax` is a tuned residual trust
region and must not be reused as a bound —
`core.planner.action_range` now derives the bound from the dataset's action
statistics instead.
