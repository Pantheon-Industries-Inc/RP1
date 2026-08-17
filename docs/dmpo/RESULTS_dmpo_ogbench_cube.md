# DMPO on OGBench Cube — results (2026-08-16)

Offline DMPO (`model=dmpo`, `core/solver=dmpo`; method and deviations in
[README_dmpo.md](README_dmpo.md)) against the same-critic MPPI baseline on both
Cube bases, at both horizons.

**Status: superseded action clip.** These cells clip plans to a symmetric
`amax` inherited from the RLP recipe (LeWM 1.6, PLDM 4.5) rather than the
environment's own per-dimension action bounds. On Cube those bounds are
[-3.50, +3.43], [-2.54, +2.55], [-1.56, +1.55], [-2.55, +2.55], [-4.64, +3.37]
in z-scored units — so **`amax=4.5` binds on only 1 of 5 dimensions (effectively
no clip) while `amax=1.6` binds on 4 of 5**. The two bases were therefore
searched over qualitatively different action sets, and the LeWM-vs-PLDM
comparison below is confounded. Reruns are pending (see "Incomplete").

## Protocol

| | |
|---|---|
| data | `tool=fetch_dataset dataset=ogb_cube` (~20 GiB, 2,010,000 rows) |
| split | held-out — value/optimizer train on episodes 0–7999, eval draws 8000–9999 |
| seeds | 3 optimizer seeds (0/1/2) x eval draws 42/43/44 x 50 episodes = n=9 per cell |
| horizons | h25 (`goal_offset_steps=25 budget=50`), h100 (`=100`/`=200`) |
| jobs | 6758 / 6759 (train + h25), 7025 / 7026 / 7152 / 7153 (h25 + h100) |

## Results (success %)

| planner | roll./decision | LeWM h25 | LeWM h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** | 256 | **72.9** | **55.8** | 61.8 | 51.6 |
| value MPPI (same critic) | 9,000 | 62.7 | 50.0 * | **66.0** | 50.0 * |
| RLP (repo record, h25) | 9 | **86.9 ± 3.0** | — | **82.0** | — |
| no-op floor (record) | — | 54.0 | — | 54.0 | — |

\* 1–2 eval seeds only, measured before h100 MPPI was dropped from the grid
(a cell is 8 replans x 9,000 rollouts x 50 episodes); footnote-grade.

Per-seed h25 spread — LeWM 72.0 / 73.3 / 73.3, PLDM 60.7 / 62.7 / 62.0: tight
on both bases, so seed variance is not what separates the cells.

## Reading

1. **LeWM: DMPO beats same-critic MPPI by +10.2** (72.9 vs 62.7) at 1/35th the
   rollouts. **PLDM: DMPO loses by 4.2** (61.8 vs 66.0) — the only MPPI loss in
   the campaign, and the prime suspect is the clip confound above (PLDM ran
   effectively unclipped at `amax=4.5` with `init_std=1.0`).
2. **RLP leads both bases by ~14–20 points** at h25 while using 9 rollouts per
   decision against DMPO's 256.
3. **Both DMPO cells clear the no-op floor** (54.0) at h25; at h100 LeWM stays
   above it (55.8) and PLDM is at it (51.6), so the h100 numbers should be
   treated as near-floor until re-measured.
4. Horizon costs DMPO ~10–17 points on both bases. On TwoRoom the equivalent
   drop turned out to be largely a clipping artifact
   ([RESULTS_dmpo_tworoom.md](RESULTS_dmpo_tworoom.md)), which is a further
   reason to re-measure these cells before reading anything into them.

## Incomplete

The action-bounds reruns (jobs 7728/7729) produced **0 and 1 usable cells** of
18: at `EVAL_PAR=8` the eval processes stalled ~36 minutes and were killed with
no traceback — 8 concurrent cells each pulling 50 episodes of 224x224 pixel
history from a 20 GB dataset. Relaunches at `EVAL_PAR=2` are pending a SkyPilot
client upgrade (the API server is currently dropping job submissions).

Expect movement, particularly on LeWM: its `amax=1.6` was the furthest from the
true bound of any cell in the campaign, and on TwoRoom correcting a wrong clip
moved one cell by +38 points.

## Wall-clock

The campaign's timing benchmark ran on this environment; see
[RESULTS_dmpo_20260816.md](RESULTS_dmpo_20260816.md) for the measured
per-decision latencies (graphed RLP 1.60, graphed DMPO 3.03, CEM 39.5,
MPPI 95.3 ms/env at B=50).
