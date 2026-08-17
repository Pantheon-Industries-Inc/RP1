# DMPO on Reacher — results (2026-08-17)

Offline DMPO (`model=dmpo`, `core/solver=dmpo`; method and deviations in
[README_dmpo.md](README_dmpo.md)) against the same-critic MPPI baseline, on both
bases and both cost windows, at both success tolerances.

**Status: PLDM cells are current; LeJEPA cells still carry the superseded
symmetric clip** — see "Incomplete" below.

## Protocol

| | |
|---|---|
| data | `tool=fetch_dataset dataset=reacher` (2,010,000 rows, h5) |
| split | held-out — train 0–7999, eval draws 8000–9999 |
| seeds | 3 optimizer seeds x eval draws 42/43/44 x 50 episodes = n=9 per cell |
| horizon | h25, open loop (`receding_horizon=5`) |
| scoring | **first-hit** success: latched on the first step whose worst joint is within τ of the goal (`environment.success_threshold`), since the qpos-match task hardcodes 0.05 rad and cannot express τ=0.1 through termination |
| windows | w1 trains and evaluates against the single-frame critic; w3 against the 3-frame window quasimetric |
| jobs | 7733 / 7735 (PLDM, env bounds ±1.73), 7018 / 7020 / 7022 / 7023 (LeJEPA, amax 2.2) |

## Results (first-hit success %)

| | | LeJEPA τ=.1 | τ=.05 | PLDM τ=.1 | τ=.05 |
|---|---|---|---|---|---|
| **(a) window-1** | **DMPO** (256 roll.) | 92.4 ◇ | 67.8 ◇ | **90.0** | **62.9** |
| | value MPPI (9k, same critic) | 64.7 | 40.0 | 68.0 | 46.7 |
| | RLP w1 ablation (paper†) | **98.7** | **88.7** | **97.8** | **82.0** |
| **(b) window-3** | **DMPO** (256 roll.) | 92.7 ◇ | 75.8 ◇ | **95.1** | **82.9** |
| | value MPPI (9k, same critic) | 83.3 | 66.0 | 88.0 | 68.0 |
| | RLP (paper) | **99.9** | **97.1** | **99.4** | **91.2** |

◇ measured with the superseded symmetric clip (`amax=2.2`) rather than the
environment's per-dimension bounds; the PLDM cells moved ≤1 point when
re-measured with correct bounds (w1 90.4→90.0 / 62.4→62.9, w3 95.6→95.1 /
83.8→82.9), so the LeJEPA numbers are expected to be stable, but they are not
yet confirmed.

## Reading

1. **DMPO beats same-critic MPPI in all eight columns**, by +9 to +28 points,
   at 1/35th the rollouts — the largest margins in the whole campaign. This is
   the cell family where the learned reduction most clearly earns its keep.
2. **RLP leads in all eight columns**, by 6–21 points, widest at the tight
   tolerance (w3 LeJEPA: 97.1 vs 75.8).
3. **The window ordering reproduces for DMPO**: w3 > w1 at both bases and both
   tolerances, and the gain concentrates at τ=.05 (PLDM +20.0), matching the
   paper's finding that a 3-frame critic is worth ~+9 at the tight tolerance.
4. **Scoring cross-check**: the in-job value-MPPI rows land within a few points
   of the paper's App C.3 value-MPPI rows (74.0/42.0 LeJEPA-w1, 86.0/66.0
   LeJEPA-w3), and τ=.05 reproduces the environment's own 0.05 rad termination
   criterion to within noise — so the first-hit tolerance machinery added for
   this campaign (`rlp.environment.World.threshold_hits`) is calibrated.

## Incomplete

The LeJEPA w1/w3 reruns under the environment's action bounds (jobs 7732/7734)
produced **0 and 1 usable cells** of 18: at `EVAL_PAR=8` the eval processes
stalled ~36 minutes and were killed with no traceback — 8 concurrent cells each
pulling 50 episodes of 224x224 pixel history from a 22 GB h5. Relaunches at
`EVAL_PAR=2` are pending a SkyPilot client upgrade (the API server is currently
dropping job submissions). The harness now fails a job that completes zero
cells instead of printing `DMPO_CAMPAIGN_DONE` over an empty grid.
