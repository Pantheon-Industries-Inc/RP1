# L2O-MPC on Reacher — results

Method and port: [README_l2o.md](README_l2o.md). Campaign index:
[RESULTS_L2O_20260816.md](RESULTS_L2O_20260816.md).

**Protocol.** DAgger imitation (beta = 0.8^k over 20 rounds) of an MPPI expert
(N = 512) through the frozen world model against the frozen critic; learner
M = 64 samples x K = 4 iterations = 256 forward unrolls per decision. Three
optimizer seeds, report draws 42/43/44, 50 episodes per cell, held-out task
range. First-hit scoring at h25 with an explicit tolerance
(`environment.success_threshold`), reported at tau = 0.1 and tau = 0.05.
`amax` 2.2 (LeJEPA) / 1.8 (PLDM), matching the RLP cells.

Protocol caveat: the earlier Reacher tables in this repository report draws
42–47 (six draws); everything here is 42–44.

## Window-3 critic (success %, tau = 0.1 / tau = 0.05)

The protocol-matched family: window quasimetric, expectile 0.05, lag 5.

| planner | LeJEPA | PLDM |
|---|---|---|
| **L2O-MPC** | **93.3 ± 1.1 / 74.9 ± 2.3** | **94.7 ± 0.0 / 77.3 ± 1.4** |
| in-cell value-MPPI (draws 42/43) | 85.0 / 66.0 | 87.0 / 64.0 |
| value-w3 MPPI (replication, 6 draws) | 86.0 / 66.0 | 83.7 / 61.7 |
| value-w3 CEM (replication) | 99.3 / 89.3 | 98.3 / 84.7 |
| latent-w3 CEM (replication) | 99.0 / 94.3 | 98.3 / 89.3 |
| RLP | 99.9 / 97.1 | 99.4 / 91.2 |

**L2O-MPC beats the MPPI it imitates in all four cells** (+7 to +13 points) at
35× fewer rollouts, and the ruler is calibrated: the in-cell controls run here
(85.0/66.0, 87.0/64.0) agree with the prior campaign's independent six-draw
MPPI rows (86.0/66.0, 83.7/61.7) to within 1–3 points. It does not reach CEM at
the same 9,000-rollout budget, and RLP leads every cell.

Seed variance is remarkably small — PLDM tau=0.1 is **exactly 94.7 on all three
seeds** — which follows from imitating a deterministic expert, and contrasts
with RLP's actor-seed spread of several points.

## Window-1 ablation (success %, tau = 0.1 / tau = 0.05)

Identical recipe with a single-frame critic in place of the three-frame one;
`amax` held at 2.2/1.8 so the critic window is the only difference.

| planner | LeJEPA | PLDM |
|---|---|---|
| **L2O-MPC, window-1** | **90.9 ± 2.6 / 69.6 ± 2.2** | **89.3 ± 2.7 / 60.9 ± 1.7** |
| L2O-MPC, window-3 (above) | 93.3 ± 1.1 / 74.9 ± 2.3 | 94.7 ± 0.0 / 77.3 ± 1.4 |
| *window-1 penalty* | *−2.4 / −5.3* | *−5.4 / −16.4* |
| value-w1 MPPI (App C.3 replication) | 74.0 / 42.0 | 60.0 / 38.7 |
| value-w1 CEM (App C.3 replication) | 97.3 / 82.0 | 96.0 / 76.0 |

Two readings.

**(a) The penalty is tolerance-dependent** — 2–5 points at tau 0.1 but 5–16 at
tau 0.05. That is the signature of a critic that cannot represent arrival
velocity: hitting a target loosely tolerates a bad approach, hitting it tightly
does not. The RLP window ablation found the same mechanism (~+9 for window-3 at
the tight tolerance). PLDM is hit hardest, as it is wherever information is
reduced.

**(b) L2O's margin over its expert *widens* under the weaker critic** —
+27.6 / +22.2 at tau 0.05 (vs w1 MPPI) against +8.9 / +13.3 at window-3. DAgger
fits the learned update to whatever cost it is handed, so it degrades more
gracefully than the fixed MPPI reduction does. Both w1 jobs trained cleanly
(imitation loss 0.58 → 0.05 LeJEPA, 0.34 → 0.07 PLDM), so this is not an
underfitting artifact. It still does not reach CEM at either window.

Taken with the OGBench result, this is the campaign's most transferable lesson:
**L2O-MPC loses least where the objective is degraded and most where the search
is hard** (cube, where it only matches MPPI).

## Provenance

Window-3: jobs 7040 / 7041 (+ 7150 / 7151), tags
`l2o-reacher-{lejepa,pldm}-20260815`.
Window-1: jobs 7342 / 7343, tags `l2o-reacher-{lejepa,pldm}-w1-20260816`.
Artifacts on `/checkpoints/armin@pantheon.inc/<tag>/`: `value_td`,
`l2o_s{0,1,2}.pt` with value siblings, `results_*.txt` per cell.
Code: de4c165 (port), a91a35c (Reacher wiring: fetched dataset, window critic,
first-hit at both tolerances).

The public Reacher h5 pads every episode's terminal step with NaN actions; the
shared `WindowSampler` is nan-aware, which is why these runs trained rather
than silently producing NaNs.
