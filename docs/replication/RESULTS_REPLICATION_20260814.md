# Replication campaign results — 2026-08-13/14

Every cell below is held-out (value/planner training on episodes 0–7999,
evaluation task draws from 8000–9999), 50 episodes per cell. RLP rows use the
checkpoint-verified paper recipes; baseline rows use CEM/MPPI 300×30 and
Adam(AdamW) 100×30. TwoRoom/Cube report draws 42–44; Reacher reports draws
42–47. RLP additionally averages actor training seeds (Cube/TwoRoom 3,
Reacher 6; Cube PLDM is single-seed). The vendored checkpoints behind the RLP
rows are in `assets/core/planner/`; producing jobs: cube 5959/5960/6082/6083,
tworoom 5961/5962 (+ baselines 6315/6316), reacher 6088–6092
(+ baselines 6375–6378).

## OGBench Cube, h25 (success %)

| cell | LeWM | PLDM | paper quote |
|---|---|---|---|
| **RLP** | **86.9 ± 3.0** (seeds 86.0/86.7/88.0) | **82.0** (1 seed) | 89.1 / 82.9 |
| latent + CEM | 78.0 | 62.7 | 74.0 / 62.7 |
| no-op floor | 54.0 | 54.0 | 56.0 |

## TwoRoom (success %, h25 / h100)

| objective + planner | LeWM | PLDM | paper (Tab. 1 / App C.1) |
|---|---|---|---|
| latent CEM | 84.0 / 13.3 | 93.3 / 52.0 | 89.3 / 15.3 · 96.7 / 54.7 |
| latent MPPI | 70.7 / 20.0 | 64.0 / 33.3 | 65.3 / 20.7 · 71.3 / 38.7 |
| latent Adam | 94.7 / 24.0 | 90.7 / 42.0 | 92.0 / 29.3 · 92.0 / 44.0 (main text) |
| value CEM | 100.0 / 96.0 | 100.0 / 89.3 | 100.0 / 95.3 · 98.7 / 90.0 |
| value MPPI | 87.3 / 64.0 | 78.7 / 58.7 | 87.1 / 63.3 · 83.1 / 56.7 |
| value Adam | 96.7 / 83.3 | 96.0 / 73.3 | 96.0 / 78.7 · 96.0 / 63.3 |
| **RLP** | **100.0 / 94.2** | **98.2 / 96.0** | 100.0 / 98.0 · 97.8 / 99.1 |

Notes: the split value rows replicate the appendix's full-pool quotes closely
(value+CEM within 1 pt at h100 on both bases) — the appendix value rows were
not meaningfully leak-inflated. The paper's appendix PLDM latent-Adam h100
cell (25.3) is wrong; the main-text 44.0 is correct (measured 42.0).

## Reacher, first-hit (success %, τ=0.1 / τ=0.05)

| objective + planner | LeWM | PLDM | paper (Tab. 2 / App C.3) |
|---|---|---|---|
| latent (single-frame) CEM | 98.7 / 80.3 | 96.7 / 80.0 | 97.3 / 80.7 · 97.3 / 80.0 |
| latent MPPI | 63.7 / 39.3 | 64.7 / 35.7 | 62.7 / 42.7 · 63.3 / 36.7 |
| latent Adam | 94.0 / 66.0 | 94.3 / 66.0 | 91.3 / 71.3 · 94.7 / 68.7 |
| value (3-frame w., e0.05) CEM | 99.3 / 89.3 | 98.3 / 84.7 | 97.3 / 82.0 · 96.0 / 76.0 |
| value MPPI | 86.0 / 66.0 | 83.7 / 61.7 | 74.0 / 42.0 · 60.0 / 38.7 |
| value Adam | 98.3 / 81.0 | 97.3 / 76.7 | 88.0 / 64.7 · 92.7 / 66.7 |
| **RLP** (6 seeds × 6 draws) | **99.9 ± 0.3 / 97.1 ± 2.8** | **99.4 ± 1.0 / 91.2 ± 6.7** | 98.2 / 89.8 · 97.8 / 82.0 |
| latent 3-frame-window CEM (extra) | 98.7 / 94.3 | — | not a paper row |

Notes: the latent baselines replicate the paper to within noise — PLDM latent
CEM τ=0.05 is exact (80.0 vs 80.0) — confirming the paper's latent cost is
single-frame terminal L2 and that the eval ruler is calibrated; the RLP gains
over the quoted numbers are therefore real. The paper's *value* baseline rows
came from a different (older) critic; the rows above use the same window-3
e0.05 value family the RLP actors train against and are the protocol-matched
replacements. The 3-frame-window *latent* cost (94.3 at τ=0.05) is a strong
baseline the paper does not report.

## Reacher window ablation (recovered from W&B, campaign rlp12_reacher_w1)

Previously documented only as a comment in the campaign yaml; recovered
2026-08-14 from the `rs_w1diag_*` runs (PLDM, clean split, 3 seeds × draws
42–44, first-hit rh5). Hypers: mean-weight 0.3, actor-LR 1e-4, replay 0.5,
max-delta 12, steps 6000, batch 128*, K=8, H=5, **window-1 value** (single
frame) for both the init value and the co-trained critic; grid over
expand-weight {0, 1} × amax {1.0, 1.8}.

| window-1 RLP arm (PLDM) | τ=0.1 | τ=0.05 |
|---|---|---|
| expand 0, amax 1.0 | 96.2 | 79.3 |
| expand 0, amax 1.8 | 97.8 | 82.0 |
| expand 1, amax 1.0 | 87.8 | 64.2 |
| expand 1, amax 1.8 | 89.6 | 62.9 |
| (reference) window-3 RLP | 99.4 | 91.2 |
| (reference) window-1 value + CEM | 96.0 | 76.0 |

Reading: the "w1 collapse" is specifically the **expand-weight × window-1
interaction** (value expansion bootstraps the critic on imagined terminals
whose arrival velocity a single frame cannot represent → actor/critic
co-exploitation). With expand=0, window-1 RLP is not collapsed — it sits at
value-baseline level (~82 at τ=0.05) — and the window-3 critic is worth
roughly +9 points at the tight tolerance on top of that.

## Provenance of the paper's App C.3 value baseline rows

The `rlp12-measure-reacher-*-value-window1-r2` runs reproduce the paper's
App C.3 value rows **exactly** (LeWM CEM 97.3/82.0, MPPI 74.0/42.0, Adam
88.0/64.7; PLDM CEM 96.0/76.0): those rows are **window-1 value** planners.
The protocol-matched window-3 value rows measured in this campaign (above)
are uniformly stronger and are the fair comparison against window-3 RLP.

## Reacher — complete window-3 table (all planners on 3-frame-window costs)

Same protocol as above (held-out, first-hit rh5, draws 42–47; latent-w3 =
3-frame window L2 via make_l2window_n; value-w3 = window quasimetric e0.05).
Jobs 6568/6569 (latent-w3), 6376/6378 (value-w3), 6088–6092 (RLP).

| planner (all window-3) | rollouts | LeWM τ=.1 / τ=.05 | PLDM τ=.1 / τ=.05 |
|---|---|---|---|
| latent-w3 CEM | 9,000 | 99.0 / 94.3 | 98.3 / 89.3 |
| latent-w3 MPPI | 9,000 | 87.7 / 68.0 | 85.7 / 64.3 |
| latent-w3 Adam | 3,000 | 97.3 / 80.0 | 96.7 / 77.3 |
| value-w3 CEM | 9,000 | 99.3 / 89.3 | 98.3 / 84.7 |
| value-w3 MPPI | 9,000 | 86.0 / 66.0 | 83.7 / 61.7 |
| value-w3 Adam | 3,000 | 98.3 / 81.0 | 97.3 / 76.7 |
| **RLP** | **9** | **99.9 / 97.1** | **99.4 / 91.2** |

Reading: at matched window-3 information RLP still tops every cell, but the
margins over the best hand-designed planner shrink to +2.8 (LeWM τ=.05, vs
latent-w3 CEM 94.3) and +1.9 (PLDM, vs 89.3) — parity-to-narrow-win in
success rate, at 1,000× fewer world-model queries. On Reacher the window-3
*latent* beats the window-3 *value* under CEM (94.3 vs 89.3), consistent
with the paper's claim that Reacher's geometry makes the latent an adequate
surrogate; the learned-value advantage on this domain is therefore RLP's
gradient-based refinement, not the objective. The earlier single-cell
control (98.7/94.3) reproduces within noise.

## Reacher LeWM window-1 diagnostic (job 6770, 2026-08-14/15)

Protocol-matched to the PLDM w1 diagnostic (mw 0.3, lr 1e-4, replay 0.5,
6k steps, batch 128, single-frame value e0.05 for init and critic; 3 seeds ×
draws 42–44, first-hit rh5, clean split). Arms: expand {0,1} × amax
{1.4, 2.2 (= the base's w3-recipe value)}.

| LeWM w1-RLP arm | τ=0.1 | τ=0.05 |
|---|---|---|
| expand 0, amax 1.4 | 96.9 | 81.8 |
| expand 0, amax 2.2 | 98.7 | 88.7 |
| expand 1, amax 1.4 | 29.1 | 14.7 |
| expand 1, amax 2.2 | 41.6 | 18.2 |

Revises the "w1 ceiling" reading: on LeWM, w1-RLP at the recipe amax (88.7
@τ=0.05) clears the best w1 baselines (value+CEM 82.0, latent+CEM 80.3) by
~+7–8 — the learned refiner contributes even at window-1 — with the windowed
critic adding a further +8.4 to reach 97.1. On PLDM w1-RLP (82.0) only
matches the w1 baselines. The expand×w1 collapse is much more severe on LeWM
(29–42/15–18) than PLDM (88–90/63–64): with a single-frame critic, value
expansion destroys the actor rather than merely capping it. July-era anchor
(82.9, older recipe + padded 1-frame conditioning) is superseded by this
protocol-matched grid.
