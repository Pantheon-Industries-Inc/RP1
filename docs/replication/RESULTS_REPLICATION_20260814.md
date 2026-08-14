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
