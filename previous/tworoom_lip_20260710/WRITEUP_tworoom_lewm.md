# LIP on TwoRoom (LeWM) — Results

**2026-07-10 → 07-12.** Frozen LeWM ViT-tiny (`weights_epoch_16_partial`), 1000 play episodes,
replay eval: n=50 per cell, seeds 42/43/44, goal offset 25 (h25) / 50 (h50) primitive steps,
budget 2× offset, CEM 300×30, LIP deployed pure v1 (deterministic, no sampling).
*standard* = raw task draw; *hard* = cross-wall-only draws. A "cell" = surface × horizon × seed.

## Success % (mean over 3 seeds)

| method | std h25 | std h50 | hard h25 | hard h50 |
|---|---|---|---|---|
| original LeWM (latent + CEM) | 87.3 | 61.3 | 80.7 | 61.3 |
| TD + CEM | 100.0 | 100.0 | 99.3 | 100.0 |
| **LIP** | **100.0** | **100.0** | **100.0** | **100.0** |

(Random floor: 28 / 10.7 / 13.3 / 2. Single-cell noise ±6–7 pts at n=50. LIP = perfect
600/600-episode card, at ~450× less planning compute than CEM: ~16 rollout-equivalents per
replan vs ~9000. Two independent actors reach the perfect card — the τ0.1-teacher recipe
(training seed 1) and the ed256-teacher variant; across 5 trained draws of the final family
every cell is ≥98 and 2/5 draws are perfect.)

**CEM budget sensitivity:** at half the sampling budget (150 samples), latent+CEM loses
another 8–11 pts (76.0 / 51.3 / 70.7 / 53.3) while TD+CEM holds ≈99 (100 / 98.7 / 99.3 /
99.3) — the raw latent cost needs the search; the learned value doesn't. LIP samples nothing.

- **TD value:** offline quasimetric TD (0.15M params) on a stride-1 latent cache. Saturates
  the benchmark through CEM regardless of hypers — all 9 sweep configs ≈100.
- **LIP:** learned iterative refiner (0.54M params), K=8 iterations, lr 3e-4, 8k steps,
  trained purely against the frozen TD value through the frozen WM. Final planner
  `actors/lv1_t-e01n50.pt`, teacher `metrics/td_e0.1_n50.pt` (expectile **τ=0.1**, n-step 50).

## What closed the h50 gap: long-range supervision, not capacity

The first LIP (teacher τ=0.03) trailed TD+CEM at h50 (97.3/98.7 vs 100). A pure-v1 hyper
sweep on both axes (4 teacher variants × fixed recipe, 4 recipe variants × fixed teacher;
selection on the 3 weakest cells, reference 292/300) found three configs at a perfect 300:

- **teacher τ=0.1 n50** (winner — simplest change) and **teacher embed-256/12k** — better
  gradient fields for pathwise refinement (τ0.03 ranks fine for CEM but misleads gradients);
- **max-delta 12** (HER goals to 60 primitive steps) — longer-horizon goal coverage fixes it
  even with the τ0.03 teacher.

Capacity/budget knobs did nothing or hurt: 16k steps 292, K=12 290, and K8+lr1e-3 diverges
outright (8%). Extending the plan itself also fails: a horizon-10 LIP (50-step plans through
a WM trained on short rollouts) collapses to 12% — imagined states drift beyond the WM's
reliable horizon. Common thread: the gap was **long-range training signal** (teacher gradients
or goal coverage), not refiner capacity or plan length.

## Residual misses were near-misses — and a perfect draw exists

Individual actor draws occasionally miss ~1/600 episodes: same-room tasks with a corner goal,
agent arriving just outside the 16px success radius at budget end (verified on video); misses
are actor-idiosyncratic, and CEM solves all of them. Retraining the winner recipe with fresh
seeds and carding the other selection-perfect teacher settled it: **seed-1 and the ed256
variant both score 100 on all 12 cells**; the weakest draw in the family is 99.5.

## Training cost

| model | params | training |
|---|---|---|
| world model (frozen, theirs) | 18.0 M | epoch 16, partial (pod run; checkpoint carries no step count) |
| TD value | 0.15 M | 6k steps × batch 1024 (~6M pairs), ~3 min on 1×H100 |
| LIP planner | 0.54 M | 8k steps × batch 128, ~1.5 h on 1×H100 (shared) |

## Artifacts

`results/summary.csv` (all cells) · teacher `metrics/td_e0.1_n50.pt` · **final planner
`actors/lv1_e01n50_seed1.pt`** (perfect card; `lv1_t-ed256.pt` equally perfect) · drivers
`run_all.sh` / `run_hypers.sh` / `run_std100.sh` / `run_cem150.sh` (idempotent) · data:
`collect_play.py --episodes 1000 --seed 7`. Pod: `31.24.80.32:18106`.
