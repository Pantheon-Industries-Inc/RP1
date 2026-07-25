# TRM on TwoRoom — H100 results (DINO-WM)

World model: **DINO-WM** (frozen DINOv2-small, mean-pooled patches → 384-d latent, +
learned latent predictor) trained on TwoRoom pixels (800 eps, mixed expert+random).
Task: TwoRoom, **hard cross-wall** goals (start/goal on opposite sides of the wall),
goal_offset 20. Planner: CEM (paper's planner) and Predictive Sampling (MPC shooting),
horizon 20, receding 20, 200 samples, 8 CEM steps. n=40 goals. Terminal cost only differs
across conditions (encoder/dynamics/sampler frozen) — the TRM setup.

## Paper reference (LeWM, TwoRoom hard n100)
- raw LeWM latent MSE ≈ **7%**, horizon-matched **TRM (regression)** ≈ **97%**, oracle upper bound.

## Unified headline — ALL conditions, one shared eval set (TwoRoom hard cross-wall, DINO-WM, CEM, n=40)
| terminal cost            | success% |
|--------------------------|----------|
| latent (raw DINOv2)      | 57.5     |
| regression (paper recipe)| 50.0     |
| contrastive              | 52.5     |
| hybrid                   | 75.0     |
| **offline TD**           | **95.0** |
| **online TD**            | **97.5** |
| shuffled (control)       | 15.0     |
| oracle                   | 100.0    |

All conditions evaluated on the **same** 40 cross-wall goals via `eval_trm` (online-TD value trained
by `online_td.py --init-metric dino_td --save-value`, then scored as a normal condition). Clean
ordering: **online TD (97.5) > offline TD (95) ≫ latent (57.5) > shuffled (15)**, oracle 100.
The online_td run's own curve (same WM, its eval set): offline 92.5 → online 97.5, collection→100%.

## Phase 2+3 — all metric learners (n=40 cross-wall)
| condition (terminal cost)        | CEM   | Pred.Samp. |
|----------------------------------|-------|------------|
| latent (raw DINOv2 MSE)          | 57.5  | 50.0       |
| trm_regression (paper's method)  | 50.0  | 25.0       |
| **trm_td (offline TD)**          | **95.0** | 65.0    |
| trm_contrastive                  | 52.5  | 40.0       |
| hybrid (std(latent)+std(metric)) | 75.0  | 30.0       |
| shuffled (negative control)      | 15.0  | 27.5       |
| oracle (task-state geodesic)     | 100.0 | 90.0       |

**Takeaways**
- **Offline TD is the strong variant: 95% vs 57.5% raw latent (CEM), near the 100% oracle —
  in the same ballpark as the paper's 97% TRM.**
- The latent baseline is 57.5% (not the paper's 7%) because DINOv2's pooled latent is a much
  less-misleading navigation latent than LeWM's SIGReg latent; the *relative* TRM gain is still large.
- The paper's exact **regression** recipe ≈ latent here (the metric-learner choice matters a lot
  in this latent); hybrid helps (+17.5 over latent).
- **Shuffled-label control collapses (15%)** — temporal supervision is what carries the signal.
- **CEM ≫ Predictive Sampling** for the metric conditions (TD 95% vs 65%): the planner exploits
  a good terminal metric far better with iterative refinement than single-shot shooting.

## Regression scale ablation (CEM, n=40)
| scale s | success% |
|---------|----------|
| 20  | 22.5 |
| 60  | 50.0 (Phase 2) |
| 100 | 55.0 |
| 200 | 65.0 |
| (TD ref) | 95.0 |
Regression is scale-sensitive (larger s better, since DINOv2 temporal distances span a wide
range) but even best (65%) stays far below TD (95%).

## Phase 4 — Online vs Offline TD (DINO-WM, same eval set, matched 8000-step offline + 1500/round)
| round   | buffer | collect% (online policy) | eval% |
|---------|--------|--------------------------|-------|
| offline | 62401  | –                        | 70.0  |
| 0       | 63152  | 82.5                     | 70.0  |
| 1       | 63911  | 75.0                     | 82.5  |
| 2       | 64601  | 90.0                     | 80.0  |
| 3       | 65265  | 90.0                     | 90.0  |
| 4       | 65879  | 92.5                     | 87.5  |
| 5       | 66580  | 82.5                     | **92.5** |

**Takeaways**
- **Online TD clearly beats offline TD on the same eval set: 70.0% → 92.5% (+22.5).** Each round
  collects MPC-visited trajectories with the current value, appends to the buffer (hindsight goals),
  and TD-updates — so the metric is trained on exactly the states the planner induces.
- The online policy's collection success climbs to **82–92%**: the value learned online guides the
  planner reliably toward goals.
- (An earlier matched-at-4000-steps run gave 72.5% → ~80–85%; raising the offline budget to 8000 and
  matching TD hypers to the sweep sharpens the offline baseline and the online gain.)

## Files
- `phase2_cem.txt`, `phase2_predictive_sampling.txt` — full sweep tables.
- `regression_scale.txt` — scale ablation.
- `online_td_dino.txt` — online-vs-offline TD curve.
