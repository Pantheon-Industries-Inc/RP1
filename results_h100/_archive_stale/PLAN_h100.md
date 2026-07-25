# TRM on H100 NVL — Staged Plan & Schedule

**Box:** 2× H100 NVL (96 GB), 144 cores, torch 2.4.1+cu124, CUDA verified (kernels run).
`ssh root@216.81.245.58 -p 15081`. Repo target: `/root/stable-worldmodel`.

**World model:** DINO-WM (PreJEPA = frozen DINOv2 encoder + trained latent predictor) on
**TwoRoom pixels** — the paper's task, rendered fast via pygame (no MuJoCo/EGL needed). DINOv2
gives a genuinely *non-position* latent where Euclidean distance is misleading → the regime where
TRM shows large gains (unlike the position-like state-WM). GPU accelerates DINOv2 encoding,
predictor training, and CEM/PS rollouts.

**What's already built (Mac, transferring):** trm package (PairwiseMetricHead, MetricCost,
samplers, regression/TD/contrastive learners, oracle, SCSA diagnostics, io), StateWM, pipeline
scripts (cache_latents, train_metric, eval_trm with `--solver cem|predictive_sampling`, run_sweep),
11 passing unit tests, TwoRoom state-WM results.

**New work needed:** (a) DINO-WM↔MetricCost adapter (PreJEPA uses patch embeddings → mean-pool to a
vector); (b) DINO-WM featurizer for caching; (c) online-TD loop; (d) results storage.

---

## Phases / schedule

**Phase 0 — Setup (GPU).** Transfer code; install deps (torch already there; + stable-pretraining,
pygame, pymunk, shapely, einops, lancedb, sklearn, tabulate, pytest). Verify: CUDA kernel, swm
import, `pytest tests/test_trm.py`, TwoRoom pygame render. Set `$STABLEWM_HOME`.

**Phase 1 — DINO-WM substrate.**
1. Collect TwoRoom pixel dataset (mixed expert+random for wall-aware coverage), with `pixels`.
2. Train DINO-WM (PreJEPA) on TwoRoom pixels: frozen DINOv2-small + CausalPredictor (~10 epochs, GPU).
3. Build & unit-test the PreJEPA↔MetricCost adapter (pool predicted/goal pixel patch embeddings) +
   DINO-WM featurizer.
4. Cache pooled DINOv2 latents from logged trajectories. Sanity: temporal-distance corr of a quick
   regression head; confirm latent is non-position (Euclidean ≠ geodesic).

**Phase 2 — REIMPLEMENT THE PAPER (store all results).** Horizon-matched regression TRM on DINO-WM
latents. CEM-MPC (paper's planner) + Predictive Sampling. Conditions: `latent` (raw DINO patch-MSE
baseline), `trm_regression` (replacement), `hybrid`, `shuffled` (negative control), `oracle`
(task-state). Hard cross-wall manifest. Store success rates + SCSA (Spearman, best-rank-pct, regret)
per condition × solver to `results/`.

**Phase 3 — OFFLINE TD.** Train offline goal-conditioned temporal-distance TD metric on the same
DINO-WM latents. Add to the eval sweep; store. Compare vs regression & latent.

**Phase 4 — ONLINE TD.** `online_td.py`: interleave (a) env interaction (MPC with current metric +
hindsight goal relabeling) → replay buffer, with (b) TD updates — so the metric trains on the
states the planner actually visits. Compare online-TD vs offline-TD in MPC success + SCSA; store.

**Phase 5 — Consolidate.** Pull results back to Mac; write a results table + short comparison
(latent vs regression vs offline-TD vs online-TD vs oracle; CEM vs PS). Save key checkpoints.

---

## Result storage
On box: `/root/stable-worldmodel/results/<phase>/...` (JSON + txt tables + configs). Checkpoints in
`$STABLEWM_HOME/checkpoints`. Pull summaries back to `Value_Metric_LeWM/results_h100/`.

## Risks / fallbacks
- PreJEPA adapter brittle (patch + action-in-latent layout) → fall back to a CLS-pooled DinoWM
  (frozen DINOv2 CLS + small predictor) that drops straight into existing MetricCost/StateWM code.
- If GPU gives no speedup for the tiny metric heads, train those on CPU; keep DINOv2 encode + CEM on GPU.
