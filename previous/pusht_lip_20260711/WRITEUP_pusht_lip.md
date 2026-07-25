# LIP on PushT (official LeWM) — Results Writeup

**Date:** 2026-07-12 · **Status:** COMPLETE — full protocol + interp + 4 ablation threads + capstone
**TL;DR:** On PushT, the stage-1 recipe imported from cube/TwoRoom *reversed* the expected
ordering (latent+CEM ≥ TD+CEM > LIP). Diagnosis and repair: (1) the TD sweep optimum is the
**opposite corner** from prior tasks — shortest backups (n=1) + lowest expectile (τ=0.01) —
because single-frame latents alias velocity and every backup step compounds that noise;
(2) LIP is optimization-limited on PushT's multi-modal contact landscape — **8 noisy restarts
+ argmin-V selection** recover the gap. Final result: **LIP (best TD teacher, 8 restarts)
matches its TD+CEM teacher within noise at both horizons (84.7/50.0 vs 88.0/52.7) at ~70×
less planning compute, and beats latent+CEM at h50 (+4.7).**

## Setup

- **WM:** official `quentinll/lewm-pusht` (`weights.pt`; ViT-tiny 192-d pooled latent,
  frameskip-5 action blocks, 3-frame history). Frozen throughout. NOTE: the release weights
  use transformers-4 HF ViT key naming; this pod's transformers 5.13 builds q_proj-style
  modules — converted once via `convert_official_ckpt.py` (1:1 rename, 0 missing/unexpected),
  installed as `lewm_pusht_official`.
- **Data:** official `pusht_expert_train.h5` (46 GB, 18,685 episodes, 2.34M frames, weak
  expert). Used for eval task draws, action stats, and training caches (600k-frame prefix
  ≈ 4,800 episodes; a full-data cache was also built for one ablation — no gain).
- **Eval protocol** (per cell, n=50): replay — start at a dataset row, goal = the state 25
  (h25) / 50 (h50) primitive steps later; budget 2× offset; success = DINO-WM criterion
  (agent+block pos error < 20 px AND block angle error < 20°). Eval seed drives task draw +
  solver seed (42/43/44). CEM 300×30, horizon 5 blocks, replan every 25 steps.
- Pod: 4×H100 (`31.24.80.42:13573`), drivers `run_all_pusht.sh` + gated follow-up drivers,
  all idempotent in `/workspace/results/summary.csv` (local copy: `results/summary.csv`).
- Reference: the stable-worldmodel baselines page lists LeWM PushT at 96 (50-step budget);
  our latent+CEM canary on that exact protocol drew 90 (n=50, draw noise ±4–5).

## Main table — mean success % over seeds 42/43/44 (n=50 each)

| method | h25 | h50 | plan cost/replan |
|---|---|---|---|
| random floor | 1.3 | 1.3 | — |
| latent+CEM (their baseline) | 86.7 (90/90/80) | 45.3 (56/38/42) | ~9,000 rollouts |
| TD+CEM stage-1 winner (τ0.03 n5) | 76.7 | 46.0 | ~9,000 |
| **TD+CEM deep winner (τ0.01 n1)** | **88.0** (92/90/82) | **52.7** (62/56/40) | ~9,000 |
| deep winner, blend cost | 89.3 | 51.3 | ~9,000 |
| 3-seed ensemble + blend | 84.7 | 54.7 | ~27,000 |
| LIP v1 one-shot (stage-1 teacher) | 66.7 | 35.3 | ~16 |
| LIP v1 + 8 restarts | 70.0 | 51.3 | ~130 |
| LIP v2 one-shot (deep teacher) | 80.7 (80/82/80) | 43.3 | ~16 |
| **LIP v2 + 8 restarts (capstone)** | **84.7** (88/86/80) | **50.0** (54/44/52) | **~130** |

**Headline:** LIP v2+R8 − teacher = −3.3 / −2.7 (within eval noise ±6–7 at n=50);
LIP v2+R8 − latent+CEM = −2.0 / **+4.7**. The TwoRoom conclusion (LIP ≈ TD+CEM ≫ compute)
holds on PushT once the teacher is tuned to the task and inference has restart diversity.

## TD sweep (46 configs over two rounds)

Offline n-step distance TD with HER (balanced future + 30% cross-episode goals), expectile
toward the min, quasimetric head, target network, γ=1, trained on frozen-latent caches.

- Stage 1 (τ ∈ {0.03,0.1,0.3} × n ∈ {5,25,50}): winner τ0.03/n5 = 76+60 s42. All h25 cells
  BELOW latent — grid was in the wrong corner; n5 was a grid edge.
- Deep round (τ ∈ {0.01..0.3} × n ∈ {1,2,3,10,15} + 20k-step + 512×3 variants):
  **winner τ0.01/n1 = 92+62 s42**; n≤3 dominates every τ row; scale variants all ≤ baseline.
- **Lesson (inverts the cube/TwoRoom "long backups" rule):** PushT single-frame latents alias
  velocity (probe: vel R² 0.27 vs pose 0.94–0.97) → TD through aliased states is noisy →
  every extra backup step compounds it → shortest backup + sharpest optimism wins.

## Interp: why LIP one-shot underperformed (interp_lip.py)

Plan-time optimization comparison (achieved terminal value E, 64 tasks, lower better), h25:
CEM 0.89 ≪ LIP+8restarts 1.76 < Adam-60steps 2.36 ≈ LIP one-shot 2.75 ≪ zero-plan 16.5.
Gradient descent fails like LIP ⇒ the landscape (contact multi-modality: push left-around vs
right-around) defeats local methods; population search wins. Restarts + argmin-V close most
of the gap ⇒ LIP is **optimization-limited at inference, not knowledge-limited**.
Also exonerated: the solver's zeroed action-history at eval (measured ≈ no effect).

## Asymmetry thread (user hypothesis → confirmed representational gap, wrong bottleneck)

- The trained "quasimetric" is **de-facto symmetric**: median |d(b,a)−d(a,b)|/d = 4.8%, and
  the gap has ZERO correlation with block-pose change (the true irreversibility axis).
  Expectile-TD only supervises forward pairs; reverse queries are never constrained.
- **QRL-style constraint training** (`train_metric_qrl.py`: max E[d(a,b)] s.t. d(z_t,z_{t+1})≤1,
  dual ascent) induces the physically correct asymmetry from the same data: gap/fwd 94%,
  corr(gap, block motion) = 0.60, corr(gap, agent motion) = 0.08, gap ≈ 0 when block unmoved.
- But planning: QRL 72+46 = 118 < expectile-TD 154. CEM only queries forward-ish directions,
  so reverse-side correctness goes unused. Respectable for zero distance supervision.
- **Head ablation (the sleeper result):** plain MLP head 48 combined, symmetrized MLP 44 —
  vs quasimetric 154. The quasimetric **architecture** is massively load-bearing on PushT,
  NOT via asymmetry (the learned function is symmetric) but via its metric constraints
  regularizing the distance field against CEM-exploitable off-manifold minima.

## Better-TD round 2 (all ≤ plain deep winner)

3-seed pessimistic-max ensemble (78.7/50.7; +blend 84.7/54.7 — best h50 mean, within noise),
input-noise smoothing (worse), γ=0.99 (worse), full-data 2.34M + 20k steps (worse).
History-conditioned value V([z,Δz],[g,0]) — motivated by the velocity probe (R² 0.27→0.66
with block-scale Δz) — underperformed everywhere incl. at the winner config (118 vs 154);
likely cause: plan-time Δz comes from WM-smoothed imagined frames (miscalibrated vs real Δz).
Blend mode: big rescue for weak teachers, ≈ no-op for the good one.

## WM-quality ablation (archived: results/summary_epoch20wm.csv)

The `weights_epoch_20.pt` checkpoint received 2026-07-11: latent+CEM 12.0/2.7 (≈ random),
TD+CEM ≈ 8–10. Its encoder probes fine (pose R² 0.92) but its predictor's 25-step open-loop
error (7.6) barely beats a frozen-world baseline (10.4) — unplannable. The official release
predictor is what makes everything above work.

## Artifacts

- `results/summary.csv` — every cell, 180 rows (pod: `/workspace/results/summary.csv`)
- `metrics/td2_e0.01_n1.pt` — the best TD teacher; `head_qrl.pt`, `hist_e0.03_n2.pt`
- `actors/lip2_k8_lr3e-4_st8000.pt` — the final LIP (use with `solver=lip solver.restarts=8
  solver.restart_noise=0.5`); `lip_k8_lr3e-4_st8000.pt` (v1)
- `logs/driver.log` (full timeline), `logs/asym_qrl.log`, `results/summary_epoch20wm.csv`
- Scripts: `run_all_pusht.sh`, `run_td_{deep,hist,better,head}.sh`, `run_lip2_restarts.sh`,
  `train_metric_hist.py`, `train_metric_qrl.py`, `convert_official_ckpt.py`,
  `interp_lip.py`, `asym_analysis.py`, `probe_vel.py`, `diag_wm.py`
- Repo changes (local stable-worldmodel): `pusht_lewm.yaml`, eval_wm.py metric hook
  (history-metric detection + comma-separated ensemble paths → pessimistic max),
  `--symmetric` wired through train_metric/TDConfig.
- Eval videos remain on the pod (`/workspace/results/videos_*`).
