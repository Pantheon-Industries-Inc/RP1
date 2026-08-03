# DINO-WM: TD/LIP campaign results — 2026-08-01/02

Pod `157.66.255.84:11890` (4×H200). **Pod became unreachable 2026-08-03 ~00:30
UTC (connection refused) with work in flight** — see §7 for exact state.
Artifacts live on `/workspace` (MFS network volume, survived pod swaps before);
the code fixes are reproducible from the patcher scripts listed in §6.

---

## 1. Headline numbers (held-out episodes 8000–9999, 196 px, fp32)

| planner | result |
|---|---|
| **plain latent+CEM** | **80.0** (82 / 80 / 78) |
| TD+CEM, best of 11 teachers | 76.0 |
| LIPv4, pre-action-fix (INVALID, see §3) | 72.0 |
| LIPv4, post-action-fix | 69.3 (70 / 70 / 68) |
| LIPv4 on the improved critic | **in flight at outage** |

Plain latent MSE remains the best planner on this world model. Every
learned-value planner is below it.

## 2. Critic quality — the two levers that worked

Pooled 384-d latents, 2×3 factorial, probed on held-out episodes:

| samples | raw | standardized |
|---|---|---|
| 24.6M | 0.4743 / 0.6870 / 0.6236 | 0.5111 / 0.7018 / 0.6162 |
| 49.2M | 0.5009 / 0.6967 / 0.6262 | 0.5352 / 0.7132 / 0.6367 |
| **98.3M** | 0.5262 / 0.7080 / 0.6458 | **0.5532 / 0.7252 / 0.6534** |
| *LeWM teacher (reference)* | | *0.484 / 0.691 / 0.923* |

(columns = spearman / pair_acc / monotone)

* **Per-dim standardization: +0.03 spearman, +0.016 pair_acc at EVERY volume.**
  Structural justification: LeWM (v2WM) and both PLDM checkpoints end their
  encoder in `MLP(192→2048→192, norm_fn=BatchNorm1d)`, so their critic consumes a
  normalised isotropic space by construction. DINO latents are raw ViT
  activations — no normalisation anywhere in the path.
* **Sample volume: +0.05 spearman from 24.6M→98.3M, NOT saturated.** The stock
  recipe (6000 × 1024 = 6.1M) is ~16× undertrained on this substrate.
* Folding: train on transformed latents, then fold the affine map into the
  head's single entry `Linear` (`W' = W/σ`, `b' = b − W·μ/σ`). The saved metric
  then accepts RAW latents, so the CEM hook / LIP / probes work unchanged.
  Verified numerically per cell (max|diff| ~1e-5).
* **The best DINO critic now BEATS the LeWM teacher** on both unconfounded
  measures (+14% spearman, +5% pair_acc) — yet planning did not improve. The
  critic was not the binding constraint either.

## 3. Bugs found (all mine, all in the DINO LIP path)

**Action convention — invalidates every pre-fix LIP number.** Measured one-step
to five-step rollout error against the true next latent:

| h | raw actions | normalized | zero actions |
|---|---|---|---|
| 1 | 0.1117 | **0.0604** | 0.1718 |
| 5 | 0.1859 | **0.0678** | 0.2720 |

The predictor was trained on **z-scored** action blocks. `rollout_terminal_dino`
/ `rollout_traj_dino` took RAW actions (the original `train_lip_dino.py`
docstring asserts this and my ports inherited it). At raw scale the WM is barely
better than being fed ZERO actions — the action pathway is nearly inert, which is
why the LIPv4 surface was flat at 70–72 across every amax/lr/step setting.
With normalization, error is 2.7× lower at h=5 and nearly flat over the horizon
(+13% vs +66%).

**Double-scaled returns.** `_proposal_lip_dino` returned `A*ast+amu` (raw), but
`policy.py:303` applies `process['action'].inverse_transform` to the solver's
output. LeWM's known-good `_proposal_lip` returns `A` un-scaled. Fixed: the dino
path is now convention-identical to LeWM's.

**GPU-0 contention.** Running the critic sweep on GPU 0 while the EGL eval daemon
used it core-dumped three evals (`timeout: the monitored command dumped core`).
Known failure mode; GPU 0 must be eval-only.

## 4. Hypotheses I raised and then falsified by measurement

Recorded because each cost real time and the refutations are the durable result:

1. *"Pooling/reduction is second-order."* WRONG — full tokens beat mean-pooling
   (monotone 0.575 → 0.702).
2. *"Frozen DINOv2 geometry can't support a cost-to-go."* Not supported — no
   train/held-out gap (0.7993 vs 0.8014 at 25-step span), so it underfits rather
   than fails to generalise.
3. *"Head capacity is the bottleneck"* (75,264→256 is a 294:1 squeeze). WRONG —
   7-cell null: width 256→2048, embed 128→512, depth 2→3 all within ±0.01.
4. *"The held-out split explains the 80-vs-86 gap to the published baseline."*
   WRONG — all-10k draws scored 78.0, *below* held-out's 82.0.
5. *"The TD bootstrap signal is the bottleneck."* WRONG — fully supervised
   regression on true temporal distance was no better (monotone 0.616 vs 0.624)
   and worse on ranking.
6. *"Low effective rank means the extra dims are noise."* WRONG — PCA-whitening
   to k=24 (79.9% of variance) scored 0.399 vs raw's 0.474 at equal samples. The
   low-variance directions carry task-relevant signal; effective rank measures
   variance concentration, not information. **This is why the large sample count
   is justified rather than wasteful.**
7. *"The rollout carries no signal by h=5."* WRONG — that was my own action-scale
   bug compounding, not a substrate property.

**Also: `monotone` is a confounded metric.** TD on *ground-truth cube xyz* scores
only 0.4907 on it, because the cube is static during the reach phase and a flat
distance counts as non-decreasing. Prefer spearman / pair_acc.

## 5. Mechanism that survives

With the action bug fixed the actor drives its predicted cost **5.6× down
(27.7 → 4.97)** while success stays flat at ~69. So dynamics fidelity is not the
constraint, and neither is critic ranking quality (ours now exceeds LeWM's).
What remains is that the learned value's minima do not correspond to task
success, while token-grid MSE — anchored at the goal embedding and un-hackable by
construction — does. CEM replans every step, so it only needs good local signal,
which latent MSE supplies.

## 6. Reproducible code fixes (patchers in the session scratchpad)

* `make_lip4_dino.py` — adds `rollout_traj_dino` + `train_lip_ac_dino.py`
  (the real v4 recipe: PlannerNetV4, co-trained critic, canonical schedules) and
  whitelists `kind="lip4_dino"` in the solver.
* `add_replay_dino.py` — ports `--replay-prob` to token space (buffer stores the
  last 3 imagined TOKEN frames with the action slice zeroed, matching what a
  deployed replan queries).
* `fix_action_convention.py` — the §3 fix at all three call sites.
* `critic_sweep.py` / `critic_whiten.py` — §2 and falsification 6, with folding.
* `exp_action_convention.py`, `exp_conditioning.py`, `exp_offmanifold.py`,
  `exp_ceiling.py`, `probe_fit_vs_gen.py` — the diagnostics.

## 7. Exact state at the outage (2026-08-03 ~00:30 UTC)

* **Running:** 3 LIP cells on the `std96k` critic (`fx96_ctrl`,
  `fx96_exp0.5`, `fx96_rep0.25`), ~230/3000 steps — almost certainly lost.
* **Running:** whitening k-sweep, `k=16/32/64` cells pending.
* **Paused:** eval daemon (paused deliberately to keep GPU 0 EGL-only).
* **Unscored:** `fx_a12_s3000`, `fx_a20_s3000`, `fx_ctrl_s3000` (their earlier
  FAILs were the §3 contention crash; the blobs verified loadable).
* On `/workspace`: caches (`dinopool_*`, `dinofull_*`), critics
  (`dinopool_td_{std,raw}*`, `dinopool_td_wh*`), actors (`fx_*`, archived
  pre-fix ones under `actors/invalid_pre_actionfix/`), and result CSVs
  (`summary_critic_sweep.csv`, `summary_critic_whiten.csv`,
  `summary_lip4d_fx.csv`).

## 8. What I would do next

1. **Re-run LIP on `std96k`** (interrupted). It is the only untested rung with
   every upstream fault corrected.
2. **TD+CEM with `std96k`** — 3 draws, ~2 h. Directly tests whether a
   LeWM-beating critic moves CEM at all.
3. Finish the whitening k-sweep only to bound how much compression is affordable
   (k=64 retains 91.2% of variance); it is not a candidate fix.
4. Do **not** pursue: expectile, lr, tau, n_step, p_cross, head width/depth
   (all swept flat), `head=mlp` (inverts), or expand/replay before the critic
   question is settled — both assume a teacher that is reliable on imagined
   latents.
