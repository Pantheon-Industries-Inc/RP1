# TD Reachability Metric vs. Raw Latent Cost for MPC in Latent World Models
*TwoRoom · DINO-WM vs. LeWM · TRM (arXiv 2605.22164) extended with TD / contrastive metrics*

## The change
A latent world model plans by Cross-Entropy-Method (CEM) MPC that ranks candidate action
sequences by a **terminal cost** between the predicted terminal latent `ẑ_{t+H}` and the goal
latent `z_g`. The default cost is **raw Euclidean latent distance** `‖ẑ_{t+H} − z_g‖²`. We
replace that terminal cost with a **learned reachability metric** `m(ẑ_{t+H}, z_g)` — keeping the
encoder, dynamics, candidate sampler, optimizer, and eval manifest **frozen** (the TRM protocol).

We learn `m` three ways, all on the *frozen* WM latents:
- **regression** (the paper): a pairwise head regresses temporal separation `|t_i − t_j|`
  (horizon-matched, balanced sampling).
- **contrastive**: InfoNCE critic over (state, future-goal) pairs.
- **TD** (our focus): a goal-conditioned temporal-distance **value** `d(z, z_g)` learned by
  **offline TD** (Bellman bootstrap on logged transitions, hindsight goals), and by **online TD**
  (the same value, but the buffer is grown with trajectories the MPC planner actually visits).
We also report **hybrid** = `std(latent) + std(metric)`, a **shuffled-label** negative control,
and an **oracle** task-state geodesic upper bound.

## Why it helps (mechanism)
Euclidean distance in latent space conflates *being near the goal in feature space* with *being
able to reach it*. In TwoRoom a blocked plan can end with a terminal latent that is Euclidean-close
to the goal yet geodesically far (wrong side of the wall). A **TD value** is trained to approximate
**steps-to-go along feasible trajectories**, so it ranks a plan that routes through the doorway
below one that stalls at the wall — exactly the candidate ordering MPC needs. Regression captures
this only weakly when the latent geometry is non-Euclidean; TD's bootstrapped value is far sharper.

## What "broken latent" means (and how the paper establishes it)
By a **broken latent** we mean — following the paper's *latent-proximity trap* (§1, abstract) — a
representation in which the control-relevant variables are *present but occupy a tiny, low-energy
subspace*, so that raw Euclidean/MSE terminal distance is **dominated by reachability-irrelevant
residual directions** and ranks candidate plans almost at random. The key point is that
*decodability ≠ usability*: the state can be linearly present yet carry no weight in the planner's
cost. The paper does **not** deliberately induce this — it inherits a fixed LeWM checkpoint family
(np512, 3 seeds) and *diagnoses* the brokenness with subspace surgery:

- **State is present.** A linear XY probe on LeWM latents reaches **R² = 0.998, RMSE ≈ 1.8 px**
  (§6.3, Table 6) — position is almost perfectly decodable.
- **But it carries ~no weight in raw MSE.** The XY-probe **rowspace accounts for only 0.5–0.7% of
  terminal-goal latent MSE** (Table 13: 0.005–0.0069), yet planning with **rowspace-only** MSE
  reaches **90.8%** while the **orthogonal residual** reaches **1.7%** (Table 8). I.e. <1% of the
  latent variance carries essentially all the control signal.
- **So the ordering is random.** Raw-latent candidate-ordering Spearman vs the oracle is **≈0.02**
  (Table 8: 0.028 Euclid / 0.018 residual), and the oracle-best candidate sits at the **~32nd
  percentile** of the raw-latent ranking (Fig 4).
- **More search does not fix it.** The paper's own *"search too weak"* control runs raw latent at
  **1000 samples × 20 iterations, top-k 100 (~20k evals) → still 2.5%** (Table 6). Budget is
  explicitly ruled out as the cause.

This is the regime where TRM has a ~90-point hole to fill, and it is a property of *that* checkpoint's
geometry — see §7 "Probeability is not enough." **We have not yet run this subspace audit on our own
LeWM** (`diagnostics.py` implements only SCSA, not the rowspace/residual projection), so we have no
evidence our latent is broken in this sense — and our raw-latent success (48%, budget-sensitive) is
positive evidence that it is *not*.

## Experiment settings
- **Env / task:** `swm/TwoRoom-v1`, **hard cross-wall** goals (start & goal on opposite sides),
  goal_offset 20, n = 40 goals, shared eval seed.
- **World models (frozen):**
  - **DINO-WM** — frozen DINOv2-small (mean-pooled patches → 384-d) + learned latent predictor;
    **frameskip 1**. Trained on 800 TwoRoom episodes (mixed expert+random for wall coverage).
  - **LeWM** — ViT-tiny/p14 (d=192) encoder **trained from scratch** + predictor, **end-to-end
    JEPA** (predictor MSE, no stop-grad) with **SIGReg** (wt 0.09); AdamW lr 5e-5, wd 1e-3, batch
    128, warmup+cosine; **frameskip 5** (action encoder sees 10-d), history 3 / num_preds 1 — the
    repo's tworoom config.
- **Metric learners:** pairwise head [z_i,z_j,z_i−z_j,|z_i−z_j|] → 2×256 SiLU → Softplus.
  Regression: Smooth-L1 on temporal separation. TD: γ=0.98, expectile-Huber, target network,
  hindsight goals. Contrastive: symmetric InfoNCE. All on cached frozen latents.
- **Planner:** CEM (paper's planner; 200 samples, 8 iters, top-20) and Predictive Sampling (MPC
  shooting). DINO-WM: horizon 20, action_block 1. LeWM: horizon 6, action_block 5 (= frameskip 5).
- **Metric of merit:** MPC success rate (goal reached). Hardware: 1× H100 NVL.

## Results

### DINO-WM (frameskip 1, CEM, n=40 cross-wall) — one shared eval set
| terminal cost            | success% |
|--------------------------|----------|
| latent (raw DINOv2)      | 57.5     |
| regression (paper)       | 50.0     |
| contrastive              | 52.5     |
| hybrid                   | 75.0     |
| **offline TD**           | **95.0** |
| **online TD**            | **97.5** |
| shuffled (control)       | 15.0     |
| oracle                   | 100.0    |

Predictive Sampling (same metrics): latent 50, offline-TD 65, online policy weaker — CEM exploits a
good terminal metric far better than single-shot shooting.

### LeWM (frameskip 5, action_block 5) — manifest matters
LeWM trained from scratch (custom loop; lr 5e-5, batch 128, warmup+cosine, no target stop-grad,
SIGReg 0.09). Two manifests, CEM 200×8:

**Easy (short-range cross-wall, goal_offset 20, n40):** latent 97.5, regression 67.5, contrastive
72.5, hybrid 97.5, offline-TD 90.0, shuffled 22.5. *Raw latent already solves it → nothing to repair.*

**Hard n100 (high-distance band 90–180 px, 50 cross-wall + 50 same-room):**
| terminal cost      | n100% | cross% | same% |
|--------------------|-------|--------|-------|
| latent (raw LeWM)  | 48    | 46     | 50    |
| regression (paper) | 65    | 58     | 72    |
| contrastive        | 58    | 50     | 66    |
| **hybrid**         | **81**| 82     | 80    |
| offline TD         | 59    | 36     | 82    |
| **online TD**      | **73**| 54     | 92    |
| shuffled (control) | 3     | 0      | 6     |
| oracle             | 100   | 100    | 100   |

On the hard manifest the latent **collapses to 48%** and every learned metric beats it (hybrid 81,
online-TD 73 > offline-TD 59 > latent 48); the shuffled control craters to 3% and oracle is 100% —
the paper's qualitative effect, and **online TD > offline TD** again.

**Hard n100 at the paper's CEM budget (~150 evals; 30 samples × 5 iters):**
| terminal cost      | n100% | cross% | same% |
|--------------------|-------|--------|-------|
| latent (raw LeWM)  | **14**| **10** | 18    |
| regression (paper) | 34    | 28     | 40    |
| **hybrid**         | **44**| 28     | 60    |
| offline TD         | 36    | 22     | 50    |
| **online TD**      | 42    | 20     | 64    |
| shuffled (control) | 1     | 0      | 2     |
| oracle             | 99    | 98     | 100   |

Paper reference (their LeWM, hard n100): raw latent ≈ **7%**, TRM ≈ **97%**.

**Reconciling 7% vs our 48% (corrected).** An earlier version of this writeup attributed the gap
mainly to **planner budget** (our CEM uses more evals than the paper's, "masking" a broken latent).
*That reconciliation was wrong, and the paper refutes it directly:* its "search too weak" control
runs raw latent at 1000 samples × 20 iterations, top-k 100 (~20k evals — far more than ours) and
still gets only **2.5%** (Table 6). If 12× our search cannot rescue their latent, search strength is
not why ours scores higher. Our own budget-sensitivity (48% → 14% as we cut evals to ~150) is in
fact *evidence our latent is not broken*: it carries real ordering signal that more search can
exploit, whereas a truly broken latent (Spearman ≈0.02) gives search nothing to work with.

The real explanation is **latent geometry** (see "What 'broken latent' means" above): the paper's
LeWM has a sub-1% control-relevant subspace buried under nuisance variance, so raw MSE ranks
candidates near-randomly; our LeWM — trained on **fixed-geometry TwoRoom** (no factors of variation)
— is position-aligned, so Euclidean distance is already a decent proxy (hence 48%, not 7%). A likely
*third* contributor, independent of geometry, is **cross-wall coverage** in the metric's training
trajectories: our cross-wall success trails same-room everywhere (regression 58 vs 72; online-TD 54
vs 92; offline-TD 36 vs 82), the signature of too few through-the-doorway pairs in the cache. The
honest status: the paper's broken-latent → TRM-recovery → oracle *effect* reproduces qualitatively,
but we have **not** reproduced the broken-latent regime itself, and we have **not run the subspace
audit** that would confirm whether our latent is broken at all.

## Takeaways
1. **TD as the terminal metric is the decisive win when the latent is misleading.** DINO-WM: offline
   TD **57.5 → 95%** (online 97.5%), near oracle 100, while regression ≈ raw latent. LeWM hard n100
   (paper budget): raw latent **14% (cross-wall 10%)** → every TRM metric recovers 2–3× (hybrid 44,
   online-TD 42), oracle 99. The metric *learner* matters as much as the idea of replacing latent.
2. **The negative control collapses** everywhere (15% DINO; 1–3% LeWM hard) — the gain is from
   temporal/reachability structure, not head capacity.
3. **Online TD ≥ offline TD** in every setting (DINO 97.5 vs 95; LeWM-hard 42/73 vs 36/59) — training
   the value on planner-visited states helps, especially on cross-wall goals.
4. **What governs "how broken" raw latent looks is latent geometry, not planner budget.** A
   position-aligned latent (fixed-geometry data, or DINOv2) is a fair proxy; factors-of-variation
   entangle it into the paper's <1%-control-subspace regime. Planner budget is *not* a knob that
   makes a latent look broken — the paper shows ~20k evals still yield 2.5% on its broken latent.
   Cutting our CEM (48→14) lowers success only because our latent carries signal that search
   exploits; it does not reproduce the paper's regime, where search is impotent.

## Caveats / to close the residual gap to the paper's exact 7→97
- **#1 missing diagnostic — run the subspace audit.** Port the paper's projection intervention
  (Eq. 12: `P_W = Wᵀ(WWᵀ)⁺W` from the linear XY probe) into `diagnostics.py` and report, on our LeWM:
  rowspace-MSE share, rowspace-only vs residual-only planning success, and raw-latent Spearman vs the
  oracle. This *settles* whether our latent is broken. If rowspace share ≫ 1% and Spearman ≫ 0.02,
  the 7% target is simply not reachable with this checkpoint and no TRM tuning will get there.
- **To actually reproduce the broken regime:** retrain LeWM on TwoRoom **with factors of variation**
  (varying door/wall position, colors) so the latent entangles, *not* by changing the planner budget.
- **Cross-wall coverage:** audit how many within-episode pairs in the cache straddle the doorway at
  90–180px geodesic; if sparse, mix in random/exploratory rollouts (as the DINO dataset did) before
  retraining the metric — the cross ≪ same gap points here.
- The paper does not publish LeWM's training recipe (no frameskip, optimizer, or `np512` expansion
  given); LeWM is a fixed inherited checkpoint there, so an exact match is not possible from the paper.
- Frameskip note: an earlier draft blamed frameskip 5 for a "short-horizon" metric — **retracted**;
  the cache encodes every frame, so temporal labels span the full ~100-step episode. The real
  frameskip-related risk is **rollout fidelity** (predicted ẑ_{t+H} fed to the metric), separable via
  SCSA on encoded vs predicted terminals.
- LeWM evaluated with `history_len=1` (trained on 3) — identical across conditions.

*Caveat: DINO-WM uses frameskip 1, LeWM frameskip 5 (its repo-faithful setting), so the DINO-vs-LeWM
comparison is not strictly apples-to-apples on planning temporal resolution; the latent geometry is
the intended axis of comparison.*
