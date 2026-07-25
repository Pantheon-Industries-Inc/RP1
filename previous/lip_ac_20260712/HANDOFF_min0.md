# HANDOFF — min0: the minimal-input LIP actor

**Status:** confirmed 2026-07-13. Self-contained; you should be able to pick this up cold.
**One-line:** the LIP actor needs neither the raw current state nor the raw goal — an actor
fed only `[plan, ∇_A V, V]` **ties** the full-input champion (87.8 vs 88.0, 3-draw h25) at
lower seed variance, strictly leaner input, and with the gate optional.

> Honest framing up front: this is a **design** result, not a score breakthrough. min0
> does NOT beat the champion's 88.0 — it matches it. The value is the leaner, more robust,
> better-understood model, not a higher number. >90 is still open (see §7).

---

## 1. What min0 is

The champion actor ("schedamax", see LIP_AC_MLP_CHAMPION.md) is a 2×512 MLP applied K=8
times as a learned refinement rule. min0 is **identical in every way** — same MLP, same
tandem actor–critic training, same schedamax recipe, same TD warm-start for the critic —
except its **input vector drops the two raw latents**:

    champion:  u = [ vec(A) | vec(∇_A V) | E | z0 | z_g ]      ∈ R^635
    min0:      u = [ vec(A) | vec(∇_A V) | E ]                 ∈ R^251

where, per refinement iteration k on the current plan A:

    A ∈ R^{5×25}        current plan (5 action-blocks × 25 dims)
    ∇_A V = ∇_A V(ẑ_H(A), z_g)     grad of terminal value w.r.t. the plan (detached)
    E     = V(ẑ_H(A), z_g)         scalar terminal value = imagined steps-to-go (detached)
    ẑ_H(A) = frozen-WM unroll of A from the encoded history

    net   = Linear(251→512) → ReLU → Linear(512→512) → ReLU → Linear(512→126)
    dA, gate = split(net(u))
    f_θ(A, ∇_A V, E) = clip_±3.5( A + σ(gate)·dA )        [gate optional — see §5]

    A(0) = 0;   A(k+1) = f_θ(…);   deploy A(8), receding horizon, no restarts.

**The point:** the goal z_g and current state z0 reach the actor ONLY through the teacher's
two signals — the value's gradient and its level. The actor never sees an absolute latent.
That makes f_θ a **pure learned optimizer** for the objective family {V(·, z_g)}: a map
(plan, ∇V, V) ↦ refined plan, invariant to the latent representation and agnostic to which
goal it is solving. "The value is the only teacher" becomes literal — it is also the only
*input* carrying task information.

## 2. The result (h25, full protocol, 50 tasks/draw)

3-draw confirm. Control: champion schedamax reproduced **96.0 exactly** on draw s43 (its
known value) → harness + checkpoints trustworthy.

| min0 gated, seed | s42 | s43 | s44 | 3-draw mean |
|---|---|---|---|---|
| 0 | 88 | 94 | 84 | 88.7 |
| 1 | 86 | 96 | 78 | 86.7 |
| 2 | 86 | 94 | 84 | 88.0 |
| 3 | 86 | 94 | 84 | 88.0 |
| **mean (4 seeds)** | 86.5 | 94.5 | 82.5 | **87.8** |

| reference | 3-draw mean | seeds |
|---|---|---|
| champion schedamax (full input) | **88.0** (88/96/80) | 1 confirmed |
| min0 gated | 87.8 | 4 |
| min0 no-gate | 87.3 (87.3 / 87.3) | 2 |

**Verdict: tie.** 87.8 vs 88.0 is 0.2 apart, far inside the noise (n=50/draw → binomial
sd ≈ 5/cell). min0's win comes on the *other* axes:
- **leaner:** 251-dim input vs 635 (no raw state, no raw goal)
- **lower variance:** 4 seeds within 2.0 points (3-draw means 86.7–88.7) vs the champion's
  14-point spread on the 2-draw selection metric
- **gate-optional:** pure residual A+dA loses nothing (87.3), unlike the transformer where
  the gate was worth +5
- **better-evidenced:** confirmed on 4 training seeds vs the champion's single seed

## 3. Why the earlier "+7" was wrong (read before trusting any 2-draw number)

The input sweep selected on s42+s44 only, where min0 led 169 vs 161 — apparently +7. That
was **task-sampling noise**. The excluded draw s43 is the champion's strong draw (96); min0
averages ~94.5 there, slightly behind. Averaged over all three draws the lead vanishes.
Lesson, now paid for twice: the 2-cell selection metric controls training-seed variance but
under-samples the *task* axis. **Confirm on 3 draws before believing any delta < ~8 points.**

## 4. Reproduce (pod 31.24.80.32:15419, all assets there)

    export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel
    export TQDM_DISABLE=1 MUJOCO_GL=osmesa
    WM=/workspace/ckpts/ogbench_cube_single_v2WM
    H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5

    # train (schedamax recipe + the only two flags that make it min0):
    python scripts/plan/train_lip_ac.py \
      --cache /workspace/caches/cube_full_fs5.pt --cache-td /workspace/caches/cube_full_fs1.pt \
      --h5 $H5 --wm $WM --init-value /workspace/metrics/cf_dE_t003n50.pt \
      --horizon 5 --iters 8 --steps 8000 --n-step 50 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 \
      --drop-z0 --drop-zg   [--no-gate]   --seed 0 \
      --out actor.pt --out-value value.pt

    # eval one draw (repeat seed=42/43/44 for the 3-draw mean):
    python scripts/plan/eval_wm.py --config-name cube seed=43 ++bf16=true eval.img_size=224 \
      policy=$WM solver=lip solver.actor_path=actor.pt \
      eval.goal_offset_steps=25 eval.eval_budget=50 eval.dataset_name=$H5

Trained checkpoints already on the pod:

    gated:    /workspace/actors/lip_ac90s5_min0_s0.pt, lip_ac90s5_min0_s1.pt,
              /workspace/actors/lip_ac90s6_w_s2.pt,     lip_ac90s6_w_s3.pt
    no-gate:  /workspace/actors/lip_ac90s6_wng_s0.pt,  lip_ac90s6_wng_s1.pt

Code: `--drop-z0 / --drop-zg / --no-gate` in `scripts/plan/train_lip_ac.py`;
`PlannerNet(use_z0, use_zg, use_gate)` in `stable_worldmodel/solver/lip.py`. All flags
default to the full-input champion, and every legacy checkpoint loads via `ck.get(...)`
defaults, so nothing here breaks the champion.

## 5. Gate

min0 works with or without the gate (87.8 gated / 87.3 no-gate — tied). On the MLP the gate
is `σ(gate)·dA`, a single scalar per plan; expressively redundant (absorbable into dA), its
only real job was step-size damping through the K=8 weight-tied unroll. Dropping it (pure
residual A+dA) costs nothing here — the opposite of the v3 transformer, where removing the
per-token gate cost ~5 points. Prefer no-gate for min0 on simplicity grounds.

## 6. Caveat that matters for tuning: E_final anti-correlates with success

min0 trains to E_final ≈ 3.8–4.0 (imagined steps-to-go on the training pairs); the champion
reaches 1.46. Yet they tie on real success. Removing z0/z_g removed the actor's ability to
*exploit value idiosyncrasies* — it can no longer drive the teacher's number down by
steering toward latents the value happens to score well; it's forced onto the robust signal.
So a worse fit to the teacher = equal real-world success. **Do not select min0 variants by
E_final** (it will pick the exploiters). Select by 3-draw eval only.

## 7. Open threads (in expected-value order)

1. **min0 has never had a hyper sweep.** Every min0 run used the champion's inherited
   schedamax hypers. Actor-side knobs are unexplored *for this input config*: mean-weight
   (loss shaping — S2 showed a strong monotone effect on the v3 base, 0→0.1→0.3; never
   tried on min0 OR champion; top candidate given min0's different E-regime), amax, K,
   steps. The critic-side recipe stays fixed (same TD warm start + schedamax schedules,
   mechanically re-run inside each tandem run) — no separate teacher work. ~6–8 runs.
   This is the most likely path to >90 that hasn't been tried.
2. **Failure autopsy.** The champion/min0 fail ~6 of 50 episodes per draw. Are those actor
   failures, or world-model / value failures? Cross-reference which episodes fail across
   min0 / champion / latent+CEM and watch the min0-specific ones. Disk-only, no GPU. Tells
   you whether >90 is even reachable by changing the actor.
3. **Structural:** horizon > 5 blocks, K-schedule, eval-time K extrapolation (scalar-iter
   makes K>8 well-defined on the v3 line, not min0).

## 8. Related files

    LIP_AC_MLP_CHAMPION.md   full champion method + explicit f_θ; min0 = its §9
    EXPERIMENTS.md           full campaign log (sequential → AC → v3 arc → input sweep → this)
    LIP_AC_MATH.md           v3 transformer line (closed; loses at H=5)
    run_ac90s{3,5,6}_ogbench.sh, run_min0_confirm.sh   the drivers that produced these numbers

## 9. Provenance

min0 emerged from a user-directed input-ablation square on the champion (drop z_g / z_0 /
both, 2 seeds each; S3→S5→S6), then a 3-draw confirm (run_min0_confirm.sh). Training mode:
the SAME tandem actor–critic recipe as the champion (warm + schedamax) — each run co-trains
its own critic from the shared TD warm start (--init-value cf_dE_t003n50.pt in the S3/S5/S6
drivers) for the first 80% of steps, then freezes it for the tail (--freeze-critic-frac 0.8
default). The coupling is one-directional: with --expand-weight 0 the critic learns only
from the fs1-cache TD pairs and never sees the actor, so its trajectory is actor-independent
— identical recipe per run, differing only in seed/batch order. (Corrected 2026-07-14: an
earlier version of this section claimed sequential-style training against a FROZEN teacher;
that matches neither the drivers nor how the champion was made.) The whole LIP-AC campaign,
its dissociation finding, and the closed v3 transformer arc are in EXPERIMENTS.md.
