# HWM × LIPv4 on LeWM / OGBench-cube — results (2026-07-19, FINAL — campaign complete 06:51)

**Question:** does HWM-style hierarchical planning (arXiv 2604.03208: high-level WM over
latent macro-actions → first predicted waypoint = subgoal for the low level) improve
LIPv4 planning on the LeWM cube stack, at short horizons (core-grasp subgoal hypothesis)
and long horizons (h200 universal cliff)?

**Answer:** **No hierarchy configuration beats its flat counterpart at any horizon.**
Best case tie (h100: cutoff-hierarchy 78.7 = flat LIPv4 78.7; hybrid 76.7), worst case
catastrophic (always-on at short range: 43.3 vs 87.6). The mechanism is fully diagnosed
(below). h200 resists every method — it is the data-support boundary (episodes are 201
steps; no training trajectory or in-episode value pair spans 200).

## Protocol

Canonical cube protocol: 10k-episode lewm-cube h5, task draws = eval seeds 42/43/44
(seed drives draw + solver rng), 50 tasks/draw, success = block within 4cm, budget = 2×offset,
++bf16, img 224, OMP capped. Harness anchors on this pod: champion s43 = 96.0 (ref 96.0),
TD+CEM s42 = 82.0 (ref 82.0) — **bit-exact**. HLIPSolver smoke: 75.0 on 4-episode probe.

## Card (3-draw means; per-draw s42/43/44 in results/summary_hwm.csv)

| method | h25 | h50 | h100 | h200 |
|---|---|---|---|---|
| flat latent+CEM | 75.3 (ref) | 54.7 | 62.0 | 2.0 |
| **flat LIPv4 (5 blocks, the standard)** | **87.6 (ref)** | **68.0** | **78.7** | 1.3 |
| flat LIPv4 wide (10 blocks, n50 value) | – | 63.3 | 63.3 | – |
| flat LIPv4 wide (10 blocks, n100 value) | – | 64.0 | 62.0 | 2.0 |
| hlip (always-on, LIP both levels) | 43.3 | 23.3 | 48.7 | 1.3 |
| hcem (always-on, CEM both levels = HWM-faithful) | 54.0 | 42.0 | 45.3 | 1.3 |
| hlipc (cutoff 35: HL far-phase only) | – | 58.0 | **78.7** | 1.3 |
| hcemc (cutoff, CEM) | – | 46.7 | 60.0 | 1.3 |
| hlipp (always-on + subgoal NN-projection) | 42.7 | – | 48.7 | – |
| hyb0 (CEM subgoals + LIP low level) | 56.0 | – | – | – |
| **hybaut (hybrid + reach-selected waypoint)** | **83.3** | **66.7** | **76.7** | 2.0 |

(Cross-horizon comparisons are invalid — each h has its own task draw pool; compare within columns.)

## Findings

**F1 — Flat LIPv4's value function already performs the hierarchy's job.** Receding
25-step windows + the stitched quasimetric (env-steps-to-go, HER cross-episode, n-step 50)
hold 78.7 at h100 — far beyond the actor's 50-env-step training-goal range. The "chain of
strong short-range signals" that hierarchy is supposed to provide is already implicit in
value-guided receding-horizon planning. Explicit latent subgoals can only match it.

**F2 — Always-on subgoaling is poison at short range, and the learned HL actor is the
worst generator.** h25: hlip 43.3 (floor) < hcem 54.0 < hyb0 56.0 (CEM subgoals + LIP LL)
≪ flat 87.6. Subgoal probe (probe_subgoals2.py, ridge readout R²=.993): for goals 21.5
env-steps away, the HL-LIP subgoal reads 42 env-steps away (2× detour), NN-dist 5.6 vs
2.3 real-latent baseline (off-manifold), decodes the block +6cm airborne with zero xy
progress. Mechanism: the HL actor's slack 8-macro plans fill with the expert lift-prior
(cube experts always lift); the terminal-only objective never pins wp[0] en-route. This is
macro-level off-manifold optimism — the twin of the documented LL attachment-optimism.

**F3 — Manifold projection does NOT fix it** (hlipp 42.7 ≈ hlip 43.3): the nearest real
latent to an imagined airborne state is a real airborne state; the semantic detour
survives. Diagnosis-first beats mechanism-patching: the fix must change WHICH waypoint is
proposed, not clean it afterwards.

**F4 — Reachability-arbitrated waypoint selection (subgoal_select=reach) removes the
penalty but adds no gain.** Pick the predicted waypoint whose LL-value distance ≈ 30
env-steps among progress-making waypoints (the trusted LL quasimetric as arbiter). hybaut
83.3/66.7/76.7 ≈ flat everywhere: it self-regulates into flat behavior near goals and
subgoal-chains far away — a hierarchy that has learned when not to interfere, and
plateaus exactly at flat's level.

**F5 — "Just roll the stable WM deeper" fails (the wide-window controls).** 10-block flat
actors, with the value span kept (n50) or doubled (n100), are at-or-below the 5-block
standard at every horizon (63-64 vs 68 at h50; 62-63 vs 78.7 at h100). Value span is a
wash → window widening itself is net-negative: deeper refinement compositions are harder
to optimize and wander further off-manifold. Short windows + frequent replanning + a
stitched value is the stronger flat design.

**F6 — Prediction stability ≠ planning reach, in both directions.** FIG6: the frozen
v2WM's autoregressive rollout beats F² at expert-action prediction at ALL horizons (ℓ1
0.37 vs 0.42 at 200 env steps; z-scale 0.80) — LeWM's stability removes the paper's
compounding-error opening. A 3× capacity/2.7-D-deeper F² (hwm_s25m8d10) improves 1-2-step
fit but WORSENS 200-step rollouts (0.56) — the high level's error floor is the 125→8-D
macro information bottleneck, not capacity. Conversely flat's stable predictions don't
crack h200 either: guidance, not prediction, binds at range.

**F7 — h200 is a data-support boundary.** Every family — flat (narrow/wide), always-on,
cutoff, hybrid — lands ≤6 at h200 (hlipc/hcemc 1.3, hybaut 2.0, wide-flat 2.0). Episodes
are 201 steps: no trajectory demonstrates a 200-step task, and no in-episode value pair
spans it. Cracking h200 needs new DATA (self-play, stitched multi-episode) or explicit
compositional value training, not a better planner on the same data.

## Where this leaves the HWM paper's claims

Their gains (VJEPA2-AC Franka 0→70; DINO-WM PushT d75 17→61; PLDM maze 44→83) live in
regimes our stack does not occupy: (a) backbones whose long rollouts genuinely degrade
(huge token-space models on real video — our pooled 192-d LeJEPA latent is near-contractive),
and (b) flat baselines that are sampling planners with local goal-matching costs and NO
long-range value. On a stack with a trained quasimetric + learned refiner, both of
hierarchy's advertised advantages (fewer autoregressive steps; smaller search space) are
already priced in. The one HWM design element that transfers unconditionally: bypass the
hierarchy near the goal (their flat final-approach = our cutoff/reach-select).

## Methods (reproducibility)

- HWM high level: train_hwm.py — A_psi MLP posterior (125→256²→8, LayerNorm mu,
  deterministic), F2 = LeWM Predictor clone (AdaLN ConditionalBlock, 192-d, depth 6,
  ctx 3) + Embedder(8→192); rollout-MSE, stride 25 env steps, 9 waypoints/ep, holdout
  eps 9500+; solver-convention contexts (z0×3 history, zero macro history) in training
  AND deployment. Sweep: stride {15,25,50} × macro {4,8} × AE {mlp,tf} × loss {mse,l1};
  s25m8 primary. Macro space needs no z-scoring (LayerNorm); amax 3.0 ≈ 2/98-pct clamp.
- HL LIPv4: train_lip_hl.py — tandem schedamax-6k on dense fs1 cache (n-step 50, τ .1→.03,
  warm from cf_dE), PlannerNet v4 over macros, horizon 8, E_final+0.1·mean-path.
- Solver: stable_worldmodel/solver/hlip.py (HLIPSolver) — hl/ll ∈ {lip,cem} each,
  hl_cutoff, subgoal_index / subgoal_select=reach (reach_target 30), subgoal_project
  (+project_pool). Config scripts/plan/config/solver/hlip.yaml.
- Probes: probe_subgoals2.py (subgoal quality vs LL value + ridge readout + NN),
  FIG6 built into train_hwm.py (--fig6-wm).
- Pod: 31.24.80.32:14476 (6×H100). Drivers: run_hwm_campaign.sh (tiers), run_cutoff_variants.sh,
  run_hybrid.sh, run_projected.sh, run_flatwide_control.sh, run_flatwide_n50.sh,
  run_hl_trainings.sh, run_hwm_sweep.sh. Scores: results/summary_hwm.csv. GPU claim
  locks /workspace/.gpu[0-5].
- Ops gotcha: tar/rsync chown onto RunPod net-volumes fails and poisons exit codes —
  always --no-same-owner / -rlt (one fully-downloaded 95GB h5 was deleted on this false
  negative before diagnosis).

## Open directions

1. h200 via data: self-play / multi-episode stitching for the value; or train F2/critic
   on cross-episode HER waypoint chains (the quasimetric stitches states, nothing
   stitches SUBGOAL curricula yet).
2. HL actor objective: per-step en-route losses (path-weighted, not terminal-only) would
   attack F2-actor detours directly if hierarchical subgoaling is revisited.
3. The reach-select arbiter (LL quasimetric judging waypoint reachability) is a reusable
   primitive independent of HWM — e.g. for option/skill selection.
