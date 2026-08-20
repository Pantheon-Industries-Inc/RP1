# LIP planning on OGBench cube: LeWM vs PLDM world models

**Status:** COMPLETE — LeWM line 2026-07-14; PLDM line 2026-07-16 (4 sweep rounds,
champion + co-champion 3-draw-confirmed at both horizons, failure mechanism located
and offline fix attempts bounded, §6).
**One-line:** the LIPv4 actor–critic planner transfers from the LeWM world model to the
PLDM world model unchanged in architecture, beats base PLDM+CEM planning on every
confirmed cell (+12.3 success at h25, +7.0 at h50) at ~1/500 the planning compute, and
its *relative* advantage over CEM is largest exactly where the world model's native
cost is weakest.

---

## 1. What is being compared

**LIPv4 ("min0") actor.** A 2×512 MLP applied K times as a learned refinement rule over
an action plan. Input per iteration: `[vec(A), vec(∇_A V), V]` — the current plan, the
gradient of the learned terminal value through the frozen world model, and the value
itself. No raw state, no raw goal, no gate: task information reaches the actor *only*
through the learned value ("the value is the only teacher", and the only input).
Deploy: `A(0)=0`, K weight-tied refinement iterations, execute, replan (receding
horizon). Cost per replan ≈ K+few WM rollouts (~16 rollout-equivalents).
Code: `rlp/core/solver/lip.py` (`kind='lip4'`), trainers
`rlp/train/lip.py` (frozen teacher) and `rlp/train/lip_ac.py`
(tandem actor–critic used everywhere below).

**Tandem critic (AC).** Goal-conditioned temporal-distance TD (HER hindsight goals,
balanced full-horizon + 30% cross-episode, quasimetric head, n-step 50, low expectile —
optimistic toward the min), warm-started from a sequential TD checkpoint, co-trained
1:1 with the actor via an EMA teacher, frozen for the last 20% of steps. The critic
learns from the dense cache only — the coupling to the actor is one-directional.

**Baselines.** (a) *Native CEM*: the world model's own latent-MSE terminal cost searched
by CEM (~9,000 rollouts per replan) — for PLDM this is "base PLDM planning" as in the
PLDM paper (modulo MPPI→CEM, the repo's standard protocol). (b) *TD+CEM*: same CEM
search with the learned TD value as cost — isolates value quality from search paradigm.

**World models** (architecturally identical twins: ViT-tiny/14@224 encoder, 192-d
latent, depth-6 predictor over 3-frame windows, 25-d action blocks = 5 primitives):

| | LeWM `ogbench_cube_single_v2WM` | PLDM `PLDM_OgBench` |
|---|---|---|
| training objective | SIGReg (LeJEPA) | VICReg-style + temporal + IDM (Sobal et al. 2025) |
| provenance | authors' exact checkpoint (84.0-replication verified) | authors' baseline checkpoint |

The PLDM checkpoint arrived in the old HuggingFace ViT key layout; it was converted by
pure key rename (198-key bijection) and validated numerically: a from-scratch reference
forward of the old ViT semantics matches to 1.1e-5; encode/rollout/cost paths through
the repo's LeWM execution wrapper match the original PLDM class to ~2e-6 relative.
Converted checkpoint: `checkpoints/PLDM_OgBench_lewm/`; converter + validation:
`pldm_lip_20260715/convert_pldm_encoder.py`.

## 2. Protocol

lewm-cube full protocol: 10k-episode expert h5, 50 tasks/draw, draws = eval seeds
{42, 43, 44} (identical task sets for every planner and both WMs), goals = expert state
25 (h25) or 50 (h50) primitives ahead, env budget 2× offset, success = block within 4cm,
224px, bf16. Values/actors train on caches encoded by the *respective* WM's encoder;
per-WM TD warm starts (τ0.03/n50/6k: `cf_dE_t003n50` for LeWM, `cf_pldm_t003n50` for
PLDM).

**Harness anchors** (fresh pod, before any PLDM number was trusted): LeWM champion
actor s43 h25 → 96.0 (documented 96.0, exact); LeWM TD+CEM s42 h25 → 82.0 (documented
82.0, exact). Determinism spot-check: repeated eval reproduced to the decimal.

**Random floors** (draw-dependent — ~half of h25 tasks start quasi-solved): h25 =
46/56/50 (mean 50.7), h50 = 30/34/28 (mean 30.7). Floors depend only on env+tasks, not
the WM. "Over-floor" = per-draw (score − floor); it isolates what planning contributes.

## 3. Results — absolute success (%, 3 draws × 50 tasks)

### h25

| planner | LeWM: 42 / 43 / 44 | mean | PLDM: 42 / 43 / 44 | mean |
|---|---|---|---|---|
| random floor | 46 / 56 / 50 | 50.7 | 46 / 56 / 50 | 50.7 |
| native CEM | 80 / 84 / 62 | 75.3 | 64 / 78 / 50 | 64.0 |
| TD+CEM | 82 / 90 / 66 | 79.3 | 78 / 64 / 64 | 68.7 |
| **LIPv4-AC** | **88 / 96 / 80** | **88.0** | **70 / 83 / 76** | **76.3** |

LeWM LIP row = tandem champion ("schedamax", K8); the input-minimal min0 ties it
(87.8, 4 seeds) at lower variance. PLDM LIP row = confirmed champion "k12"
(schedamax schedules + K=12 + amax 3.5, min0 no-gate), **mean of two training seeds**
(seed cells: 68/88/74 and 72/78/78 — 3-draw means 76.7 and 76.0, i.e. seed-consistent
to 0.7).

### h50

| planner | LeWM: 42 / 43 / 44 | mean | PLDM: 42 / 43 / 44 | mean |
|---|---|---|---|---|
| random floor | 30 / 34 / 28 | 30.7 | 30 / 34 / 28 | 30.7 |
| native CEM | 58 / 66 / 38 | 54.0 | 48 / 56 / 32 | 45.3 |
| TD+CEM | 64 / 74 / 56 | 64.7 | 46 / 60 / 34 | 46.7 |
| **LIPv4-AC** | **74 / 86 / 62** | **74.0** | **50 / 64 / 43** | **52.3** |

(LeWM h50 = AC-warm confirm; the h50-swept sequential LIP reaches the same 74.0 mean.
PLDM h50 = mean of the same two k12 seeds: 52/62/42 and 48/66/44.)

**Headline (PLDM):** LIPv4-AC beats base PLDM+CEM by **+12.3 at h25 and +7.0 at h50**,
with no confirmed cell below its baseline counterpart, largest gains on the hardest
draw (s44 h25: +24/+28), at ~16 vs ~9,000 rollouts per replan.

## 4. Results — floor removed

Over-floor performance (per-draw score − floor, averaged), and the over-floor ratio
vs native CEM:

| metric | LeWM h25 | PLDM h25 | LeWM h50 | PLDM h50 |
|---|---|---|---|---|
| native CEM − floor | +24.7 | +13.3 | +23.3 | +14.7 |
| TD+CEM − floor | +28.7 | +18.0 | +34.0 | +16.0 |
| **LIPv4-AC − floor** | **+37.3** | **+25.7** | **+43.3** | **+21.7** |
| TD+CEM ratio vs CEM | 1.16× | 1.35× | 1.46× | 1.09× |
| **LIPv4-AC ratio vs CEM** | **1.51×** | **1.93×** | **1.86×** | **1.48×** |
| LIP failure-rate reduction vs CEM | 51% | 34% | 43% | 13% |

Two floor-honest readings:

1. **The learned planner's relative advantage is anti-correlated with the WM's native
   cost quality.** PLDM's latent MSE is a much weaker planning signal than LeWM's
   (+13.3 vs +24.7 over floor at h25), and precisely there LIP's multiplier is largest:
   **1.93× at h25** (vs 1.51× on LeWM). A learned value + learned optimizer recovers
   most of what the weaker representation loses.
2. **At h50 on PLDM the learned value stops paying in CEM** (TD+CEM 1.09× ≈ native) —
   yet the actor still extracts 1.48×. On LeWM, both learned stages keep growing with
   horizon (1.46× / 1.86×). The actor line is the more robust consumer of the value
   across WMs and horizons.

## 5. Recipe transfer (what had to be re-tuned)

Architecture transferred; schedules did not. 3 sweep rounds, 22 tandem runs,
selection on s42+s44 h25 sums, all min0 no-gate, warm-started τ0.03/n50:

| knob | LeWM optimum | PLDM optimum | evidence |
|---|---|---|---|
| refinement depth K | 8 (K12 saturated) | **12** | K12 recipes 142–149 vs K8 plateau ~133–139 |
| amax (plan clamp, σ) | 3.5 (with schedules) | **3.5** | amax2.5 arms plateau; amax3.5×K12 = the only combo above it |
| τ / lr schedules | load-bearing ("schedamax") | ~neutral | 3-draw: k12 (sched) 76.3/52.3 ≈ flat35k12 (static) 75.7/50.7 |
| critic n-step | 50 | **50 (inverted-U)** | TD+CEM s42: n25=70, n50=78, n100=60 |
| critic τ | 0.03 | 0.03 | TD+CEM s42: t003=78 > t01=72 |
| mean-weight | 0.1 | 0.1 | 0.3 arm mid-pack |
| actor lr | 3e-4 | 3e-4 | 1e-4 arm worst (124) |

Round-3 seed replication (2-cell sums): flat35k12 {144, 154} mean 149; k12
{142, 150, 134} mean 142; flat35 {152, 140, 144, 120} mean 139 spread 32; everything
else ~133. The two K12+amax3.5 recipes tie on the full 3-draw confirm (above) — the
**combination is load-bearing, the schedules are interchangeable**; flat35's K8 single
seeds do not replicate.

Selection-statistics caveat (paid for twice on LeWM, re-confirmed here twice more):
2-cell sums carry ±9–13 noise (seed-pair symdiff 4–12 episodes); single-seed leads of
<10 are not real (round-1 "flat 144" → 128/128 on replicas; round-3 "flat35k12 149"
lead vanished at the third draw: 75.7 ≈ k12's 76.3). Only seed-replicated 3-draw means
were promoted; the champion (k12) and co-champion (flat35k12) are both confirmed on
all 3 draws × both horizons (flat35k12: h25 74.7/76.7, h50 49.3/52.0 per seed).

## 6. Failure structure on PLDM (per-episode analysis, all 50 evals)

Same draws as LeWM ⇒ episode-level cross-WM comparison is exact.

1. **The LeWM "core" transfers verbatim.** The 12 grasp-with-airborne-goal tasks that
   cap LeWM at ~88 fail at ~100% on PLDM too, for every planner (LIP, native CEM,
   TD+CEM). The counterfeit-grasp optimism is a property of the expert-only *data*
   (zero attempted-and-missed grasps in 2M frames), not of either WM. No planner-side
   change fixes these; the fix is the negative-data WM fine-tune — since **proven on
   the LeWM thread** (2026-07-16: misses-augmented WM × negative-critic × gated min0
   reached 89.3 3-seed, breaking the 88 plateau and the 90.7 core-capped ceiling on
   its best seed; see `ftmisses_lipv4_20260715/`). The same treatment is the obvious
   PLDM follow-up but is out of scope for this planner comparison.
2. **PLDM adds a long-transport failure family** (block must move 15–41cm laterally;
   ~4–5 tasks/draw at ≥75% LIP failure) **plus a wider airborne-lift net** and a fat
   marginal band (~15 tasks/draw flipping at 25–60% — the variance source; LeWM's
   marginal band was ~3 episodes).
3. **The transport mechanism is located** (video + block-tracking + probes): plan 1
   grasps and carries correctly; block motion collapses **exactly at the step-25
   replan boundary**. The replan is queried from the agent's own mid-carry state —
   off the expert manifold the actor trained on (and with deploy's zero action-history
   vs training's expert histories). CEM tolerates off-manifold queries (coarse ranking
   suffices — TD+CEM solves the s42 transport set with the same value); the gradient
   refiner does not. Confirming probes: value-guided initialization at deploy — no
   transport flips (and broad damage: zero-init-trained actors degrade from any
   non-zero start); replanning *more* often (receding 2–3 blocks) — strictly worse
   (60/48 and 58/56 vs 68/72), i.e. more off-manifold queries, more damage.

**Offline fix attempted and bounded ("replan curriculum",
`replay_prob`):** with probability r an actor-training context is replaced by the
previous step's final *imagined* window (same goal, zero action history) — the replan
query as imagination renders it, keeping the recipe strictly offline. Result (round 4,
r∈{0.3,0.5}×2 seeds): selection parity with the k12 base (means 143/141 vs 142) and
**1/30 transport cells flipped** — imagination-generated boundary states do not stand
in for real ones. Combined with the deploy-time probes, this bounds Mode 2: the
failure lives in the gap between imagined and *real* agent-visited states at the
replan boundary, and no offline, planner-side change tested here closes it. The two
remaining routes are protocol-changing (training on real self-play rollouts) or
WM-side (the negative-data fine-tune above, which targets exactly these off-manifold
contact states).

## 7. Reproduction

Pod layout and all drivers: `pldm_lip_20260715/` (campaign README with full status
log), `lip_ac_20260712/` (LeWM AC campaign), `pod_migration_20260709/`
(full-protocol LeWM writeup incl. floors and h50/h75 sweeps).

PLDM champion training call (k12):

    pixi run training model=lip_ac \
      training.cache=cube_pldm_fs5.pt training.cache_td=cube_pldm_fs1.pt \
      training.h5=cube_single_expert.h5 training.wm=checkpoints/PLDM_OgBench_lewm \
      training.init_value=cf_pldm_t003n50.pt \
      core.planner.horizon=5 core.planner.iterations=12 training.steps=8000 training.n_step=50 training.batch=128 \
      training.expectile=0.1 training.expectile_final=0.03 training.critic_lr=1e-3 training.critic_lr_final=1e-4 \
      training.actor_lr=3e-4 training.actor_lr_final=3e-5 core.planner.action_limit=3.5 \
      core.planner.drop_state=true core.planner.drop_goal=true core.planner.use_gate=false runtime.seed=0

    # eval (per draw / horizon):
    pixi run inference benchmark=lewm runtime.seed=42 runtime.bfloat16=true \
      benchmark.image_size=224 data.path=<h5> benchmark.goal_offset_steps=25 \
      planning.budget=50 core.world_model.checkpoint=checkpoints/PLDM_OgBench_lewm \
      core/solver=lip core.solver.actor_path=pldm_k12_s0.pt

Artifacts (pod 213.181.105.210:14755): actors `/workspace/actors/pldm_*.pt`, values
`/workspace/metrics/`, all eval rows `/workspace/results/summary.csv`, per-episode
success arrays in `/workspace/logs/ev_*.log`, failure-analysis script
`/workspace/valscripts/analyze_failures.py`, transport videos
`/workspace/videos/k12s0_s42/`.
