# RLP unified-config campaign — results (2026-08-27)

Median-primary reporting (per-run early stopping, n=6 where noted, no
cross-seed selection). Metrics: TwoRoom/Cube success at goal-offset 25/100
("h25/h100"); Reacher latched first-hit at τ=0.1/0.05. Paper column = the
RLP paper's reported numbers (3 seeds, select-on-42 era).

## Headline

A single planner+training configuration — **deep-capacity refiner+critic,
anti-constancy regularizer, K=8** — reaches or beats the paper on 9 of the
10 numbers it reports for these cells. The only per-environment settings are the two that are genuinely
environment properties, not tuning: the **observation window** (1 frame for
position-goal TwoRoom/Cube, 2 for velocity-dependent Reacher) and the
**per-env step budget**. Two of the three environments' residual failures,
diagnosed mechanistically this campaign, were then *fixed*; the third
(Cube's task-intrinsic hard core) was shown to be a property of the
benchmark, not the method.

## FINAL table (median, uniform n=6)

| cell | h25 / τ.1 | h100 / τ.05 | paper | Δ h100/τ.05 |
|---|---|---|---|---|
| TwoRoom LeWM | 100.0 | 96.8 | 94.2 | **+2.6** |
| TwoRoom PLDM | 95.9 | 94.0 | 96.0 | −2.0 |
| Cube LeWM | 89.3 | 83.3 | 89.1 / 82.4 | **+0.9** |
| Cube PLDM | 84.0 | 81.2 | 82.9 / 77.1 | **+4.1** |
| Reacher LeWM (w=2) | 99.7 | 93.7 | 98.7 / 88.7 | **+5.0** |
| Reacher PLDM (w=2) | 99.7 | 91.0 | 97.8 / 82.0 | **+9.0** |

The paper reports 10 numbers for these cells (TwoRoom rows: h100 only;
Cube and Reacher: both horizons). We are **at-or-above on 9 of those 10**;
the single deficit is TwoRoom-PLDM h100 (94.0 vs 96.0, −2.0) — itself up
from 89.4 at campaign start with the catastrophic-seed mode eliminated
(n=6 seeds 89–97). Our TwoRoom h25 cells (100.0 / 95.9) have no paper
counterpart and are not counted either way.

The λ ladder was completed post-lock at uniform n=6 (see below): λ0.2 has
the best aggregate (724.6 over the 8 TwoRoom/Cube cells vs 720.7 / 719.4 /
716.6 for λ0.1 / 0.3 / 0.5), so the FINAL table stands as selected.

Campaign start (depth-2, no regularizer) had Cube-LeWM at −1.8/−1.4,
TwoRoom-PLDM h100 at 89.4, and Reacher (w=1) at 82.7/72.7 τ.05. Every one
of those moved up.

## The unified config

γ=0.98 · n-step=50 · vnorm=none · boundary=legacy · amax=2.5 (train;
deploy-clip free) · max_delta=20 · **K=8** · **refiner depth-3 + critic
(MRN) depth-3 ("deep-capacity"), 2× step budget** · **anti-constancy
λ=0.2** · per-run early stopping (snapshots every 2000, val draws 48–51) ·
ema_tau=0.005. Per-environment: observation window (w=1 / w=2 Reacher);
step budget (declared, not tuned).

## The two fixes (both discovered this campaign)

1. **Deep capacity, done right.** Actor *and* critic depth 2→3 with 2× the
   step budget. An earlier actor-only depth-3 at fixed budget was flat — the
   critic must deepen too and the bigger net needs more training. Lifts the
   cells with good latent geometry (TwoRoom-LeWM, both Cube bases).
2. **Anti-constancy regularizer** (`--ac-weight`). Penalizes the batch-level
   constancy of the emitted plan displacement,
   `‖E_b[ΣA]‖² / E_b‖ΣA‖²`. Targets the diagnosed TwoRoom failure directly
   (below); scale-free, rollout-free, unified. λ=0.2 selected from a
   5-point ladder and then confirmed post-lock at **uniform n=6** on all 8
   TwoRoom/Cube cells (sums: λ0.1 720.7, **λ0.2 724.6**, λ0.3 719.4,
   λ0.5 716.6; λ0.4 only n=4, not eligible) and re-run on Reacher at the
   adopted w=2 (n=3 per λ, both bases): flat at τ.1 (98.7–100), mild
   low-λ tilt at τ.05 (LeWM best at λ0.1 97.3 vs λ0.2 93.7; PLDM best at
   λ0.3 94.7 vs λ0.2 91.0) — λ0.2 within seed noise of best everywhere,
   closing the earlier w=1-selection caveat. Dose-response is
   cell-dependent (TwoRoom wants heavier — λ0.5 reaches 100.0/100.0 on
   TwoRoom-LeWM; Cube wants lighter) but the aggregate is nearly
   λ-independent over 0.1–0.5.
   - **Deep capacity and anti-constancy are complements, not alternatives:**
     capacity gives the refiner room to be both task-faithful and
     input-sensitive; the regularizer stops it from spending that capacity
     on a world-model exploit. Neither alone suffices on the hard cells.

## Failure atlas — three environments, three unrelated mechanisms

- **TwoRoom seed lottery = actor-only WM-hallucination basin.** Bad PLDM
  seeds emit a ~80%-constant plan the frozen world model *imagines* reaching
  the goal (imagined direction-cos +0.9) while true motion points away
  (−0.5). Critic exonerated: value tracks the BFS geodesic at Spearman
  ρ 0.983–0.992 for every seed; cross-swap is causal (good-actor×bad-critic
  100%, bad-actor×good-critic 19%). The training loss is imagined energy
  through the frozen WM, so the exploit basin and the honest basin are
  equally good minima — hence a training-level lottery invisible to loss
  curves. **Fixed by anti-constancy** (rollout-free canary: plan-constancy
  0.77–0.80 bad vs 0.26–0.39 good). Full write-up +
  figures: `docs/figures/tworoom_failures/DIAGNOSIS.md`.
- **Cube = task-intrinsic hard core.** 12 of 150 h100 tasks fail for every
  seed, both bases, both capacities (8% floor); LeWM carries ~6–9 extra
  representation-specific hard tasks. h100 ceiling ≈ 87; paper 82.4 and our
  82–83 both sit mid-band. Not a method deficiency —
  `docs/campaigns/2026-08-23/FAILURE_ATLAS.md`.
- **Reacher τ.05 = velocity-blind terminal precision (the w=1 artifact).**
  Every τ.05 miss reaches within 0.05–0.13 rad; zero planning failures. The
  single-frame critic cannot represent velocity, so the actor cannot brake
  into the tight ball. **Fixed by w=2** (below).

## Reacher window ablation — the campaign's biggest single swing

Reacher was mid-campaign run at w=1 (an ablation, not the paper's protocol —
the paper uses a 3-frame window). Restoring an observation window transforms
it. Medians, ACR λ0.2:

| config | LeWM τ.05 | PLDM τ.05 |
|---|---|---|
| w=1 deepcap | 82.7 | 72.7 |
| w=3 deepcap | 93.3 | 92.0 |
| **w=2 deepcap** | **96.7 (n=3) / 93.7 (n=6)** | **91.3 (n=3) / 91.0 (n=6)** |
| paper (w=3) | 88.7 | 82.0 |

- **w=2 is the principled choice** — the minimal window that makes velocity
  observable in a second-order system — and it *also wins* (≥ w=3 on LeWM,
  ≈ on PLDM). The paper's third frame added redundant WM-noisy signal.
- **Deep capacity flips from hurting to helping Reacher at w≥2**: at w=1 the
  extra net had no velocity to steer with; at w=2 it does. This dissolves
  the strict-uniformity conflict — deepcap+acr is now best on all three
  environments simultaneously.

### Honest comparison (w=2, latched, same shared critic)

The value function is the shared optimization objective, so the reacher
comparison runs every planner against the same w=2 critic *specification*
(same window, lag, γ, expectile, depth, trained by the same recipe). Precise
caveat: the sampling baselines score with the offline teacher, while RLP
deploys the critic co-trained from that teacher — the standard convention in
this stack, but the two are not the same file at deploy time.

| planner (τ.05 median) | LeWM | PLDM |
|---|---|---|
| **RLP (LIPv4 refiner)** | **93.7** | **91.0** |
| latent + CEM | 88.0 | 88.0 |
| value + CEM | 60.0 | 50.0 |

RLP refinement beats the best sampler at matched window, critic, and
convention on both bases — the planner is the win, cleanly isolated.

## Cube h200 (long-horizon probe)

Cube h200 scores ~5–6% (real, not an artifact — valid tasks exist, unlike
TwoRoom where episodes cap at 100 steps so no h200 task exists). Mechanism:
a steering wall at imagined-distance E≈12–14, the same band where the h100
hard core stalls. Two candidate causes tested:
- **Missing far-range labels** (goals >50 steps get one saturated ceiling
  label): γ=0.995/n=200 *falsified* it — h200 stayed 4–5%.
- **Multi-stage / grasp-boundary barrier**: the surviving explanation. A
  grasp is a near-discontinuity in the action→outcome map; first-order
  refinement on the smoothed world model cannot reliably thread a sequence
  of such funnels. Fix classes (untested, method-level): subgoal chaining
  through demo-latent waypoints, or Dyna to sharpen the WM's contact model.

## Method/infra notes

- Migrated off a quota-exhausted shared volume to a dedicated 1 TiB PVC
  after silent-loss failures (a full volume made *failed* jobs report
  SUCCEEDED because failure sentinels also need disk).
- wandb ≥ (recent) removed `Run.get_url()`; setup probe + telemetry now use
  `r.url` with a fallback. This had silently killed a whole n=6 wave at
  setup.
- Per-seed TD-grid guard + cross-node cache backfill; portable actor import
  (repath `ck['value']` after a `load_metric` probe). All committed.

## PushT: the unified config's validity boundary

PushT (main `model=rlp` pipeline, LeWM base, w=2 lag-5, one-shot planning)
is the one environment where the unified recipe **fails**, and the failure
is attributable. All numbers are medians over seeds 0–2 (each seed = mean
over eval draws 42–44), all trained and evaluated this campaign — the
prior counterstrike-era "66.4" is NOT comparable (different recipe, K=24,
different teacher corner) and every forward-looking Δ initially anchored
to it was wrong; the honest baseline is our own re-run.

| arm | median | note |
|---|---|---|
| latent+CEM (fixed eval) | 78 | sampling baseline |
| base recipe, K=24 (shallow, no acr, w1) | 57.3 | our re-anchored baseline; K=8 arm ±0 at seed 0 |
| base + acr λ0.2 | 62.0 | null at n=3 (means 58.7 vs 59.3, sd 4–6) |
| unified (deepcap+acr+ES), K=8 | 31.3 | **deepcap −39 is the driver** |
| unified, K=16 | 34.7 | |
| unified, K=24 | 42.7 | |
| unified, K=32 | 44.7 | flattening; still −13 vs shallow base |

Attribution at the base config: acr, w=2, K (8↔24), and ES are all within
seed noise; **deep capacity is the single solid effect and it is
catastrophic (−39)** — the capacity-arms-model-exploitation mechanism from
TwoRoom, on latents whose single-frame aliasing PushT's own docs flag.
The K ablation (unified config, only `planner.iterations` varied) shows
refinement depth *partially* compensates: monotone 31.3 → 44.7 from K=8 to
K=32, recovering roughly a third of the deepcap damage before flattening,
but never reaching the shallow config, let alone CEM. Conclusion: on
contact-rich tasks with weak latent geometry the unified config's capacity
prescription inverts — PushT marks its validity boundary, and "refine
more" is not the rescue.

## Open / in flight

- PushT deepcap+acr-at-higher-λ (does heavier regularization disarm the
  deep actor as it did on TwoRoom?) — untested; subgoal chaining proposal
  unimplemented.
