# Counterstrike — PushT RLP campaign (2026-08-15/17)

RLP (LIPv4) trained and evaluated on PushT against the official
`quentinll/lewm-pusht` world model and expert dataset, wired into the
`model=rlp` pipeline (dataset `pusht`, WM converted via `tool=convert_pldm`).
Protocol: h25 **open-loop** replay (goal = state 25 primitive steps ahead,
budget 50, `receding_horizon=5`), held-out split — train episodes 0–15999,
eval draws 16000–18685. Success = the env's DINO-WM criterion. Selection on
eval seeds 50/51 only; reports on 42/43/44. Constraint (user): pure one-shot
LIPv4 — no restarts, no argmin-V selection, no MPPI polish.

## Headline — report protocol (42/43/44 × 50 episodes)

| method | plan compute/decision | 42 | 43 | 44 | mean |
|---|---|---|---|---|---|
| TD+CEM (value objective, τ0.01/n1/γ1) | ~9,000 rollouts | 68 | 86 | 86 | **80.0** |
| latent+CEM (paper's method, SOTA line) | ~9,000 rollouts | 78 | 84 | 76 | **79.3** |
| RLP LIPv4 tuned (K=24), 3 actor seeds | **24 unrolls** | 70/66/64 | 78/72/80 | 62/60/46 | **66.4** (seeds: 70.0/66.0/63.3) |
| RLP LIPv4 cube recipe (K=8, seed 0) | 8 unrolls | 60 | 66 | 46 | 57.3 |
| no-op floor | — | 2 | 0 | 0 | 0.7 |

Winner recipe: `planner.iterations=24 planner.amax=2.2 planner.mean_weight=0.1
planner.actor_lr=3e-4 planner.expand_weight=0` + the PushT offline value
corner `value.gamma=1.0 value.expectile=0.01 value.n_step=1`
(WRITEUP_pusht_lip.md, 2026-07). Everything else = cube defaults.

## The lever: refinement iterations (selection 50/51, amax 2.2, expand 0)

| K | 8 | 16 | 24 | 32 | CEM anchor |
|---|---|---|---|---|---|
| selection mean | 63.0 | 69.0 | **78.0** | 75.0 | 83.0 |

K=24 granular grid (18 cells): mean-weight 0.1 is the best column; amax
plateau 2.0–2.8 at 72–78, off a cliff at 3.0. K=32 grid (15 cells) plateaus
70–75 — saturation at K=24.

## Closed axes

- **Value function is NOT the bottleneck**: TD+CEM 80.0 ≈ latent+CEM 79.3 on
  report seeds (cs-baseline job; value trained on shared caches only).
- **expand_weight=0 > 1** in all four matched pairs (PushT behaves like
  Reacher, not Cube).
- **replay_prob=0** helps at K=8 (+5) but does not stack at K=24 (76.0 vs
  78.0) — K and replay attacked the same optimization slack.
- **BC anchor hurts** (bc 0.1/0.3/1.0 → 64/58/59 vs 63).
- **γ=1/n=1 on the co-trained critic collapses training** (8–13%); the
  short-backup corner is for the offline init value ONLY.
- actor_lr 3e-4 ≥ 1e-4; mean_weight 0.1 ≥ {0.05, 0.2}.
- **Training duration closed**: steps 12k → 73.0, 18k → 68.0 vs 6k → 78.0
  at the winner config (selection 50/51) — flat-to-worse; the K=24 actor is
  not undertrained at the recipe's 6k steps.

## Failure analysis (probe-instrumented, seed-50 identical draws)

One-shot failures are a **heavy right tail of critic-condemned plans**
(planned-E median 2.58, p90 11.6): the refiner commits to the wrong contact
mode (push-left-around vs right-around) and cannot escape locally. Restart
dose-response (diagnostic only, not shipped — vetoed): success 66→72→78 and
p90 planned-E 11.6→5.6→4.4 at R=1/8/16. Within the one-shot constraint,
more/finer iterations (K=24) recovers most of this; the residual ~13-pt
report gap to CEM is the tail K cannot cut.

Selection→report shift: CEM 83.0→79.3, RLP 78.0→70.0 (seed 0) — selection
draws run easier, and actor-seed variance is wide (63.3–70.0; seed 44 is the
hard draw: RLP 46–62 vs CEM 76).

## Artifacts

- Harness: `scripts/sky/counterstrike_pusht.yaml` (+ `launch_counterstrike_{sweep,expand,iters,k_grid}.sh`,
  `counterstrike_baselines.yaml`, `counterstrike_failure_analysis.yaml` +
  `analyze_counterstrike_failures.py`, `counterstrike_eval_actors.yaml`,
  `collect_counterstrike_sweep.sh`)
- Durable checkpoints/caches/results: `/checkpoints/armin@pantheon.inc/{counterstrike-20260815,cs-swp-20260815-*,cs-baseline-20260816,cs-fail-20260816,cs-report-20260816*}`
- Winner actors: `cs-swp-20260815-cs-k24-a22/train_checkpoints/planner.pt`
  (seed 0), `cs-report-20260816-s{1,2}/train_checkpoints/planner.pt`
- ~90 managed jobs total (jobs 7133–7402+), all p1/p2 per convention.

## If the one-shot constraint is ever relaxed

The measured route to CEM parity is inference diversity: R=16 noisy restarts
+ argmin-V scored 78 on seed 50 with the OLD K=8 actor (~300 rollouts,
still 30× under CEM). Closed-loop replanning (`planning.receding_horizon=1`,
what the upstream LeWM PushT number uses) is untested here and would require
a closed-loop CEM baseline for a fair table.
