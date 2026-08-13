# LIP — Learned Iterative Planner

A planning algorithm learned purely by optimizing against a frozen value
function through a frozen world model. No behavior cloning, no policy
gradient. See `rlp/core/solver/lip.py` for the method.

> LIP is the paper's **RLP** planner. For the maintained end-to-end
> replication commands (including the composed `model=rlp` pipeline), see
> [docs/replication/REPLICATION_RLP.md](../replication/REPLICATION_RLP.md);
> some commands below predate the current config keys.

## Pipeline

```bash
# 1. Build a latent cache of the play dataset (block-subsampled, frameskip 5)
#    (LeWM: pooled emb; DINO/PreJEPA: pooled 384-d pixel patches)

# 2. Train the value (frozen teacher): offline TD, quasimetric head
pixi run train model=metric cache=<cache.pt> learner=td \
  core.value.head=quasimetric expectile=0.03 n_step=5 out=<value.pt>

# 3. Train the planner (pathwise through the frozen WM, HER goals)
pixi run train model=lip cache=<cache.pt> h5=<play.h5> \
  wm=<wm_ckpt> value=<value.pt> core.planner.iterations=4 lr=3e-4 out=<lip.pt>
#    (PreJEPA world models: train_lip_dino.py, patch-space rollouts)

# 4. Evaluate (pure learned planner, no sampling)
pixi run eval model=lewm core.world_model.checkpoint=<wm_ckpt> \
  core/solver=lip core.solver.actor_path=<lip.pt> runtime.seed=42
```

## Benchmarks (OGBench visual cube, val planning eval, n=50/eval)

Multi-seed means (8 seeds where marked; frozen LeWM ViT-tiny WMs):

| task   | LIP            | tuned TD + CEM | latent + CEM |
|--------|----------------|----------------|--------------|
| single | **80.3** (8s)  | 73.3           | 64.0         |
| double | 74.3 (8s)      | 72.7           | 74.5 (8s)    |
| triple | 58.7           | 58.7           | 62.7         |

- Compute: ~16-20 WM rollouts per replan vs ~9000 for CEM (300x30) — ~450x less.
- Hyperparameters (fixed 4k-step budget): shallow K=4 @ lr 3e-4 ties K=8 @ 8k
  steps on single; double prefers K=4 @ lr 1e-3; deep K needs hot lr or 2x steps.
- Single-seed scores carry +-6-7 pts of noise at n=50: compare multi-seed means only.
- Known limits: gains scale with how badly raw latent distance reflects true
  cost-to-go (TwoRoom wall: +30..40; single cube: +16; double/triple: tie —
  the value cannot rank candidates refined against it, Spearman(imagined, real)
  ~ 0.02-0.09, so search/selection on top of LIP does not help there).

## LIPv4 (current default): minimal-input, gate-free

`core.planner.architecture=v4` (the default in `lip.py` and `lip_ac.py`; checkpoint kind
`lip4`). The update rule sees **only** `[A, ∇_A V, E]` — no raw z0/z_g, no
gate: `A_{k+1} = clip(A_k + f_θ(A_k, ∇V, E), ±amax)`. The state and goal
reach the planner exclusively through the value function — the purest
learned-optimizer form. The full-input gated MLP uses
`core.planner.architecture=mlp`; the transformer uses
`core.planner.architecture=traj`. Ablations use `core.planner.drop_state`,
`core.planner.drop_goal`, and `core.planner.use_gate`.

Why it is the default (evidence across two envs):
- **OGBench cube** (input sweep + 3-draw confirm): dropping raw z0/z_g ties
  the full-input champion (87.8 vs 88.0) at lower seed variance; raw-latent
  inputs let the actor exploit value idiosyncrasies (worse-loss actors eval
  better once inputs are minimal).
- **TwoRoom** (2026-07-15 campaign): min0/v4 arms *beat* the full-input
  tandem control; the canonical LIPv4 recipe is **triple-perfect** — 3/3
  training seeds at 100.0 on all 12 cells (3600/3600 episodes), plain
  deploy. Checkpoints + exact recipe: `checkpoints/tworoom_lip4/`. Full
  campaign: `tworoom_min0_20260714/` (writeup + per-cell scores).

Same-protocol baselines on TwoRoom (per-cell means, std/hard × h25/h50):
Latent+CEM 300×30 = 84.0 / 58.0 / 82.7 / 68.0 (73.2 overall, ~450× the
compute per replan); random = 11.5 overall.

Recipe notes that matter for the gate-free form:
- **amax substitutes for the gate** — tune it tighter than the gated form
  (TwoRoom: 2.2; monotone worse toward 3.0). It is env-specific (cube's
  gated champion used 3.5).
- **`max_delta=12`** (HER goals to 60 primitive steps) fixed the residual
  cross-wall wall-trap failures systematically on TwoRoom; with default
  max-delta 10 the per-seed perfect rate was a ~50% coin flip.
- **head-scale small-init does NOT transfer to the MLP** (it was the v3
  transformer's no-gate fix): `core.planner.head_scale=0.01` was the worst TwoRoom arm.
  Keep the default 1.0.
- Train with the tandem (`train_lip_ac.py`, warm-started from a sequential
  TD value) — one run replaces the TD sweep + CEM selection + LIP sweep
  pipeline, and the during-run τ anneal (0.1→0.03) does not reintroduce the
  frozen-sharp-teacher pathology.

## LIP-v2: `core.planner.feed=end`

`core.planner.feed=end` additionally feeds the refiner the imagined terminal
latent (checkpoint kind `lip2`; the solver handles it transparently). Verdict
from a 12-cell sweep + replica study: equal peak performance (8-seed paired
diff +1.0 vs v1), but a tighter training-draw distribution — 5/5 paired draws
>= v1 across two tasks (e.g. K8/lr3e-4/4k: {80,82,84} vs v1 {76,78,78}).
Use it when training reliability matters; v1 (`core.planner.feed=none`) stays
the default. `core.planner.feed=traj` (full imagined path) tested worse.

Noise discipline for any comparison on this benchmark: single-training-run,
single-eval-seed cells carry ±6-8 (train) and ±6-7 (eval) points of noise —
compare multi-seed means or paired draws only.
