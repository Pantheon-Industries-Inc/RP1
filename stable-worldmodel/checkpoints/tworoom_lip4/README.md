# tworoom_lip4 — canonical LIPv4 planners for TwoRoom (triple-perfect)

Three independently seeded **LIPv4** actors (kind `lip4`: minimal-input
gate-free learned iterative planner, inputs `[A, ∇_A V, E]`, 101-d, 340,530
params) for `swm/TwoRoom-v1` on the frozen LeWM ViT-tiny WM
(`checkpoints/tworoom`, weights_epoch_16_partial).

**Every actor scores 100.0 on all 12 eval cells** ({std, hard-cross-wall} ×
{h25, h50} × eval seeds {42,43,44}, n=50) — 3600/3600 episodes, plain deploy
(single planner pass, `restarts=1`, no sampling). Campaign of record:
`../../../tworoom_min0_20260714/` (WRITEUP_tworoom_min0.md,
RESULTS_tworoom_scores.md).

## Files

| file | role |
|---|---|
| `trm_v4c_md12_s{0,1,2}.pt` | LIPv4 actors (training seeds 0/1/2), self-contained for `solver=lip` |
| `trm_v4c_md12_s{0,1,2}_value.pt` | each actor's tandem teacher (MRN quasimetric, 192→256²→128), usable as `+metric=` for CEM |
| `td_e0.1_n50_warmstart.pt` | the sequential TD value (τ0.1, n-step 50) used to warm-start every tandem run |

## Training recipe (exact)

```bash
python scripts/plan/train_lip_ac.py \
  --arch v4 --amax 2.2 --max-delta 12 \
  --iters 8 --horizon 5 --steps 8000 --n-step 50 \
  --expectile 0.1 --expectile-final 0.03 \
  --critic-lr 1e-3 --critic-lr-final 1e-4 \
  --actor-lr 3e-4 --actor-lr-final 3e-5 \
  --init-value td_e0.1_n50_warmstart.pt \
  --cache <tworoom_lewm_fs5.pt> --cache-td <tworoom_lewm_fs1.pt> \
  --h5 <tworoom_play.h5> --wm lewm_tworoom --seed {0,1,2} \
  --out <actor.pt> --out-value <teacher.pt>
```

All other flags at their defaults (batch 128, td-batch 1024, p-cross 0.3,
mean-weight 0.1, freeze-critic-frac 0.8, ema-tau 0.005, γ=1, head-scale 1.0).
Data: `tworoom_play.lance`, 1000 expert episodes, **deterministic** —
regenerate with `collect_play.py --episodes 1000 --num-envs 32 --seed 7`
(ExpertPolicy noise 2.0 / repeat 0.05); caches via `scripts/trm/cache_latents.py`
(fs1, --state-key state) + `subsample_cache.py --frameskip 5`.
Training ≈ 95 min on one H100.

The two load-bearing recipe choices (vs the cube schedamax base):
- `--amax 2.2` — without the gate, a tighter action clamp provides the
  per-iteration damping (monotone: 2.0/2.2 ≫ 2.5 ≫ 3.0; cube wanted 3.5
  *with* a gate — env-specific).
- `--max-delta 12` — HER goals to 60 primitive steps. Fixes the residual
  cross-wall "wall-trap" failure mode (greedy descent pins the agent at the
  wall instead of detouring through the door) systematically; without it the
  per-seed perfect-card rate is ~40–60%.

## Eval

```bash
python scripts/plan/eval_wm.py --config-name tworoom_lewm \
  seed=42 eval.goal_offset_steps=25 eval.eval_budget=50 \
  solver=lip solver.actor_path=checkpoints/tworoom_lip4/trm_v4c_md12_s0.pt
# hard surface: +eval.cross_wall=true ; h50: goal_offset_steps=50 eval_budget=100
```

**Path caveat:** each actor checkpoint stores the absolute path of its teacher
(`ck["value"] = /workspace/metrics/trm_v4c_md12_sX_value.pt`) and LIPSolver
loads the value from there. On a machine with a different layout, rebind once:

```python
import torch
ck = torch.load("trm_v4c_md12_s0.pt", map_location="cpu", weights_only=False)
ck["value"] = "/abs/path/to/trm_v4c_md12_s0_value.pt"
torch.save(ck, "trm_v4c_md12_s0.pt")
```

## Provenance

Trained 2026-07-15 on pod a (4×H100), campaign `tworoom_min0_20260714`
round C, arm `md12`; md5s match the pod originals (verified before pod
deletion). Reference baselines on the identical protocol: Latent+CEM
(300×30) 73.2 overall, random 11.5, old full-input LIP v1 anchor 99.7.
