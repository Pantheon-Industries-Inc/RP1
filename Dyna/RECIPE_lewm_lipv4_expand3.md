# LeWM LIPv4 @ expand-weight 3.0 — the 90.00 recipe

**Result: 90.00 ± 0.67** (seeds 0–2 × 3 draws) / **89.56 ± 0.69** (6 seeds),
OGBench cube-single, h25, held-out episodes `8000:10000`, 50 tasks/cell, EGL.
Three independent confirmations at the correct transformers version (90.22,
90.00 fresh retrain, 90.00).

Every value below is either read back from the shipped artifact
(`/workspace/actors/lip4_re_lewm_exp30_s0.pt`) or taken from the script that
produced it (`dyna_harness/replay_expand_sweep.sh`, arm `exp30`, seeds 0–2 →
`lip4_re_lewm_exp30_s{0,1,2}.pt`; seeds 3–5 came from stage B as
`lip4_bcd_lewm_exp30_s{3,4,5}.pt`). Nothing here is from memory.

---

## 0. Environment — load-bearing, not incidental

```
transformers==4.49.0          # 5.14.1 shifted TRAINING by -3.11 uniformly across
torch==2.8.0+cu128            # seeds; evals reproduce across versions, trains do not
MUJOCO_GL=egl                 # osmesa costs ~7.3 pts on reacher; never use it
PYOPENGL_PLATFORM=egl
MUJOCO_EGL_DEVICE_ID=<gpu>    # pin per process
PYTHONPATH=/workspace/code/stable-worldmodel
STABLEWM_HOME=/workspace/swm_home
```

Never `pip install -e` the package — it drags torch to 2.13 and leaves cu13
wheels shadowing cuDNN (`CUDNN_STATUS_NOT_INITIALIZED`). Use `PYTHONPATH`.

## 1. Base world model — `v2WM`, `weights_epoch_22.pt`

```json
{ "_target_": "stable_worldmodel.wm.lewm.LeWM",
  "encoder":        { "vit_hf", "size": "tiny", "patch_size": 14,
                      "image_size": 224, "pretrained": false },
  "predictor":      { "num_frames": 3, "input_dim": 192, "hidden_dim": 192,
                      "output_dim": 192, "depth": 6, "heads": 16,
                      "mlp_dim": 2048, "dim_head": 64, "dropout": 0.1 },
  "action_encoder": { "Embedder", "input_dim": 25, "emb_dim": 192 },
  "projector":      { "MLP", 192 -> 2048 -> 192, "norm_fn": BatchNorm1d },
  "pred_proj":      { "MLP", 192 -> 2048 -> 192, "norm_fn": BatchNorm1d } }
```

**The latent is the CLS token.** `LeWM.encode` takes
`encoder(...).last_hidden_state[:, 0]` — 1 of 257 tokens — and pushes it through
`projector` to **192-d**. All 256 patch tokens are discarded inside the encoder,
so dynamics, cost, critic and actor all operate in the same 192-d space.

## 2. Latent caches

```bash
python3 scripts/trm/cache_latents.py \
  --wm /workspace/models/v2WM \
  --dataset /workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
  --out v2_full_fs1.pt \
  --state-key privileged_block_0_pos
python3 /workspace/filter_cache_eprange.py v2_full_fs1.pt v2_tr8000_fs1.pt --lo 0 --hi 8000
python3 scripts/trm/subsample_cache.py --in v2_tr8000_fs1.pt --out v2_tr8000_fs5.pt --frameskip 5
```

`fs1` (stride 1) feeds the critic, `fs5` (frameskip 5) the actor. Both filtered
to episodes `0:8000` so the `8000:10000` eval range is held out.

## 3. TD teacher — the initial critic (`--init-value`)

Arm `g98s12k` from `td_upgrade.sh`, the sweep winner:

```bash
python3 scripts/plan/train_metric.py \
  --cache v2_tr8000_fs1.pt \
  --learner td --head quasimetric \
  --expectile 0.03 --gamma 0.98 --n-step 50 --steps 12000 --seed 0 \
  --out up_lewmpre_g98s12k_s0.pt
```

`--head quasimetric` builds `QuasimetricHead`, an **MRN** head — 192 → 256 → 128,
depth 2, `sym_frac` 0.5:

```
d(z_i -> z_j) = || u(z_i) - u(z_j) ||_2  +  max_k ReLU( v(z_j)_k - v(z_i)_k )
```

A directed quasimetric obeying the triangle inequality, which is what lets the
value stitch long distances from short transitions.

## 4. LIPv4 actor + co-trained critic — **the recipe**

```bash
python3 scripts/plan/train_lip_ac.py \
  --cache      v2_tr8000_fs5.pt \
  --cache-td   v2_tr8000_fs1.pt \
  --h5         /workspace/datasets/expert_actions.h5 \
  --wm         /workspace/models/v2WM \
  --init-value up_lewmpre_g98s12k_s0.pt \
  --arch v4 --horizon 5 --iters 8 --steps 6000 \
  --amax 1.6 \
  --batch 256 --n-step 50 --gamma 0.98 \
  --replay-prob 0.5 --expand-weight 3.0 \
  --freeze-critic-frac 0.5 \
  --actor-lr  3e-4 --actor-lr-final  3e-5 \
  --critic-lr 1e-3 --critic-lr-final 1e-4 \
  --expectile 0.1  --expectile-final 0.03 \
  --seed 0 \
  --out lip4_lewm_exp30_s0.pt --out-value lip4_lewm_exp30_s0_value.pt
```

Defaults deliberately left alone, all of which are live in this result:
`--mean-weight 0.1`, `--p-cross 0.3`, `--td-batch 1024`, `--td-p-cross 0.3`,
`--critic-ratio 1`.

Per base: `--amax 1.6` for LeWM, `4.5` for PLDM. Everything else is identical
across the two bases.

> **Copy-paste gotcha.** `replay_expand_sweep.sh` sets
> `BASEFLAGS="--batch 256 --replay-prob 0.5 --expand-weight 1.0"` and the `exp30`
> arm appends `--expand-weight 3.0`, so the flag appears **twice** and argparse
> keeps the last one. The effective value is 3.0. Pass it once when copying.

### Resulting architecture, read back from the checkpoint

```
kind lip4        z_dim 192       a_dim 25        horizon 5     iters 8
amax 1.6         width 256       layers 2        hidden 512    s_dim 256
head_mode gate   use_gate False  use_grad True   head_scale 1.0
use_z0 False     use_zg False    goal_mode diff  cond_mode token
iter_mode scalar feed none       s0_mode zero    gd_init 0.0
feat_norm False  pre_ln False
```

`use_z0=False, use_zg=False, feed='none'` ⇒ **the actor never sees the latent.**
Its input is `[A, ∇_A V, E]` only, so the representation reaches it exclusively
through the critic's scalar V and that scalar's gradient. On this architecture
the critic is not one lever among several — it is the entire channel.

## 5. Eval — `solver/lip.yaml` exactly as used

```yaml
_target_: stable_worldmodel.solver.LIPSolver
model: ???
batch_size: 1
num_samples: 300      # ignored: only used when n_steps > 0 (MPPI refinement)
var_scale: 1.0
n_steps: 0            # 0 = pure learned planner, no MPPI
topk: 30
lam: 1.0
actor_path: ???       # the trained LIPv4 checkpoint
restarts: 1
restart_noise: 0.5
robust_m: 0
device: "cuda"
seed: ${seed}
```

`plan_config` from `config/cube.yaml`: `horizon: 5`, `receding_horizon: 5`,
`action_block: 5`. With `eval_budget: 50` that is `5×5 = 25` primitive steps per
decision ⇒ **2 decisions per episode**.

```bash
python3 scripts/plan/eval_wm.py --config-name cube seed=42 \
  eval.dataset_name=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
  ++bf16=true eval.img_size=224 \
  eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=8000:10000" \
  policy=/workspace/models/v2WM \
  solver=lip solver.actor_path=lip4_lewm_exp30_s0.pt
```

Repeat over draws 42/43/44 × seeds 0/1/2 and average.

**Cost at these settings:** 8 refinement iterations × (1 grad rollout + 1 no-grad
rollout) + 1 selection rollout = **17 forward rollouts and 8 backward passes**
per decision ≈ 33 forward-equivalents. `restarts=1`, `lip_select="last"`,
`robust_m=0` means no multiplier on the selection block; raising any of them
scales it by `R × C × m`.

---

## Two caveats that belong with the number

**The 192-d latent is not a held-out representation.** The caches and critic are
filtered to episodes `0:8000`, and the eval draws `8000:10000` — but `v2WM`
itself was pretrained on all 10,000 episodes. The encoder has seen the eval
frames. 90.00 is therefore an upper bound with respect to WM pretraining, even
though the planner and critic are clean. See `DATA_SPLIT_POLICY.md`.

**`iters 8` is inherited, not tuned.** K=8 was never swept at this
configuration. The only `iters` data point is K=16 = 87.8, an exact null, and it
comes from 2026-07-29 against the old pre-bundle baseline (freeze 0.8, gamma 1.0,
amax 3.5) — it says nothing about K here.

The pod's `solver/lip.yaml` is older than the repo's, which adds
`rollout_compat`, `plan_scale`, `plan_clip` and `use_frame_history`. The code
defaults (`rollout_compat=True`, `plan_scale=1.0`, `plan_clip=None`,
`use_frame_history=False`) are what the 90.00 run actually used, so the repo yaml
reproduces it — but pin them explicitly rather than trusting defaults to hold.
