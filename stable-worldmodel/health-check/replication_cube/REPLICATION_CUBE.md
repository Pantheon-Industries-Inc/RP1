# Replication package — LeWM from-scratch on OGBench-Cube, 84.0%

Everything needed to reproduce the 84.0% (50-episode CEM) result, plus the
gotchas that each cost us days. Files in this folder:

| File | What it is |
|---|---|
| `training_config.yaml` | full resolved training + eval config |
| `model_config.json` | instantiable model config saved with the checkpoint |
| `cube1_eval_lewm_scratch_cls.log` | full eval stdout (metrics dict incl. per-episode successes → 84.0) |
| `cube1_eval_random.log` | random-policy baseline, same protocol → 42.0 |
| `cube_fingerprint.json` | dataset fingerprint: rows/episodes, pixel format, action stats |
| `cube_frame_ep0_step0.png`, `cube_frame_ep0_step50.png` | raw dataset frames (exact pixels the model trains on) |

wandb (entity `pantheoninc-pantheon-inc`, project `sambhav_lewm_pusht`):
training `lewm_lewm_scratch_cls_cube1`, hourly probes `probe_cube1_lewm_scratch_cls`.

## The specific fields you asked for

- `eval.dataset_name`: local lance copy of the **training dataset itself** —
  `quentinll/lewm-cube` (HF) converted h5→lance, staged at
  `/root/run/data_cube_dinowm/shard_0.lance`. Eval samples (episode, start)
  pairs from it and initializes the sim from the stored `qpos/qvel`.
- `num_eval`: **50** `goal_offset_steps`: **25** `eval_budget`: **50** `seed`: **42**
- solver (CEM): `num_samples 300 / n_steps 30 / topk 30 / var_scale 1.0 /
  batch_size 1 / seed 42`; plan_config `horizon 5 / receding_horizon 5 /
  action_block 5`; env `swm/OGBCube-v0`, `env_type single`; `++bf16=true`,
  `eval.img_size 224`.

## Dataset

`quentinll/lewm-cube` on HF = the DINO-WM cube dataset. 10,000 episodes /
2,010,000 frames (~201 steps/ep). Pixels **224×224 RGB** (we store JPEG blobs in
lance; the h5 source is uint8 HWC). Actions 5-dim in [-1, 1] (dim 4 range
[-0.637, 0.908]). See `cube_fingerprint.json` for exact per-dim action stats and
the two PNGs to diff against your renders. If your renders don't match the
PNGs (camera pose, arm colors, lighting, shadows), your env version differs —
we used the stable-worldmodel `swm/OGBCube-v0` wrapper (mujoco 3.10, EGL).

## Ranked list of what most likely broke your replication

1. **Action z-scoring contract (this alone took us from 0% → working).**
   `eval_wm.py` fits a sklearn `StandardScaler` on the dataset's action column
   and the planner's `WorldModelPolicy.get_action` calls
   `inverse_transform(action)` on whatever the CEM solver outputs **before
   sending it to the env**. So the model MUST be trained on z-scored actions
   (stats over the full dataset action column). Train on raw actions and the
   planner is off by ~2.5–4× per dim (std ≈ 0.25–0.64) → near-random success.

2. **NaN actions in the dataset.** The canonical data has NaN action rows
   (episode boundaries). We apply `np.nan_to_num(act)` before computing stats
   and training. Note eval's StandardScaler drops NaN rows instead — the
   resulting stats agree to ~1e-4, which is fine.

3. **Silent NaN-poisoning of training.** Rare batches produce non-finite
   gradients on this dataset (nondeterministic, roughly one in tens of
   thousands of steps). `clip_grad_norm_` then scales ALL grads by NaN and Adam
   is dead from that step on — loss curves look plausibly flat-ish for a while,
   so it's easy to miss. Guard: if the grad-norm returned by clip is not
   finite, `opt.zero_grad(); continue` (skip the step). We skipped a handful of
   steps over the whole run.

4. **Train long enough.** 22 epochs over 2.01M frames ≈ 313k steps at batch
   128. Probe curve: ~60% by epoch 3, 70s by epoch 8, low 80s by epoch ~15,
   84 at the end. A 5–10 epoch run lands in the 60s–70s.

5. **Success metric context.** `env_type=single`, success = cube within 4cm of
   target (`terminate_at_goal=True`). The **random baseline is 42%** under this
   protocol — expert episodes have long static-block segments, so many
   (+25-step) goals have the block already at the target. If you're comparing
   absolute numbers, run the random baseline too (log included); the
   model-over-random gap is the meaningful quantity. Also seed 42 fixes which
   50 (episode, start) pairs are evaluated — a different seed draws a different
   goal mix and moves the number a few points.

6. **bf16 consistency.** We train in bf16 autocast and eval with `++bf16=true`
   (model cast to bf16, images bf16). Mixing (bf16-trained, fp32-eval'd) is a
   ~noise-level effect but keep it matched to reproduce exactly.

7. **Architecture details easy to get wrong** (see `model_config.json` /
   `training_config.yaml`): ViT-tiny **patch 14** (not 16), dim 192; predictor
   is AdaLN-conditioned (action embedding modulates each block), depth 6,
   heads 16, mlp 2048, dim_head 64, dropout 0.1; action token per step is the
   **flattened 5×5 chunk** (frameskip × action_dim = 25) through the Embedder;
   history 3, num_preds 1; SIGReg weight 0.09 (knots 17, proj 1024) on the
   trainable encoder's latents; AdamW 5e-5 / wd 1e-3 / cosine + 1% warmup /
   clip 1.0.

## Exact eval command

```bash
python scripts/plan/eval_wm.py --config-name cube \
  policy=lewm_lewm_scratch_cls_cube1/weights_epoch_22.pt \
  eval.dataset_name=<path>/cube_dinowm/shard_0.lance \
  ++bf16=true eval.img_size=224
```

(Reference: DINO-WM 86, paper LeWM 72, GCBC 84 on this benchmark; our 84.0 with
random floor 42.0.)

## Also available on request

- The final checkpoint itself (`weights_epoch_22.pt`, ~120MB, on the Modal
  Volume `lewm-data:final/lewm_lewm_scratch_cls_cube1` + local copy).
- The converted lance dataset or the raw HF download (46GB) — but it's public:
  `huggingface.co/datasets/quentinll/lewm-cube`.
- Hourly probe jsonl (success-rate-vs-step curve) and eval rollout videos.
