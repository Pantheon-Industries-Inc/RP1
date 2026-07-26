# Dyna / RLP on OGBench cube-single — complete hyperparameter record

Everything needed to reproduce the reported numbers. Values transcribed from the
actual configs and launch scripts on the pod, not from notes.

**Pod:** `157.66.254.11:15788`, 4× H100. Repo `/workspace/code/stable-worldmodel`.
Harness scripts mirrored in `Dyna/dyna_harness/`.

---

## 0. Pipeline overview

```
v2WM (frozen encoder+predictor)
  └─ latent cache (fs1 stride-1, fs5 stride-5)
       └─ TD teacher  V(z, z_g)      [train_metric.py]
            └─ LIPv4 actor-critic     [train_lip_ac.py]      ← "RLP" / pre-Dyna
                 └─ on-policy collection in the real sim      ← Dyna round 1
                      └─ WM fine-tune (expert ⊕ on-policy)  = WM₁
                           └─ fresh cache → fresh TD → fresh LIPv4  ← post-Dyna
```
Every round retrains cache, TD teacher and actor from scratch: the fine-tune
moves the latent space, so nothing downstream transfers.

---

## 1. World model — v2WM (WM₀), used as-is, never retrained

In-house from-scratch LeWM replication (wandb `pantheoninc/sambhav`, epoch 22),
trained on HF `quentinll/lewm-cube` — the same 10k-episode / 224px expert set our
eval draws from (train == eval dataset by design).

| | |
|---|---|
| encoder | ViT-tiny from scratch (`vit_hf`), hidden 192, 12 layers, 3 heads, patch 14, image 224, **CLS** readout → latent dim 192 |
| predictor | AdaLN `Predictor`, num_frames 3, dims 192→192, depth 6, heads 16, mlp_dim 2048, dim_head 64, dropout 0.1 |
| action encoder | `Embedder`, input 25 (= frameskip 5 × action_dim 5) → 192 |
| projector / pred_projector | MLP(192, 2048, 192), BatchNorm1d |
| loss | MSE(next-latent) + **0.09** × SIGReg (knots 17, num_proj 1024) |
| windowing | history 3, num_preds **1**, frameskip 5, span 20, obs_offsets [0,5,10,15] |
| optimizer | AdamW, lr **5e-5**, wd 1e-3, batch **128**, **22 epochs** (~313k steps) |
| schedule | cosine→0, linear warmup 1% of steps; grad-clip 1.0; GradGuard (skip non-finite) |
| precision | bf16 autocast, fp32 params |

**Action normalization (critical, must match at eval):** NaN→0, then per-dim z-score
with dataset-wide constants
`mean = [0.010831, −0.003126, 0.002633, 0.000422, 0.15846]`,
`std = [0.2887, 0.392736, 0.641535, 0.391823, 0.249935]`.

**Checkpoint note:** the archived v2WM is in transformers-5.x ViT key layout; pods run
4.49.0, so keys are converted new→old (inverse of `convert_tworoom_bases.py::VIT_RENAMES`).
Anchor after conversion: latent+CEM s43 = **88.0** vs historical 84 ✓.

---

## 2. Latent caches

`scripts/trm/cache_latents.py`, encoder frozen, one latent **per frame** (`emb[:,0]`).

| | |
|---|---|
| dataset | `ogb_cube_single.lance` (= quentinll expert, 10k eps × 201 steps, 224px) |
| fs1 | stride 1 → 2,010,000 latents (dim 192); used for the critic/TD |
| fs5 | `subsample_cache.py --frameskip 5` → 41 rows/episode; used for the actor |
| state key | `privileged_block_0_pos` |

---

## 3. TD teacher — `scripts/plan/train_metric.py`

```
--learner td --head quasimetric --expectile 0.03 --n-step 50 --gamma 1.0
--steps 6000 --seed 0 --batch-size 1024
```
| | |
|---|---|
| head | **MRN** quasimetric: `d = ‖u(zᵢ)−u(zⱼ)‖₂ + maxₖ ReLU(v(zⱼ)ₖ−v(zᵢ)ₖ)`, hidden 256, embed 128, depth 2, sym_frac 0.5 |
| target | n-step distance TD with **HER** hindsight goals; reached-within-n ⇒ exact MC label δ, else `c(n_eff) + γ^{n_eff}·d_target` |
| goal sampling | `NStepGoalSampler`, balanced buckets over the **full** episode horizon, `p_cross 0.3` (cross-episode stitching goals) |
| expectile | **0.03** (low = optimistic toward the min; campaign rule "low tau for cost-to-go") |
| target net | EMA, tau 0.005 · optimizer AdamW lr 1e-3, wd 1e-4 · Huber beta 1.0 |

---

## 4. LIPv4 actor-critic (RLP) — `scripts/plan/train_lip_ac.py`

```
--horizon 5 --iters 8 --steps 6000 --n-step 50 --amax 3.5
--expectile 0.1 --expectile-final 0.03
--critic-lr 1e-3 --critic-lr-final 1e-4
--actor-lr 3e-4 --actor-lr-final 3e-5
--arch v4 --seed {0..8}
--cache <fs5> --cache-td <fs1> --h5 expert_actions.h5 --wm <WM> --init-value <TD>
```
| | |
|---|---|
| arch | **v4** = gate-free min0 (campaign default; clip substitutes for the gate) |
| plan | horizon 5 action *blocks* × frameskip 5 = **25 env steps**; 8 inner optimization iters |
| amax | 3.5 (action clip) |
| annealing | expectile 0.1→0.03, critic lr 1e-3→1e-4, actor lr 3e-4→3e-5 |
| tandem | critic is **co-trained** during actor training (`c_opt.step()`), with a frozen EMA `teacher` copy the actor's gradient flows *through*, never into |
| actor goal sampling | `d ~ U{1..max_delta}` in **fs5 units** (×5 = env steps), `p_cross 0.3`; **`--max-delta` default 10 ⇒ goals only 5–50 env steps** (see §8) |
| actor input | 3-frame latent history `zh = [z_{t−2}, z_{t−1}, z_t]`; **critic sees single frames** `V(z_t, z_g)` |
| trust region | `--bc-weight 0` (off) — `aref` reference blocks sampled but unused |

Called "LIP"/"LIPv4" in code and artifacts; "RLP" (Reinforcement Learned Planning) in writeups.

---

## 5. Dyna round 1

**Collection** (`SWM_RECORD_PATH` hook in `World._evaluate_from_dataset`; records
pixels/action/qpos/qvel through the *eval* path, because the actor needs goals in
`infos` which `World.collect` doesn't set):
| | |
|---|---|
| policy | the 3 pre-Dyna LIPv4 actors, rolled out in the real sim |
| protocol | `goal_offset_steps 25`, `eval_budget 50`, `num_eval 50` envs/call |
| seeds | 1000+ (disjoint from eval draws 42/43/44) |
| volume | 40 calls × 3 actors → **1,743 episodes / 80,746 steps** |
| filtering | episodes < 25 steps dropped (successes terminate early ⇒ the kept data is failure-enriched) |

**WM fine-tune** (`scripts/train/lewm_expert.py`, warm-start via `INIT_WEIGHTS`):
| | |
|---|---|
| init | v2WM `weights_epoch_22.pt` |
| data | expert ⊕ on-policy, **50/50** exposure (on-policy duplicated 25×; the 80/20 arm used 6×) |
| lr | **1e-5** (5× below base) · batch 128 · 1 GPU · `+action_stats_pin=expert` |
| epochs | 2 (50/50 arm) / 3 (80/20 arm); **gate every epoch checkpoint** |
| selection | 50/50 epoch-1 won the gate → `wm1 = dyna_r1_5050b` |
| gate | A/B divergence + old-actor LIP + CEM canary, vs WM₀ refs (div 36.1 / LIP 66 / CEM 82) |

Round 2 used a 60/30/10 expert/r1/r2 **no-duplication** mix (201,949 rows) — reproduced
round 1 but did not exceed it.

---

## 6. Evaluation protocol

```
scripts/plan/eval_wm.py --config-name cube
  seed={42,43,44} eval.dataset_name=ogb_cube_single.lance ++bf16=true
  eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50
  policy=<WM> solver=lip solver.actor_path=<actor>
```
| | |
|---|---|
| env | `swm/OGBCube-v0`, single view, 224×224, `terminate_at_goal: true` |
| tasks | `num_eval` **50** per draw; the eval seed draws the task set (draw spread ≈ 22 pts ⇒ always report the seed) |
| horizon | goal = state 25 steps ahead; budget 50 env steps (long-horizon sweeps use budget = 2 × horizon) |
| success | block within **4 cm** of target |
| plan_config | horizon 5, receding_horizon 5, action_block 5 |
| reported | mean over draws 42/43/44, then mean over seeds |
| **concurrency** | **evals MUST run sequentially.** 3-way parallel → SIGABRT; 2-way silently corrupts results (cost us an 8-pt baseline error). Use `MUJOCO_GL=osmesa` when anything else occupies a GPU — EGL crashes under concurrent training. |

---

## 7. Environment pins

`torch 2.4.1+cu124` · `transformers==4.49.0` (5.x breaks on torch 2.4.1; <4.47 lacks
`TimmWrapperModel`) · `stable-pretraining 0.1.7` · `lightning 2.6.5` · `mujoco 3.10.0` ·
`dm_control 1.0.43` · `ogbench 1.2.1` · `pylance 8.0.0` · `MUJOCO_GL=egl` (or `osmesa`) ·
`OMP_NUM_THREADS=8–16` (uncapped OMP exhausts the cgroup pid quota on crash-restart).

DDP (only needed for WM training): **`NCCL_NVLS_ENABLE=0` and P2P must stay ENABLED** —
NVLS is broken on these pods, and disabling P2P forces SHM which hangs mid-run.

---

## 8. Known deviations / caveats to report honestly

1. **Actor goal range is capped at 50 env steps** (`--max-delta` default 10 in fs5 units)
   while deployment at h100–h200 asks for goals 4× beyond that. Raising it to 36
   (=180 steps, the structural max for 41-row episodes) was tested: h25 unchanged,
   h100 84.7 vs 84.0 = a wash on 3 draws.
2. **The critic sees single frames, the actor sees 3.** Real asymmetry; probed and shown
   *not* to be the cause of the long-horizon failure.
3. **Round-2 / rollout-loss / value-metric variants are ablations, not the headline.**
   The offline multi-step rollout loss was removed by directive and is empirically
   justified (it measured CEM 44 vs 68.7 reference).
4. **Pre-Dyna numbers measured before the solo-eval rule are unreliable** — the stage-1
   v2WM card (a0 68.0, a1 81.3) ran 2-way concurrent and is excluded; clean re-runs give
   86.7, matching the historical 87.6.
