# Hyperparameters — TwoRoom planner × cost matrix (2026-07-27)

Every value below was read out of the running configuration or the checkpoint
`config.json`, not from notes. Companion documents: `RESULTS_tworoom_matrix.md`
(results + findings), `MANIFEST.md` (checksums), `code/` (exact scripts).

---

## 1. Environment — `swm/TwoRoom-v1`

| parameter | value |
|---|---|
| observation | 224×224 RGB, **pure PyTorch CPU renderer** (no MuJoCo; `MUJOCO_GL` is inert) |
| action space | `Box(-1, 1, shape=(2,))` — 2-D velocity direction |
| `MAX_SPEED` | 10.5 px/step |
| control rate | 10 Hz |
| `IMG_SIZE` / `BORDER_SIZE` | 224 / 14 |
| `WALL_CENTER` | 112 (vertical wall) |
| `MAX_DOOR` | 3 (count and positions randomized per episode) |
| success criterion | agent within **16 px** of target |
| episode length | ≤ 100 primitive steps |
| randomized per reset | `agent.position`, `target.position` (target in the opposite room) |

---

## 2. Dataset

| property | value |
|---|---|
| source | HuggingFace **`quentinll/lewm-tworooms`** → `tworoom.tar.zst` → `tworoom.h5` |
| size | 3.43 GB compressed / 12.8 GB extracted |
| frames | **920,809** |
| episodes | **10,000** (≈92 frames/episode) |
| collection policy | env's built-in weak `ExpertPolicy` (defaults: `action_noise=0.0`, `action_repeat_prob=0.0`) |
| `pixels` | (920809, 224, 224, 3) uint8 |
| `action` | (920809, 2) float32 |
| `proprio` | (920809, 2) float32 — agent (x, y) |
| `pos_agent` | (920809, 2) float32 — the state key the eval resets from |
| `observation` | (920809, 10) float64 — [agent xy, target xy, 3×door xy] |
| index columns | `ep_idx`, `step_idx`, `ep_offset`, `ep_len` |
| action stats | z-scored from this same file (`dataset.stats`) |
| action representation | 10-d z-scored **blocks** = 2-d action × frameskip 5 |

**Not used:** our own `tworoom_play.lance` (1,000 episodes, `ExpertPolicy(action_noise=2.0,
action_repeat_prob=0.05)`). σ=2.0 exceeds the [−1,1]² action range, so those trajectories
meander and produce a different task distribution. Since evaluation replays dataset rows,
the dataset *is* the eval set and this choice is load-bearing for paper comparability.

---

## 3. World models (author-released, frozen)

### 3.1 LeWM (`lejepa_tworoom`) and PLDM (`pldm_tworoom`)

**Architecturally identical** — verified field-by-field from both `config.json`. They
differ only in training objective.

| component | value |
|---|---|
| encoder | `vit_hf` ViT-**tiny**, patch 14, image 224, `pretrained: False`, `use_mask_token: False` |
| latent | **192-d CLS token** (single token) |
| predictor | adaLN `Predictor`: `num_frames 3`, 192→192→192, `depth 6`, `heads 16`, `mlp_dim 2048`, `dim_head 64`, `dropout 0.1` |
| action encoder | `Embedder` 10 → 192 |
| projector | MLP 192 → 2048 → 192, `BatchNorm1d` |
| pred_proj | MLP 192 → 2048 → 192, `BatchNorm1d` |
| proprio | **not consumed** (`encode()` takes pixels only) |
| objective (LeWM) | ℒ_pred + λ·ℒ_SIGReg |
| objective (PLDM) | ℒ_sim + α·ℒ_std + β·ℒ_cov + δ·ℒ_temp + ω·ℒ_idm |

### 3.2 DINO-WM (`dinowm_tworoom`) — the **proprio** variant

This is the variant the LeWM paper's DINO-WM row uses: the default trainer config
`scripts/train/config/prejepa.yaml` carries `wm.encoding.proprio: 10`, and the
no-proprio config is explicitly named `prejepa_action_only.yaml`.

| component | value |
|---|---|
| backbone | **frozen** `dinov2_small` (≡ stock `facebook/dinov2-small`) |
| `img_size` | **196** (images resized 224→196 internally) |
| patch tokens | **196** × 384-d, CLS dropped, **no pooling** |
| token composition | [pix 384 ‖ proprio-emb 10] = 394 |
| predictor `dim` | **404** (= 394 + action-emb 10) |
| latent | **196 × 394 = 77,224-d flat** |
| predictor | `CausalPredictor`: `num_patches 196`, `num_frames 3`, `depth 6`, `heads 16`, `mlp_dim 2048`, `dim_head 64`, `dropout 0.0`, `emb_dropout 0.0` |
| action encoder | `Embedder` `in_chans 10` → `emb_dim 10` |
| proprio encoder | `Embedder` `in_chans 2` → `emb_dim 10` |
| `history_size` | 3 (⇒ 588 tokens per predictor forward) |
| `interpolate_pos_encoding` | True |
| `cost_chunk` | 1024 (candidate chunking; verified numerically neutral vs 128) |

### 3.3 Conversion validation

Encoder parity vs the reference implementation: **max abs diff 2.4–2.8 × 10⁻⁵**.
Open-loop predictor MSE relative to a copy-last baseline:

| base | 1-step pred | / copy-last | / shuffled-action | 4-step / copy-last |
|---|---|---|---|---|
| lejepa | 0.0649 | 0.043 | 0.038 | 0.077 |
| pldm | 0.2444 | 0.168 | 0.130 | 0.216 |
| dinowm | 0.0329 | 0.102 | 0.076 | 0.072 |

Shuffled-action ratios exceed the copy-last ratios in all cases ⇒ genuine action
conditioning.

---

## 4. Planners

Shared receding-horizon controller for **all** planners:

| parameter | value |
|---|---|
| `plan_config.horizon` | 5 action blocks |
| `plan_config.receding_horizon` | 5 primitive steps (replan interval) |
| `plan_config.action_block` | 5 (= frameskip) |
| `solver.batch_size` | 10 environments per solver call (1 for DINO+Adam; pure chunking, numerically neutral) |
| `world.max_episode_steps` | 2 × `eval_budget` |

### 4.1 CEM — `config/solver/cem.yaml`

| parameter | value |
|---|---|
| `num_samples` | 300 |
| `n_steps` (iterations) | 30 |
| `topk` (elites) | 30 |
| `var_scale` | 1.0 |
| **rollouts / plan step** | **9,000** |

### 4.2 MPPI — `config/solver/mppi.yaml`

| parameter | value |
|---|---|
| `num_samples` | 300 |
| `n_steps` | 30 |
| `topk` | 30 |
| `temperature` | 0.5 |
| `var_scale` | 1.0 |
| **rollouts / plan step** | **9,000** |

⚠ **Untuned.** The repository shipped no plan-time MPPI config; this one was authored
for this campaign by mirroring `cem.yaml` and taking the class-default temperature.
Temperature — the parameter MPPI is most sensitive to — has never been swept. MPPI rows
are a lower bound.

### 4.3 Adam (`GradientSolver`) — `config/solver/adam.yaml`

| parameter | value |
|---|---|
| `num_samples` | 100 |
| `n_steps` (gradient steps) | 30 |
| optimizer | `torch.optim.AdamW`, lr **0.1** |
| `action_noise` | 0 |
| **rollouts / plan step** | **3,000** |

**Memory:** backpropagates through the world-model rollout, retaining
`batch_size × num_samples` graphs. On DINO-WM's 588-token predictor that is ≈50 GB for
100 candidates and **78.5 GB for 200** — it exhausts an 80 GB H100 at any
`solver.batch_size ≥ 2`, and `expandable_segments` does not help because the overflow is
live activation memory, not fragmentation. Not run on DINO-WM for this reason.

### 4.4 LIPv4 — `config/solver/lip.yaml`, `train_lip_ac.py --arch v4`

`PlannerNetV3`: transformer refiner over the H plan blocks. Token *t* =
[A_t, ∇_{A_t}V, proj(z_t), proj(z_g − z_t), E, pos_t] + iteration embedding; each token
emits its own ΔA_t and gate, applied as A_{k+1} = clip(A_k + σ(gate)·ΔA_t).

| parameter | value |
|---|---|
| `--arch` | `v4` (gate-free min0 lineage) |
| refinement iterations **K** | **8** (LeWM, PLDM); 4 and 8 both run for DINO-WM |
| `--amax` (action clip) | **2.2** (canonical); 2.8 selected in a PLDM sweep |
| `--max-delta` | 12 |
| `--horizon` | 5 |
| width / layers / heads | 256 / 2 / 4 |
| feed-forward | 2 × width |
| `zproj` | 64 |
| `goal_mode` | `diff` |
| `head_mode` | `gate` |
| `iter_mode` | `emb` |
| `cond_mode` | `token` |
| **deploy** | `restarts 1`, `robust_m 0`, `n_steps 0` — **pure learned planner, no sampling** |
| **rollouts / plan step** | **8** (= K) |

---

## 5. Learned cost — TD quasimetric

`train_metric.py --learner td --head quasimetric`; implementation
`stable_worldmodel/trm/learners/td.py`, head `trm/head.py`.

### 5.1 Head — MRN quasimetric (Liu et al., 2022)

d(z_i → z_j) = ‖u(z_i) − u(z_j)‖₂ + max_k ReLU( v(z_j)_k − v(z_i)_k )

| parameter | value |
|---|---|
| encoder | latent_dim → 256 (SiLU) → 256 (SiLU) → 128 |
| `hidden_dim` | 256 |
| `depth` | 2 |
| `embed_dim` | **128** |
| `sym_frac` | **0.5** (64 symmetric ‖ 64 asymmetric — *not exposed by `TDConfig`; hardcoded in every run*) |
| properties | non-negative; obeys the triangle inequality ⇒ long distances stitch from short transitions |

### 5.2 Objective — n-step distance TD with HER

```
reached within n_eff steps  →  target = δ                                     (Monte-Carlo)
else                        →  target = c(n_eff) + γ^{n_eff}·d_target(z_{t+n}, z_g)
loss = expectile_Huber( d(z_t, z_g) − stop_grad(target) )
```

| parameter | value |
|---|---|
| `n_step` | **50** (primitive steps) |
| `gamma` | **1.0** (undiscounted true steps-to-go) |
| `expectile` | **0.1** — deliberately LOW: for a cost-to-go you want optimism toward the *min* |
| `huber_beta` | 1.0 |
| `p_cross` | 0.3 (cross-episode hindsight goals, for stitching) |
| `balanced` | true (balanced full-horizon hindsight goals) |
| `n_buckets` | 10 |
| `steps` | 6,000 |
| `batch_size` | 1,024 |
| optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| target network | Polyak `tau` = 0.005 |
| training seeds | **0, 1, 2** |
| wall-clock | ≈90 s/seed (192-d); ≈22 min/seed (77,224-d) |

### 5.3 Known structural gap

`samplers.py` draws `z_t`, `z_tn`, `z_g` **all from the encoder cache**, but the plan-time
cost hook evaluates `cost(predictor_output, encoder_goal)`. The value is trained enc→enc
and deployed pred→enc. Harmless when the predictor's output lies in encoder space;
on bases where it does not, the learned value degrades sharply.

---

## 6. LIPv4 training — `train_lip_ac.py`

| parameter | value |
|---|---|
| critic init | warm-started from the TD checkpoint **of the same seed** (`--init-value`) |
| `--steps` | 8,000 |
| `--batch` | 128 (LeWM, PLDM); **16** for DINO K=4, **8** for DINO K=8 (memory) |
| `--expectile` → `--expectile-final` | 0.1 → **0.03** (annealed) |
| `--critic-lr` → `--critic-lr-final` | 1e-3 → 1e-4 |
| `--actor-lr` → `--actor-lr-final` | 3e-4 → 3e-5 |
| `--n-step` | 50 |
| `--iters` (K) | 8 (twins); 4 and 8 for DINO |
| `--amax` / `--max-delta` | 2.2 / 12 |
| training seeds | **0, 1, 2** (twins); **0** only for DINO-WM |
| wall-clock | ≈58 min/seed (192-d); ≈4 h/seed (77,224-d) |

### 6.1 PLDM action-clip sweep (selection on task seed 42 only)

| arm | amax | K | h25 | h50 |
|---|---|---|---|---|
| `a22k8` (baseline) | 2.2 | 8 | 94.0 | 100.0 |
| **`a28k8` — selected** | **2.8** | **8** | **96.0** | **100.0** |
| `a35k8` | 3.5 | 8 | 96.0 | 98.0 |
| `a35k12` | 3.5 | 12 | 96.0 | 96.0 |
| `a22k12` | 2.2 | 12 | 94.0 | 100.0 |

Selected on seed 42 alone, to be reported on seeds 43/44 only. No arm reaches 100 at h25.

---

## 7. Latent caches

| cache | stride | purpose |
|---|---|---|
| **fs1** | every frame | TD training — dense, so n-step targets see every transition |
| **fs5** | frameskip 5 (one latent per action block) | actor rollout contexts |

| base | fs1 size | note |
|---|---|---|
| lejepa / pldm | **696 MB** each | 920,809 × 192 × 4 B |
| dinowm | **58 GB** | capped at **200,000 rows** (`--max-rows`); full width × all frames would be 920,809 × 77,224 × 4 B = **284 GB**. The cap takes a contiguous prefix ⇒ ≈ episodes 0–2,170 |
| dinowm (alt) | 3.6 GB | `--compress rp1024` random projection — works for TD, **not** for LIPv4 (see below) |

**Why the row cap rather than compression for LIPv4:** `PlannerNetV3` bakes `z_dim` into
`zp`/`gp` as `Linear(z_dim, zproj)`, so an actor trained on a 1024-d compressed cache
shape-mismatches the 77,224-d latents `LIPSolver` feeds at deploy. The *critic* survives
compression because `CompressedMetric` dispatches on the last dim (1024 passes through,
77,224 is projected), but the actor cannot. Capping rows keeps full width so actor,
critic and deploy all agree, with no solver change.

---

## 8. Evaluation protocol

| parameter | value |
|---|---|
| plan config | `scripts/plan/config/tworoom.yaml` (**the authors' config**) |
| state keys | `pos_agent` / `goal_pos_agent` |
| `dataset.keys_to_cache` | `[action, proprio]` — **except DINO-WM: `[action]`** (see §9) |
| episodes per cell | **n = 50** |
| task seeds | **42, 43, 44** (seed drives the task draw) |
| **h25** | `goal_offset_steps=25`, `eval_budget=50` — **the published protocol** |
| **h50** | `goal_offset_steps=50`, `eval_budget=100` — our extension |
| surface `std` | any start/goal pair |
| surface `hard` | `+eval.cross_wall=true` — opposite sides of the wall (our extension) |
| reported cell counts | n=3 (task seeds) for untrained arms; n=9 (3 task × 3 training seeds) for TD/LIP |

Reference values (`docs/baselines.md`, TwoRoom, h25): **LeWM 87, PLDM 97, DINO-WM 100.**

Sanity controls, same protocol (`std × h25 × seed 42`): `nomove` **8.0**, `random` **28.0**.

---

## 9. DINO-WM: the proprio normalization fix

`dataset.keys_to_cache=[action]` for DINO-WM is **not** a protocol change — it removes a
**double normalization**:

1. `eval_wm.py` fits a sklearn `StandardScaler` for every key in `keys_to_cache`, and
   `policy.py` applies it to the info dict before the model is called.
2. `DinoWMTokens.encode()` then applies its own `pro_mu`/`pro_std`, which the converter
   fit from the same dataset.

Both transforms are numerically identical — μ = [111.80, 84.99], σ = [36.87, 38.19] on
the model and [111.80, 84.99] / [36.88, 38.19] in the data — so applying both maps every
agent position to ≈ −3:

| raw proprio | after one pass (correct) | after both (what the model got) |
|---|---|---|
| [169.43, 65.93] | [+1.563, −0.499] | [−2.990, −2.239] |
| [167.25, 60.93] | [+1.504, −0.630] | [−2.992, −2.242] |

Positions 2 px apart arrive identical to three decimals; the goal's position signal is
destroyed and the input sits ≈3σ off-distribution.

**Measured effect** — same checkpoint, seed, solver, dataset: **36.0 → 100.0**
(the paper's reference is 100). The fixed run is also ~1.6× faster (2,063 s vs 3,365 s)
because CEM converges instead of thrashing and episodes terminate on success.

Only DINO-WM reads proprio, so LeWM and PLDM are unaffected and need no override. The
override also aligns the eval path with `cache_latents.py`, which feeds proprio **raw** —
the two previously disagreed.

---

## 10. Software environment

| component | version |
|---|---|
| torch | 2.4.1+cu124 |
| transformers | **4.57.6** (< 5; see below) |
| stable-pretraining | 0.1.8 |
| CUDA driver | 570.124.06 |
| Python | 3.11.10 |
| hardware | 4 × H100 80 GB, 224 CPU, 2 TB RAM |
| install | `pip install -e ".[train,format]"` + `opencv-python-headless` |

**transformers major version matters.** `convert_tworoom_bases.py`'s `VIT_RENAMES` maps
the released twins' old-HF ViT layout onto the transformers-**5.x** layout. Under 4.x the
renames convert the checkpoint *away* from what the model builds and `load_state_dict`
rejects all 192 encoder keys. The converter now gates the renames on the installed major
version, so it works under both.

**Thread cap is load-bearing:** `OMP_NUM_THREADS`/`MKL_NUM_THREADS` are capped (12–16).
Uncapped threads cost ~40× on TwoRoom evals, because the renderer is CPU-bound.
