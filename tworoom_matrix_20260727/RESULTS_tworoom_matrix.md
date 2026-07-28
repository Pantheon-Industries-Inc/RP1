# TwoRoom planner × cost matrix — results and full experimental configuration

Campaign date 2026-07-27. Pod: 4×H100 80GB, 224 CPU, 2 TB RAM.
Branch `eval-sweep`. All numbers produced by `scripts/plan/tworoom_matrix.sh`
and `scripts/plan/tworoom_phase1_matrix.sh`.

---

## 1. Environment

`swm/TwoRoom-v1` (`stable_worldmodel/envs/two_room/env.py`). 2-D navigation:
a circular agent must reach a target in the opposite room, through one of up to
three doorways in a dividing wall.

| property | value |
|---|---|
| observation | 224×224 RGB, **pure PyTorch CPU renderer** (no MuJoCo — `MUJOCO_GL` is irrelevant here) |
| action | `Box(-1, 1, shape=(2,))`, 2-D velocity direction |
| max speed | 10.5 px/step |
| wall | vertical, centre x = 112; `door.number ∈ {1,2,3}`, positions randomized per episode |
| success | agent within **16 px** of target |
| episode length | ≤ 100 primitive steps |
| control rate | 10 Hz |
| randomized per reset | `agent.position`, `target.position` (target constrained to the opposite room) |

---

## 2. Dataset

**HuggingFace `quentinll/lewm-tworooms` → `tworoom.tar.zst` → `tworoom.h5`** (3.43 GB
compressed, 12.8 GB extracted). Note the *plural* repo name. This is the dataset the
released checkpoints were trained on (`scripts/train/config/data/tworoom.yaml`) and the
one the authors' plan config evaluates against (`scripts/plan/config/tworoom.yaml`).

| column | shape | dtype |
|---|---|---|
| `pixels` | (920809, 224, 224, 3) | uint8 |
| `action` | (920809, 2) | float32 |
| `proprio` | (920809, 2) | float32 — agent (x, y) |
| `pos_agent` | (920809, 2) | float32 |
| `observation` | (920809, 10) | float64 — [agent xy, target xy, 3 × door xy] |
| index cols | `ep_idx`, `step_idx`, `ep_offset`, `ep_len` | |

**920,809 frames / 10,000 episodes** (~92 frames per episode). Collected with the
env's built-in weak `ExpertPolicy`.

**The dataset is also the eval set.** Evaluation replays dataset rows: start = row
*t* (env `_set_state` from `pos_agent`), goal = row *t + offset* (`_set_goal_state`
from `goal_pos_agent`). `dataset.stats` is the source of the action z-scoring and is
pinned to the same file, matching the convention the frozen predictors were trained
under.

**Action convention throughout:** 10-d z-scored action *blocks* = 2-d action ×
frameskip 5.

---

## 3. World models (all author-released checkpoints, frozen)

Converted from the release archives by `scripts/plan/convert_tworoom_bases.py`.
Encoder parity against the reference implementation: max abs diff **2.4–2.8 × 10⁻⁵**.

### 3.1 LeWM (`lejepa_tworoom`)

| component | configuration |
|---|---|
| encoder | `vit_hf` ViT-**tiny**, patch 14, image 224, trained from scratch (`pretrained: False`) |
| latent | **192-d CLS token** (single token) |
| predictor | adaLN `Predictor`, num_frames 3, dim 192→192→192, depth 6, heads 16, mlp_dim 2048, dim_head 64, dropout 0.1 |
| action encoder | `Embedder` 10 → 192 |
| projector / pred_proj | MLP 192 → 2048 → 192, `BatchNorm1d` |
| training objective | ℒ_pred + λ·ℒ_SIGReg (LeJEPA-style) |
| proprio | **not used** — `encode()` consumes pixels only |

### 3.2 PLDM (`pldm_tworoom`)

**Architecturally identical to LeWM** (same encoder, predictor, embedder, projector
specs — verified field-by-field). Differs only in training objective:
ℒ_sim + α·ℒ_std + β·ℒ_cov + δ·ℒ_temp + ω·ℒ_idm (VCReg + temporal + inverse-dynamics).
Latent = 192-d CLS. **Proprio not used.**

### 3.3 DINO-WM (`dinowm_tworoom`) — the proprio variant

This is the variant the LeWM paper's DINO-WM row uses: the repo's default trainer
config `scripts/train/config/prejepa.yaml` carries `wm.encoding.proprio: 10`, and the
no-proprio config is explicitly named `prejepa_action_only.yaml`.

| component | configuration |
|---|---|
| backbone | **frozen** `dinov2_small`, weights ≡ stock `facebook/dinov2-small` |
| image | resized 224 → **196**, patch 14 → **196 patch tokens** × 384-d, CLS dropped |
| token | [pix 384 ‖ proprio-emb 10] = 394; predictor dim **404** (+ action-emb 10) |
| latent | **196 × 394 = 77,224-d flat** (all patch tokens, no pooling) |
| predictor | `CausalPredictor`, num_patches 196, num_frames 3, depth 6, heads 16, mlp_dim 2048, dim_head 64, dropout 0.0 |
| action encoder | `Embedder` 10 → 10 |
| proprio encoder | `Embedder` 2 → 10 |
| history | 3 frames (588 tokens per forward) |
| `cost_chunk` | 1024 (candidate chunking; numerically neutral — verified 128 vs 1024 identical) |

### 3.4 Open-loop fidelity of the conversions

1-step / 4-step predictor MSE relative to a copy-last baseline, on the canonical data:

| base | 1-step pred | / copy-last | / shuffled-action | 4-step / copy-last |
|---|---|---|---|---|
| lejepa | 0.0649 | **0.043** | 0.038 | 0.077 |
| pldm | 0.2444 | **0.168** | 0.130 | 0.216 |
| dinowm | 0.0329 | **0.102** | 0.076 | 0.072 |

Shuffled-action ratios below the copy-last ratios on all three ⇒ genuine action
conditioning.

---

## 4. Evaluation protocol

| setting | value |
|---|---|
| episodes per cell | **n = 50** |
| task seeds | **42, 43, 44** (seed drives the task draw) |
| **h25** | `goal_offset_steps=25`, `eval_budget=50` — **the paper's protocol** |
| **h50** | `goal_offset_steps=50`, `eval_budget=100` — our extension |
| surface `std` | any start/goal pair |
| surface `hard` | `+eval.cross_wall=true` — start and goal on opposite sides of the wall (requires threading a door); **our extension** |
| plan config | `horizon 5`, `receding_horizon 5`, `action_block 5` (= frameskip) |
| `solver.batch_size` | 10 environments per solver call (1 for DINO+Adam, memory-bound; numerically neutral chunking) |
| `max_episode_steps` | 2 × `eval_budget` |
| config | `scripts/plan/config/tworoom.yaml` (authors'), state keys `pos_agent` / `goal_pos_agent`, `keys_to_cache: [action, proprio]` |

Reference values from `docs/baselines.md` (TwoRoom column, h25): **LeWM 87, PLDM 97,
DINO-WM 100.**

**Controls** (same protocol, `std × h25 × seed 42`): `nomove` **8.0**, `random` **28.0**.

---

## 5. Planners

### 5.1 CEM — `solver/cem.yaml`
```
num_samples 300 · n_steps 30 (iterations) · topk 30 · var_scale 1.0
```
Candidate rollouts per plan step: 300 × 30 = **9,000**.

### 5.2 MPPI — `solver/mppi.yaml`
```
num_samples 300 · n_steps 30 · topk 30 · temperature 0.5 · var_scale 1.0
```
**Caveat: untuned.** The repository shipped no MPPI plan config; this one was authored
for this campaign by mirroring `cem.yaml` and taking the class-default temperature.
The temperature — MPPI's most sensitive parameter — has never been swept. Treat MPPI
*rows* as a lower bound; the latent→TD *contrast within* a row is valid, since both
arms share the identical config.

### 5.3 Adam (`GradientSolver`) — `solver/adam.yaml`
```
num_samples 100 · n_steps 30 (gradient steps) · AdamW lr 0.1 · action_noise 0
```
Backpropagates through the world-model rollout. Memory scales with
`batch_size × num_samples` retained graphs — on DINO this needs `batch_size=1`
(100 candidates ≈ 50 GB; 200 candidates OOMs an 80 GB card at 78.5 GB).

### 5.4 LIPv4 (learned amortized planner) — `solver/lip.yaml`, `--arch v4`
`PlannerNetV3`: a transformer refiner over the H plan blocks. Token *t* =
[A_t, ∇_{A_t}V, proj(z_t), proj(z_g − z_t), E, pos_t] + iteration embedding;
each token emits its own update ΔA_t and gate. Residual + clip contract
A_{k+1} = clip(A_k + gate·ΔA).

| setting | value |
|---|---|
| refinement iterations K | **8** (twins); **4** and **8** both run for DINO |
| `amax` (action clip) | **2.2** (canonical TwoRoom); 2.8 selected for a PLDM sweep |
| `max_delta` | 12 |
| horizon | 5 |
| width / layers | 256 / 2, heads 4, mlp 2×width, `zproj` 64 |
| `goal_mode` | `diff` · `head_mode` `gate` · `iter_mode` `emb` · `cond_mode` `token` |
| deploy | `restarts 1`, `robust_m 0`, `n_steps 0` — **pure learned planner, no sampling** |

Candidate rollouts per plan step: 1 × K=8 = **8** (vs CEM's 9,000).

---

## 6. Learned cost — TD quasimetric

`scripts/plan/train_metric.py --learner td`, implementation
`stable_worldmodel/trm/learners/td.py`.

### 6.1 Head — MRN quasimetric (Liu et al., 2022)

d(z_i → z_j) = ‖u(z_i) − u(z_j)‖₂ + max_k ReLU( v(z_j)_k − v(z_i)_k )

- one shared encoder: latent_dim → 256 (SiLU) → 256 (SiLU) → **128**, depth 2
- u and v are the **two halves of that single 128-d embedding**, split by
  `sym_frac = 0.5` (64 symmetric / 64 asymmetric). **`sym_frac` is not exposed by
  `TDConfig` and has been hardcoded at 0.5 in every run.**
- non-negative by construction; satisfies the triangle inequality ⇒ long distances
  *stitch* from short transitions.

### 6.2 Objective — n-step distance TD with HER

```
reached within n_eff steps  →  target = δ                                (Monte-Carlo)
else                        →  target = c(n_eff) + γ^{n_eff}·d_target(z_{t+n}, z_g)
loss = expectile_Huber( d(z_t, z_g) − stop_grad(target) ),  β = 1.0
```

| setting | value |
|---|---|
| `n_step` | **50** (primitive steps) |
| `gamma` | 1.0 (undiscounted true steps-to-go) |
| `expectile` | **0.1** — deliberately LOW: for cost-to-go you want optimism toward the *min* |
| `p_cross` | 0.3 (cross-episode hindsight goals, for stitching) |
| `balanced` | true (balanced full-horizon hindsight goals) |
| steps / batch | 6000 / 1024 |
| optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| target network | Polyak τ = 0.005 |
| training seeds | **0, 1, 2** (3 initializations) |

### 6.3 Known structural gap

`samplers.py` draws `z_t`, `z_tn`, `z_g` **all from the encoder cache**, but at plan
time the cost hook evaluates `cost(predictor_output, encoder_goal)`. The value is
therefore trained enc→enc and deployed pred→enc. Harmless when the predictor's output
lies in encoder space; on bases where it does not, the learned value degrades sharply.

---

## 7. LIPv4 training

`scripts/plan/train_lip_ac.py`

| setting | value |
|---|---|
| critic init | warm-started from the TD checkpoint **of the same seed** (`--init-value`) |
| steps | 8000 |
| batch | 128 (twins); **16** for DINO K=4, **8** for DINO K=8 (memory) |
| expectile | 0.1 → **0.03** (annealed) |
| critic lr | 1e-3 → 1e-4 |
| actor lr | 3e-4 → 3e-5 |
| `n_step` | 50 |
| training seeds | **0, 1, 2** (twins); **0** only for DINO |
| wall-clock | ~58 min/seed (twins, 192-d); ~4 h/seed (DINO, 77,224-d) |

## 8. Latent caches

| cache | stride | purpose |
|---|---|---|
| **fs1** | every frame | TD training — dense so n-step targets see every step |
| **fs5** | frameskip 5 (one latent per action block) | actor rollout contexts |

- twins: 920,809 × 192 × 4 B = **696 MB**
- DINO: full width × all frames would be 920,809 × 77,224 × 4 B = **284 GB**, so the
  cache is **capped at 200,000 rows** (`--max-rows`) = **58 GB**, at full latent width.
  The cap takes a contiguous prefix ⇒ ~episodes 0–2,170 only.
  (A `--compress rp1024` route also exists and works for TD, but *not* for LIPv4:
  `PlannerNetV3` bakes `z_dim` into `zp`/`gp`, so a 1024-d actor cannot consume the
  77,224-d latents the solver feeds at deploy. The critic survives compression because
  `CompressedMetric` dispatches on the last dim.)

---

## 9. RESULTS — std surface

n = 50 per cell. Cell counts: **n=3** = 3 task seeds; **n=9** = 3 task seeds × 3
training seeds.

### 9.1 Paper replication (std × h25, the published protocol)

| base | ours | paper | Δ |
|---|---|---|---|
| PLDM | **96.7** | 97 | −0.3 |
| LeWM (lejepa) | **89.3** | 87 | +2.3 |
| DINO-WM | **100.0** | 100 | 0.0 |

### 9.2 Full matrix

| arm | LeWM h25 | LeWM h50 | PLDM h25 | PLDM h50 | DINO h25 | DINO h50 |
|---|---|---|---|---|---|---|
| **LIPv4** | **100.0** ⁹ | **100.0** ⁹ | 97.1 ⁹ | 98.9 ⁹ | 99.3 ¹ | 99.3 ¹ |
| **TD + CEM** | **100.0** ⁹ | 99.8 ⁹ | 98.7 ⁹ | 99.6 ⁹ | **99.8** ⁹ | **100.0** ⁸ |
| TD + Adam | 96.4 ⁹ | 96.0 ⁹ | 96.0 ⁹ | 95.3 ⁹ | — | — |
| TD + MPPI | 87.1 ⁹ | 87.1 ⁹ | 83.1 ⁹ | 83.8 ⁹ | — | — |
| **Latent + CEM** | 89.3 ³ | **54.7** ³ | 96.7 ³ | **77.3** ³ | **100.0** ³ | **98.0** ³ |
| Latent + Adam | 92.0 ³ | 67.3 ³ | 92.0 ³ | 72.0 ³ | — ᵃ | — ᵃ |
| Latent + MPPI | 65.3 ³ | 47.3 ³ | 71.3 ³ | 55.3 ³ | 92.0 ¹ᶜ | 94.0 ¹ᶜ |

¹ 1 actor seed. ⁸ 8 of 9 cells. ¹ᶜ 1 cell only (partial). — not run (CEM-only scope for DINO).

ᵃ **DINO + Adam is out of scope and was not run.** Adam backpropagates through the
world-model rollout, retaining `batch_size × num_samples` graphs. On DINO's 588-token
predictor that is ~50 GB for 100 candidates and **78.5 GB for 200** — i.e. it OOMs an
80 GB H100 at any `solver.batch_size ≥ 2`, and `expandable_segments` cannot help
because this is live graph, not fragmentation. At `batch_size = 1` it fits (50 GB) but
costs ~2–3 h/cell (~9 h for the arm), and Adam ranked *below* CEM on both twins, so the
arm was dropped rather than paid for. CEM and MPPI are unaffected — both are
inference-only (~25 GB).

DINO LIPv4 K-ablation (6 std cells each, 1 actor seed):
**K=4 → h25 98.7 / h50 100.0** · **K=8 → h25 99.3 / h50 99.3** — identical 6-cell mean
of 99.3. Halving the refinement depth cost nothing.

---

## 10. Findings

### 10.1 A learned cost compensates for a weak latent — it is not universally better

The h25→h50 degradation of the **native latent-MSE cost**, ordered by latent capacity:

| base | latent | Latent+CEM h25 → h50 | drop | TD+CEM h50 | **TD gain at h50** |
|---|---|---|---|---|---|
| LeWM | 1 CLS token, 192-d | 89.3 → 54.7 | **−34.6** | 99.8 | **+45.1** |
| PLDM | 1 CLS token, 192-d | 96.7 → 77.3 | **−19.4** | 99.6 | **+22.3** |
| DINO-WM | 196 patch tokens, 77,224-d | 100.0 → 98.0 | **−2.0** | 100.0 | **+2.0** |

Latent-MSE measures raw feature-space distance, which stops tracking reachability as
the goal recedes; a temporal-distance quasimetric keeps ordering states by steps-to-go.
The size of that correction is inversely proportional to how much geometry the latent
already carries. **This is invisible under the paper's single h25 cell**, where all
three bases read 89–100.

Qualification: DINO's latent cost is near-perfect under CEM (100.0/98.0) but only
92–94 under MPPI, so the advantage is not purely representational — the optimizer
still matters.

### 10.2 LIPv4 matches the best sampling planner at ~1000× less search

| | rollouts / plan step | LeWM h25/h50 | PLDM h25/h50 | DINO h25/h50 |
|---|---|---|---|---|
| CEM | 9,000 | 100.0 / 99.8 | 98.7 / 99.6 | 99.8 / 100.0 |
| **LIPv4** | **8** | **100.0 / 100.0** | 97.1 / 98.9 | 99.3 / 99.3 |

On LeWM, LIPv4 is the single best arm and produced **two perfect 12-cell cards**
(600/600 episodes each, including the cross-wall surface). Wall-clock at deploy is
~34 s/cell vs ~49 s/cell for CEM — a smaller ratio than the FLOP saving because the
pipeline is CPU/render-bound.

### 10.3 Solver ranking is stable; the cost matters more than the optimizer

CEM > Adam > MPPI under both costs on both twins. Swapping latent→TD is worth +13 to
+45; the best-to-worst spread *among solvers at fixed cost* is smaller.

### 10.4 Where PLDM's residual failures live

Per-episode analysis, PLDM std × h25 × seed 42 (same 50 tasks):

| arm | score | failed episodes |
|---|---|---|
| Latent + CEM | 47/50 | 5, 31, 38 |
| TD + CEM | 49/50 | **15** |
| LIPv4 | 48/50 | **15**, 22 |

Episode 15 is failed by *both* learned-cost arms and **solved by plain latent-MSE** — a
value blind spot, not a planner defect. Episode 22 is LIP's only planner-specific loss.
The union of solved tasks across methods covers all 50, so 100% is not blocked by an
unsolvable task; it requires two independent fixes.

A held-out (amax, K) sweep for PLDM LIPv4 — selected on seed 42 only, per §11.1 —
gave: `a22k8` 94.0/100.0 (baseline), **`a28k8` 96.0/100.0 (selected)**, `a35k8`
96.0/98.0, `a35k12` 96.0/96.0, `a22k12` 94.0/100.0. No arm reaches 100 at h25.

---

## 11. Caveats and threats to validity

### 11.1 No train/eval split — every TD and LIP number is an upper bound

TD critics and LIPv4 actors are trained on latents cached from the **same 10,000
episodes the evaluation draws its tasks from**. Different seeds select different tasks,
but every eval state lies in the training pool. These are **in-distribution fit, not
generalization estimates**.

Calibration: memorization is implausible (≈150 eval tasks among ~2M valid (state, goal)
pairs, against a small MLP head), and the measured effects (+22 to +45) are an order of
magnitude above the ~3 points of optimism attributable to selection at n=150. But no
held-out card has been run. `eval.ep_range=lo:hi` now exists for exactly this, and
DINO's 200k-row cache touches only episodes 0–2,170, leaving 8,000–9,999 clean for a
held-out DINO card at near-zero cost.

The PLDM hyperparameter sweep above **was** selected on a single draw (seed 42) and is
intended to be reported on 43/44 only.

### 11.2 Other caveats

- **MPPI is untuned** (§5.2) — its absolute rows are a lower bound.
- **DINO LIPv4 = 1 actor seed**; the twins get 3. LeWM's LIP varied 99.8/100.0/100.0
  across seeds, so single-seed DINO numbers carry more variance than the table suggests.
- **DINO Adam not run** (memory-infeasible at comparable settings, see ᵃ above); DINO
  MPPI partial. CEM-only scope was chosen for DINO because its cells cost ~34 min each.
- **`hard` surface and h50 are our extensions** with no counterpart in the published
  table; they are never compared to 87/97/100.
- **`sym_frac = 0.5`** on the MRN head is hardcoded and has never been varied.
- Near-ceiling saturation: with several arms at 99–100, the card no longer
  discriminates and differences of ≤2 points are within binary noise at n=50 (SE ≈ 2.8).

---

## 12. Defects that had to be fixed to obtain these numbers

Relevant to reproduction — none of these reproduce on the play-data protocol, and
three were silent (produced plausible wrong numbers rather than crashing).

| # | defect | consequence |
|---|---|---|
| 1 | **proprio double-normalized** — `eval_wm.py` fits a `StandardScaler` for every `keys_to_cache` entry and `policy.py` applies it; `DinoWMTokens.encode()` then applies its own `pro_mu`/`pro_std`, fit from the same data (both μ=[111.80, 84.99], σ=[36.87, 38.19]) | positions 2 px apart arrived identical to 3 decimals; **DINO 36.0 → 100.0** |
| 2 | metric hook called `metric.cost()`, decorated `@torch.no_grad()`; `GradientSolver` asserts `costs.requires_grad` | **every TD+Adam cell** would have died (108 cells) |
| 3 | eval **log path lacked the base name** while the score is parsed back out of that log | concurrent bases read each other's scores — two checkpoints with 4× different fidelity returned byte-identical 94.0/88.0/88.0 |
| 4 | `cross_wall` hardcoded a `state` column, absent from the canonical h5 (`pos_agent`) | **6 of every 12 cells** failed silently; cards still printed a mean, over n=6 |
| 5 | converter hardcoded `episode_idx` | raises on the canonical h5's `ep_idx` |
| 6 | converter applied ViT renames unconditionally (they target transformers 5.x; 4.x uses the old layout) | all 192 encoder keys rejected |
| 7 | converter extraction ignored `--only` | converting 3 bases required the 4th archive present |
| 8 | cache compressor not recorded in the TD `arch` | `load_metric` could not rebuild the wrapper |

`reacher_matrix.sh` carries defect #3 identically; any reacher matrix rows produced by
running two bases concurrently should be re-run.

**Not a factor here:** the `MUJOCO_GL=osmesa` domain bug that cost the reacher campaign
7.3 points cannot affect TwoRoom — the env is a pure PyTorch renderer with no MuJoCo
dependency.
