# Upstream DINO-WM (noprop, cube) on our protocol — CEM done, TD/LIP in progress

**2026-07-30/31 · pod 157.66.255.80:12519 (4×H200), KILLED 2026-07-31 · branch `eval-sweep`**

Everything in this folder is small and sufficient to resume. The only things
NOT preserved are regenerable: the latent caches (26 min to re-encode) and the
`/workspace` copies of the scripts (all here).

---

## 1. What the artifact is

`stable-worldmodel/checkpoints/ogbench-dinowm/dinowm_noprop_cube.tar.zst`
(309 MB, author `lucas.maes`, Mar 14) unpacks to
`cube/dinowm_noprop_{object,weights}.ckpt`, run name
`cube_dinowm_noprop_epoch_10`. It is an upstream **PreJEPA**, NOT the repo's
much simpler `wm.dinowm.DinoWM` (mean-pool + MLP).

No config ships with it. **`ckpt/config.json` here is the reconstructed one** —
verified by a clean state_dict load (0 shape mismatches) plus a forward pass:

| | |
|---|---|
| encoder | frozen `dinov2_small` wrapped in `spt.backbone.utils.EvalOnly` (that wrapper is what produces the `backbone.backbone.*` key nesting) |
| predictor | `CausalPredictor`, depth 6, dim **394 = 384 + 10**, heads 16, mlp 2048 |
| action | `Embedder` Conv1d 25 → 10 |
| proprio | none (hence `noprop`, and 394 not the proprio variant's 404) |
| **resolution** | **196×196** |

**196 px is forced, not chosen**: the predictor's positional embedding is
588 = 3 frames × 196 patches, and DINOv2's patch size is 14. Verified — 196 px
yields exactly 588 tokens; 224 px raises
`size of tensor a (768) must match b (588)`. Every other number in our
campaigns is at 224 px, so this checkpoint is not rendered identically to them.

To use: `policy=<dir with config.json + weights_epoch_10.pt>` and
`eval.img_size=196`. Copy `weights` ckpt in as `weights_epoch_10.pt`.

---

## 2. RESULT: latent + CEM = 80.0

Held-out episodes 8000–9999 (`+eval.ep_range=8000:10000`, driver aborts the
cell unless the log confirms it), h25 (goal t+25, budget 50), 196 px, **fp32**,
`solver=cem` (`n_steps=30`, `num_samples=300`). ~33 min/cell.

| draw | 42 | 43 | 44 | **mean** |
|---|---|---|---|---|
| success rate | 82.0 | 80.0 | 78.0 | **80.0** |

Context under the same protocol: v2WM latent+CEM **75.3**; LeWM paper's quoted
DINO-WM **86**; our own from-scratch noprop DINO-WM retrain **38–52**
(action-blind, see [[dinowm-cube-retrain]]).

**This settles an open question from that campaign**: the ~50 ceiling on our
retrain was a *training* failure, not an eval-harness or convention problem —
the same architecture, trained properly, plans at 80 in our harness.

Caveats to keep attached: (i) this checkpoint's training data is unknown to us,
so the episode split constrains OUR pipeline but not the artifact — 80.0 is not
a clean generalization estimate; (ii) 196 px vs 224 px everywhere else.

---

## 3. Three real bugs found (worth fixing upstream)

1. **`wm/prejepa/__init__.py` does `from .prejepa import *`**, which never
   exports `module.py`'s classes. Any pickled PreJEPA (`*_object.ckpt`) is
   therefore unloadable: `AttributeError: Can't get attribute
   'CausalPredictor'`. Same bug the DINO-WM campaign hit. Worked around by
   using the `weights` state_dict with the reconstructed config.
2. **`eval_wm.py:160` force-sets `model.interpolate_pos_encoding = True`**
   unconditionally. That assumes a ViT encoder that accepts the kwarg; HF
   `Dinov2Model.forward` rejects it (Dinov2 always interpolates internally), so
   every DINOv2-backed WM dies with `TypeError`.
3. **The metric hook is LeWM-shaped.** `_MetricCost.get_cost` reads
   `info_dict['predicted_emb']` / `['goal_emb']` then takes `[..., -1, :]` for
   the last imagined frame. PreJEPA (a) has no `predicted_emb` at all — it uses
   `predicted_embedding` and publishes an already-split pixels-only view
   `predicted_pixels_emb` / `pixels_goal_emb`; and (b) emits
   `(B, N, T, P, D)`, so `[..., -1, :]` silently indexes the last **patch**
   instead of the last **frame**.

All fixes are in **private copies** — the shared scripts other sessions were
running were never touched.

---

## 4. Scripts in this folder

| file | what |
|---|---|
| `eval_wm_dino.py` | `eval_wm.py` + fixes 2 and 3. Pooling gated on `+metric_pool_dim=<D>`, so LeWM behaviour is bit-identical without it. Rebuild with `make_dino_eval2.py`. |
| `train_metric_sweep.py` | `train_metric.py` + `--lr`, `--weight-decay`, `--polyak-tau`. `TDConfig` had all three; the stock CLI passed **none**, so an lr sweep was unreachable. Rebuild with `make_dino_stack.py`. |
| `cache_latents_dino.py` | pooled-latent cache for PreJEPA. Stock `cache_latents.py` returns `emb[:, 0]`, which is a `(196, 394)` token grid for PreJEPA, not the flat vector the TD/LIP stack needs. Image pipeline matches `eval_wm.img_transform` **exactly** (ImageNet-normalise, *then* resize) — getting that order wrong puts cache and planner in different domains. |
| `fit_channel_pca.py` | fits the per-token channel PCA (`ckpt/dino_chan_pca_k32.pt`). |
| `probe_metric.py` | offline TD screen on held-out episodes: spearman / pair_acc / monotone. |
| `probe_reductions.py` | ridge decodability of true cube + effector xyz from each candidate reduction. |
| `dino_td_stage1.sh` | cache → filter → fs5 → canonical TD → probe → 6-task wiring smoke. |
| `dino_td_sweep.sh` | 12-cell expectile × lr sweep, 4-way, then offline screen. |
| `dinowm_cem.sh` | the plain CEM eval that produced 80.0. |
| `dynavid_cmp.sh`, `dynavid_h100.sh`, `export_individual.py`, `pick_tasks.py`, `task_stats.py` | the pre/post-Dyna video work (see `Dyna/videos_pre_post_20260730/`). |
| `vidsetup.sh` | container re-bootstrap; see [[pod-container-rebuild]] for the CUDA-13/cuDNN trap. |

Pipeline (canonical, from `dyna_repair_downstream.sh`):

```
cache_latents → filter_cache_eprange (0:8000) → subsample_cache (fs5)
   → train_metric (TD teacher) → train_lip_ac (LIPv4, warm-started from TD)
```
TD training is **95 s**; a TD+CEM eval is **~33 min**; LIP training ~1 h/actor;
LIP eval ~2 min.

---

## 5. TD stage: what was measured, and the open question

### 5a. Mean-pooled substrate — 12-cell sweep (`results/summary_dino_td.csv`)

Reduction: mean over the 196 patch tokens of `pixels_emb` → 384-d (matching
the repo's own `DinoWM._encode_pixels`, and using `pixels_emb` so the tiled
action embedding never enters a state latent).

| cell | spearman | pair_acc | monotone |
|---|---|---|---|
| **best** expectile 0.1, lr 1e-3 | 0.437 | **0.666** | 0.575 |
| canonical 0.03, lr 1e-3 | 0.413 | 0.653 | 0.638 |
| worst 0.5, lr 3e-4 | 0.374 | 0.633 | 0.539 |
| **v2WM reference** (`rep_full_TD`, the teacher behind LIPv4 ~92) | *0.484* | *0.691* | *0.923* |

Every cell is worse than the v2WM teacher. Tuning expectile/lr moves pair_acc
by 0.04 and never repairs monotonicity. **NB `TDConfig.tau` is the
target-network Polyak rate, NOT the IQL expectile** — the expectile is
`--expectile`. Swept the expectile (as the memory's "low tau" note intends).

### 5b. Decodability of each candidate reduction (ridge, held-out episodes)

Channel-PCA cumulative EVR: k=8 → 0.644, k=16 → 0.757, **k=32 → 0.843**,
k=64 → 0.909, k=128 → 0.958.

| representation | dims | cube xyz R² | effector R² |
|---|---|---|---|
| full grid | 75,264 | **0.976** | 0.996 |
| proj k=32 | 6,272 | 0.957 | 0.993 |
| grid4x4 (selection) | 6,144 | 0.956 | 0.992 |
| proj k=16 | 3,136 | 0.948 | 0.991 |
| proj k=8 | 1,568 | 0.923 | 0.986 |
| mean-pooled | 384 | **0.915** | 0.973 |

### 5c. The open question — READ THIS BEFORE SPENDING ON A BIG POD

**Mean-pooling is not the information bottleneck.** It keeps cube-position
R² 0.915 against the full grid's 0.976 — yet its TD monotonicity is 0.58–0.64
against LeWM's 0.92. A 6-point R² gap cannot explain that. (Spatial position
evidently leaks into channel statistics, since different locations excite
different DINOv2 channels.)

So the likely cause is latent **geometry**, not content: DINOv2's high-variance
directions encode arm pose, lighting and background, so distances in that space
don't track task progress — whereas LeWM's 192-d projection was *trained* under
a predictive objective that shapes the space for dynamics. A quasimetric head
must find task structure in whatever geometry it is handed.

Also note a fixed projection composed with the head's *learned* first linear
layer is mathematically a learned linear map restricted to that subspace — so
once k spans the useful directions, "full tokens" and "PCA-k" converge by
construction. That caps the upside of going to 75,264-d.

---

## 6. Resume plan

**Cheap decisive test first (~35 min on any pod):** build the k=32 and grid4x4
substrates from ONE encoding pass (`fit_channel_pca.py` output is already in
`ckpt/`), re-run `dino_td_sweep.sh` on each, compare probes to §5a.

- if pair_acc/monotone move meaningfully above 0.666 / 0.575 → spatial fidelity
  matters; scale k up (64 → 128 → full) and then run the LIPv4 grid
- if neither moves → the reduction was never the bottleneck. Do **not** buy the
  605 GB run to confirm a null; attack the geometry instead (learn a small
  projection with a temporal/predictive objective, i.e. give the DINO features
  the thing LeWM's trained space already has)

**If going to full tokens anyway**, the pod needs **1 TB local/container disk**
— that is the only gap. fp32 cache is 605 GB (484 train + 121 held-out); RAM
≥600 GB (TD holds the cache in CPU RAM and batches to GPU); 1 GPU suffices for
TD, 4 for the LIP grid. **Do not put the cache on the MFS `/workspace` volume:
measured ~63 MB/s, so 484 GB is ~2 h to write and as long to read, per run.**
Local NVMe is ~1–2 GB/s. Budget ~1.5–2 h to a full-token TD sweep result.

**LIPv4 grid (not started).** Canonical recipe:
```
train_lip_ac.py --cache <fs5> --cache-td <fs1> --h5 expert_actions.h5 \
  --wm <model_dir> --init-value <TD.pt> --horizon 5 --iters 8 --steps 6000 \
  --n-step 50 --amax <A> --expectile 0.1 --expectile-final 0.03 \
  --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 \
  --arch v4 --seed <s>
```
Planned axes: `--amax` {1.2, 1.6, 2.0, 2.5} × `--actor-lr` {1e-4, 3e-4, 1e-3},
canonical critic-lr, 1 seed, screened on 1 draw (LIP eval is cheap, ~2 min),
then best 2–3 cells at 3 seeds × 3 draws. amax 1.6 was optimal on v2WM
(plateau 1.4–2.2) — but that was tuned on 224 px LeWM latents, so re-tune here.
**LIP warm-starts its critic from the TD teacher, so point it at a good one.**

## 7. Ops notes

- fp32 only: `++bf16=true` **crashes** dino models here (CEM feeds fp32 actions
  into a bf16-cast action-Embedder Conv1d).
- `+video_dir=...` needs the leading `+`; without it the eval dumps mp4s into
  the checkpoint folder.
- EGL evals must be strictly sequential; training (no EGL) co-resides fine
  across GPUs. Pin `MUJOCO_EGL_DEVICE_ID`.
- Container rebuild trap (cost ~12 min): installing the swm `env` extras drags
  torch 2.4.1 → 2.13 and numpy → 2.x, and leftover `nvidia-*-cu13` wheels then
  shadow cuDNN (`CUDNN_STATUS_NOT_INITIALIZED`;
  `torch.backends.cudnn.version()` must read **90100**, not 92000). Full recipe
  in [[pod-container-rebuild]].
