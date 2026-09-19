# Hi-LeWM vs flat RLP — artifact audit and campaign plan (2026-09-17)

Goal: settle whether hierarchical planning adds anything **on top of RLP's flat
planner** (stitched quasimetric + learned refiner), using a third party's
trained high-level checkpoints so that "our F² was undertrained" is not a
possible explanation. Supersedes the plan to retrain F² on PLDM.

## Background: why this replaces the PLDM port

Our own hierarchy campaign (`Value_Metric_LeWM/previous/hwm_lewm_20260718/`,
HWM × LIPv4 on the LeWM cube stack) found no hierarchy configuration beating
flat LIPv4 at any horizon: best case a tie (h100 78.7 = 78.7), worst case
43.3 vs 87.6 at h25. Diagnosis: (F1) the flat stitched value already performs
the hierarchy's temporal abstraction; (F2) the high-level actor emits
off-distribution macro-actions, so the "subgoal" is a fiction — measured
`d_LL(z0, wp0)` = 42 env steps where one macro spans 25, NN-distance 5.6 vs
2.3 for real latents.

Two defects in our port were identified after the fact and never tested:
the only constraint on a macro was a loose box clamp `‖a‖∞ ≤ amax = 3`, and
the actor objective `V(z_H, g) + 0.1·mean_k V(z_H^{(k)}, g)` never evaluated
the critic at an intermediate waypoint (`e_path` indexes refinement
iterations, not plan positions). The critic was also never abstracted — it
trained on the dense fs1 cache in env-step units (`train_lip_hl.py`).

An independent group has since published the same experiment, and the fix.

## The artifact

**Paper:** "Mind the Gap: Promises and Pitfalls of Hierarchical Planning in
LeWorldModel", arXiv 2607.12547 (Caselli, Massafra, Punzo, Lo Sardo,
Pantelidis, Bhethanabhotla). Introduces **Hi-LeWM**: freeze the pretrained
low-level LeWM, add high-level planning over latent subgoals. Cells: PushT
and Cube.

**Artifact:** Zenodo 10.5281/zenodo.21353240, `package.zip`,
2,299,863,924 bytes, sha256
`b89046841d679fe70f435540d62ae87b662ec34eb457919b098d152263f63967`
(414 files, 2.5 GB unpacked). Layout: `code/` (implementation, configs,
scripts, tests) + `checkpoints/`.

Their abstract reaches our diagnosis independently — *"unconstrained search
can select latent macro-actions that appear favorable under the learned model
but produce poor control targets"*, and *"the frozen low-level controller can
execute well-aligned intermediate targets, indicating that high-level subgoal
generation is the main bottleneck"* — and then reports the fix working:
constraining search to macro-actions encoded from training trajectories gives
+11.3 pp at medium horizons and +14.7 pp at the longest PushT horizon.

### Their mechanism ("empirical-macro CEM")

Candidates are `ξ = ξ_bank + λ_res · ε`, where `ξ_bank` is drawn uniformly
from a bank of macro-action sequences encoded from training trajectories and
`ε ~ N(μ, Σ)` is a residual perturbation; one zero-residual candidate is
included per CEM iteration. Config (`config/eval/hi_pusht.yaml`):
`empirical_macro: {num_sequences: 4096, chunk_len: 5, residual_scale: 0.1,
min_residual_std: 1e-3, return_top_candidates: 8, stage_sampling: sequence}`.
This is the same fix we had identified but never ran (constrain to the
encoded-chunk manifold instead of the box clamp).

"Staged execution" = high-level CEM runs **once** at episode start, stores an
ordered list of latent targets, and advances the active target after a fixed
stage duration; the low level still replans closed-loop toward the current
target but cannot correct a bad high-level rollout.

### Their published numbers

| cell | flat LeWM+CEM | Hi-LeWM (naive) | Hi-LeWM-C |
|---|---|---|---|
| PushT d25 | 94.0 ± 2.0 | 89.3 | — |
| PushT d50 | 52.7 ± 5.0 | 38.7 | 48.7 online / **64.0** staged |
| PushT d75 | 18.0 ± 2.0 | 15.3 | **32.7** online / 22.0 staged |
| Cube d25 | 65.3 ± 4.2 | — | 67.3 ± 3.4 |
| Cube d50 | 52.0 ± 3.5 | — | 54.7 ± 0.9 |
| Cube d75 | 53.3 ± 7.0 | — | **69.3 ± 6.8** |

3 seeds × 50 episodes per configuration.

## Compatibility with our stack

`h_le_wm/checkpoints.py:17`:

```python
HF_BASELINE_SOURCES = {
    "baseline/pusht/lewm": "https://huggingface.co/quentinll/lewm-pusht",
    "baseline/cube/lewm":  "https://huggingface.co/quentinll/lewm-cube",
}
```

- **PushT: same base as ours.** `configs/eval/pusht_lewm.yaml` already names
  `quentinll/lewm-pusht`. Their high-level therefore sits on the identical
  frozen low level our PushT actors were trained against.
- **Cube: different base.** Theirs is the authors' `quentinll/lewm-cube`; ours
  is the in-house v2WM replication (`assets/core/world_model/lewm_cube/
  weights_epoch_22.pt`, sha256 `870fbe73…`), which is the stronger model
  (~84 vs the authors' ~74). A cube comparison would need their base on both
  sides, i.e. re-running RLP. **⇒ PushT is the cell.**

Their PushT checkpoint (`checkpoints/pusht/main/pusht_hi_lewm_epoch15_weights.ckpt`,
428 tensors) bundles the whole model: `model.encoder` (198), `model.low_predictor`
(81), `model.action_encoder`, `model.projector`, `model.low_pred_proj` — the
frozen LeWM — plus `model.high_predictor` (81), `model.latent_action_encoder`
(30), `model.macro_to_condition` (192×32), `model.high_pred_proj`. Their arm is
self-contained; no HF fetch needed to run it.

High-level architecture, from the weights and `checkpoints/pusht/main/config.yaml`:
`embed_dim 192`, `history_size 3`, `latent_action_dim 32`, waypoints
`{num: 5, strategy: random_sorted, min_stride: 1, max_span: 15, stride: -1,
beta(2,2)}` (variable span, unlike our fixed stride 25), action encoder =
transformer over **10-d** action tokens (LeWM's canonical action width, as in
our `Embedder(input_dim=10)`) × 15 positions + CLS → 32-d, high predictor =
6-layer AdaLN transformer, ff 2048, `adaLN_modulation` 1152 = 6×192 — i.e. our
`Predictor` class. Trained 15 epochs, AdamW lr 5e-5, bf16, `train_split: 0.9`.

## Protocol reconciliation — the 94.0 vs 79.3 gap

Their flat PushT d25 is 94.0; our flat CEM on the same env and base is 79.3
(ours: `docs/campaigns/2026-09-03/PUSHT_DIAG.md`). The axes:

| axis | theirs | ours |
|---|---|---|
| eval episodes | `pusht_expert_train` (**training split**) | held-out `episode_range: "16000:18685"` |
| plan / commit | `low_horizon 5; low_receding_horizon 5; low_action_block 5` | horizon 5, receding 5, block 5 — **identical** |
| CEM budget (flat) | 300 × 30 | 300 × 30 — **identical** |
| budget | `eval_budget 50` = 2× offset | 2× offset — **identical** |
| episodes / seeds | 50, seed 42 (baseline matrix) | 50, draws 42/43/44 |
| env + success test | `swm/PushT-v1` built-in | same |

Source: `h_le_wm/experiments/matrix/pusht/baseline_matrix.csv` row
`D25;50;5;5;5;300;30`.

**So the flat-arm gap is essentially the train/held-out split alone** — a ~15 pt
train-set premium on PushT d25. Independent corroboration of our own
`no-train-eval-split` caveat.

### Second methodological issue: their arms are unpaired

`h_le_wm/eval/hierarchical.py` samples with
`g.choice(len(valid_indices), ...)`; `h_le_wm/eval/baseline_manifest.py` uses
`g.choice(len(valid_indices) - 1, ...)`. At equal seed these return different
draws, so their flat and hierarchical arms were scored on **different**
50-episode task sets and the +11.3 / +14.7 deltas are unpaired.

## The patch

`scripts/hilewm/0001-heldout-and-paired-tasklist.patch` (applies with
`patch -p1` inside `package/code`, dry-run verified, both files compile) adds
a shared selector used by both eval paths:

- `eval.episode_min` / `eval.episode_max` — restrict evaluation starts to an
  episode range (set `episode_min: 16000` to match our held-out protocol).
- `eval.row_indices_file` — newline-separated dataset row indices, overriding
  sampling entirely, so every arm scores the same (episode, start) pairs.

Our harness already logs its own selection ("Selected evaluation row indices:
…", `src/rlp/eval/world_model.py:249`), so the task list of record can be
exported from our side and consumed by theirs.

## Plan

1. ~~**Base identity check.**~~ **DONE 2026-09-18 — IDENTICAL.** Job 25657:
   all 303 frozen-low-level tensors bit-identical (`max_abs = 0.000e+00` in
   every group: encoder 198, predictor 81, action_encoder 6, projector 9,
   pred_proj 9) against raw `quentinll/lewm-pusht` @ `22b330c2`; 122
   high-level tensors skipped, 3 `sigreg.*` unmapped (LeJEPA regularizer
   state, not part of the WM). **Their high level and our PushT actors share
   one frozen base — the head-to-head carries no base-mismatch caveat.**
   That run also validated: artifact download + sha256 gate, patch
   application, dataset wiring, checkpoint staging, and that their code
   imports and runs in our pixi env (python 3.13 / stable-worldmodel 0.1.1) —
   no conda fallback needed.
2. **Reproduce one published cell** as released, no patch: PushT d75 flat 18.0
   → Hi-LeWM-C 32.7 (largest effect). Validates the artifact end-to-end and our
   ability to drive it. Needs their conda env, `setup_paper_datasets.sh`,
   `fetch-baselines`, and the upstream `lucas-maes/le-wm` checkout.
3. **Reconcile the protocol.** Apply the patch; re-run their flat CEM at
   `episode_min: 16000`. Expected: the flat arm falls from 94.0 toward our
   79.3. Confirms the split explains the gap and puts both stacks on one
   protocol. Also re-run their flat-vs-Hi-LeWM-C **paired** on one task list.
4. **The actual test.** Hi-LeWM-C vs flat RLP (`pusht-v2l05-s{0,1,2}`, w4,
   K=8, md20, acr 0.5) on the same held-out task list, d25/d50/d75. Reading:
   Hi-LeWM-C > flat RLP ⇒ F1 falls, hierarchy is live on our stack;
   Hi-LeWM-C ≈ flat RLP ⇒ F1 confirmed and their gain is a statement about
   their flat baseline.

Compute: steps 2–4 need a GPU pod. Their budgets are heavy — high level
900–1500 samples × 20–60 CEM steps plus low level 300–1200 × 30 per decision,
× 50 episodes × 3 seeds × 3 offsets.

## Ops findings (2026-09-18)

- **Resource footprint, not priority, gates admission.** `cpus=20, memory=230`
  (the guide's 1-GPU row) sat PENDING 6 h 31 m at "Position in default-queue:
  6"; the identical job at `cpus=8, memory=90` was admitted immediately and
  ran. 20/230 is the full per-GPU physical share and is sized for training
  dataloaders; under fragmentation it cannot be placed. This yaml therefore
  asks 8/90. The PushT E-series stalls for the same reason (25467 PENDING
  14 h at 20/230). See `[[kueue-eval-jobs-small-footprint]]`.
- **Their released environment spec no longer reproduces their environment.**
  `code/environment-gpu.yml` pins nothing beyond `python=3.10` and
  `stable-worldmodel[train,env]`; that extra's `transformers>=4.50.0` resolved
  to a 4.x release when the artifact was built and resolves to **5.17.0**
  today (measured 2026-09-18), i.e. following their README verbatim installs
  the one library version that cannot load their own checkpoints. The job
  therefore pins `transformers>=4.50,<5` after env creation and asserts the
  major version. Anyone else reproducing this artifact will hit the same wall.
  Their `[env]` extra also pulls `box2d-py`, whose wheel needs `swig` in the
  image; PushT does not use Box2D, so a `[train]`-only env is a valid
  fallback.
- **Their artifact needs their own environment — our pixi env cannot run it.**
  The released object checkpoints pickle **transformers-4** ViT classes
  (`ViTSelfAttention`, `ViTSelfOutput`, `ViTIntermediate`, `ViTOutput`,
  `ViTLayer`, verified by reading the pickle opcodes without executing them)
  plus their own modules under bare top-level names (`hi_jepa.HiJEPA`,
  `module.ARPredictor`). Our env is transformers 5, whose ViT refactor removed
  those paths — the same refactor that renames weights to
  `encoder.layers.N.attention.q_proj.*` and makes their HF baseline converter
  fail with `Missing key(s)`. So the job now builds their
  `environment-gpu.yml` with micromamba and runs every h_le_wm command inside
  it; our pixi env keeps only `fetch_dataset`, `convert_pldm` and
  `compare_base`. (`--allow-non-strict` would mask this by loading a partly
  random encoder — never use it.)
- **Two earlier bugs**, both fixed here: (a)
  `fetch-baselines` takes `--checkpoint`, not `--name`; (b)
  `h_le_wm.eval.baseline_manifest` is hard-wired to the UPSTREAM config
  (`config_path=../../third_party/lewm/config/eval`, `config_name=pusht`), so
  the flat arm must pass `--config-name=pusht`, has no `planning` block, has no
  `cache_dir` key, and needs `+eval.device`. Dataset location is now handled by
  symlinking into `$STABLEWM_HOME/datasets/`, which both configs resolve.
- `policy=` values are registry relpaths minus `_object.ckpt`:
  flat `pusht/lewm`, hierarchical
  `runs/pusht_hierarchical_default/pusht_hierarchical_default_epoch_15`.

## Open items

- ~~HF download of `quentinll/lewm-pusht` for step 1~~ — done in-job, verdict
  IDENTICAL.
- Their `config/eval/` also ships `hi_tworoom.yaml` and `hi_reacher.yaml`, so
  their code covers those envs — but no checkpoints are released for them.
- `pusht/vq/{vq16,vq128}` and `pusht/fixed_stride_dim{8,32}` variants are
  included; the fixed-stride ones are the closest analogue to our own port
  (ours: fixed stride 25, macro dim 8) and are the natural ablation to read
  against our 43.3-at-h25 collapse.
