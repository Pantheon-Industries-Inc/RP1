# RLP — the good configs (verified 2026-09-14)

Every value below was read from the tree by an independent auditor and then
adversarially cross-checked against the code that consumes it. Where the tree
and the older command sheets disagree, the tree wins and the conflict is
flagged. Supersedes the recipe tables in `REPLICATION_RLP.md` and `README.md`,
both of which are stale on `amax` and on the Cube/TwoRoom data contract.

## 1. What the recipe scores

Medians over **training** seeds; each seed is the mean over **eval draws
42/43/44**, 50 episodes each. One K=8 refinement pass per decision, no
restarts, no behaviour cloning.

| environment | base | RLP, easy column | RLP, hard column | n | sampling baseline (≈9,000 rollouts/decision) |
|---|---|---|---|---|---|
| TwoRoom | LeJEPA | 100.0 (h25) | 94.7 (h100) | 6 | — |
| TwoRoom | PLDM | 96.7 | 82.7 | 3 | — |
| Cube | LeWM | 90.0 (h25) | 86.3 (h100) | 6 | latent+CEM 78.0 (h25) |
| Cube | PLDM | 87.3 | 85.3 | 3 | latent+CEM 62.7 (h25) |
| Reacher (w2) | LeWM | 100.0 (τ 0.1) | 96.3 (τ 0.05) | 6 | latent+CEM 88.0 (τ 0.05) |
| Reacher (w2) | PLDM | 99.3 | 90.7 | 3 | latent+CEM 88.0 (τ 0.05) |
| **PushT** | LeWM | **73.7** (h25) | — | **6** | **latent+CEM 79.3** |

PushT is the only environment where RLP trails the sampler, and the only one
with a single horizon column and a single world-model base.

## 2. The one thing to know before running anything

**`pixi run train model=rlp wm=… dataset=…` with no other flags is config B
for Cube only.** `configs/train/rlp.yaml` was rewritten to be cube-shaped.
For the other three environments the bare command silently trains a different
recipe. Per-environment overrides below are mandatory, not optional polish.

**Early stopping is part of config B and is OFF by default.**
`planner.ckpt_every` ships `null`, so a bare run writes no snapshots and
deploys the last step. Every shipped number deployed the argmax over
`{planner_step2000.pt, planner_step4000.pt, planner.pt}` scored on selection
draws. Selection draws are **48–51** everywhere except PushT, which uses
**50/51**. Never report a selection draw.

## 3. PushT — the current best, and what changed today

Base: the official `quentinll/lewm-pusht` release (HF rev `22b330c2…`),
fetched and converted in-job with `tool=convert_pldm`. Not tracked in-tree.
Split: train episodes 0–15999, eval draws from 16000–18685.

```bash
pixi run train model=rlp \
  wm=$WM dataset=$H5 name=counterstrike cache_directory=$CD seed=$S \
  train_episodes=16000 \
  value.gamma=0.98 value.expectile=0.03 value.n_step=1 \
  value.window_frames=1 value.window_lag=5 \
  value.near_frac=0.3 value.near_max=3 \
  planner.amax=2.5 planner.mean_weight=0.1 \
  planner.actor_lr=3e-4 planner.actor_lr_final=3e-05 \
  planner.iterations=8 \
  planner.replay_prob=0 planner.max_delta=6 planner.p_cross=0.1 \
  planner.freeze_critic_frac=0 planner.ac_weight=0.5 planner.ckpt_every=2000
```

**Updated 2026-09-14 (E36).** Three changes from the version first published
today. `planner.max_delta=6 planner.p_cross=0.1` joins `replay_prob=0`: the
pair is **73.7 at n=6**, paired 90 fixes / 47 breaks, p = 0.00, against 72.3
for replay-off alone. `planner.expand_weight` and `planner.near_frac` were
removed because they are **inert** here (see §7a). Consider also
`value.steps=6000`, which is +1.3 at three seeds and is being confirmed.

Equivalently, via the launcher: `REPLAY=0 TAGSUF=-rp0 scripts/sky/launch_pusht_critic_arms.sh W1NEAR_FRZ 0 1 2 3 4 5`.

**`planner.replay_prob=0` was the first 2026-09-14 result**: 69.0 → **72.3** on the
six-seed median, six of six seeds positive, paired sign test p = 0.03, teacher
unchanged. At the shipped 0.5, half of every actor batch starts from an
*imagined* window with zeroed action history — a state distribution the
deployed replan never sees, and on PushT exactly where the world model's
derivative is least trustworthy. The global default is deliberately still 0.5
because this is measured on PushT only.

The other non-default choices, all from the W1NEAR_FRZ arm: single-frame
critic (`window_frames=1`, against config B's declared w4 for PushT) because
the single-frame near-goal teacher is the only PushT critic that reaches
latent-L2 parity as a ranking objective; `freeze_critic_frac=0`, i.e. the
offline TD teacher is frozen from step 0 and never co-trained; near-goal
hindsight oversampling on both learners.

Eval, per training seed:

```bash
for S in 42 43 44; do
  pixi run eval model=pusht_lewm core.world_model.checkpoint=$WM \
    runtime.seed=$S evaluation.num_episodes=50 \
    core/solver=lip core.solver.actor_path=<run>/checkpoints/planner.pt
done
```

## 4. Cube — the only environment whose defaults are already right

```bash
pixi run train model=rlp \
  wm=assets/core/world_model/lewm_cube \
  dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance \
  name=cube seed=$S \
  planner.ckpt_every=2000
```

Swap `wm=assets/core/world_model/pldm_cube` for the PLDM column. Do **not**
pass `planner.amax=1.6` or `4.5`; those are the pre-config-B paper recipe and
are stale in `REPLICATION_RLP.md` and the `train/rlp.py` docstring. Config B
trains both bases at 2.5.

The Dyna-finetuned world models (`assets/core/world_model/*_cube_dyna`) are a
separate line, not part of this recipe. Their best banked numbers are LeWM
94.7 / 87.3 and PLDM 94.0 / 86.0 at iteration 2.

## 5. TwoRoom — seven overrides

```bash
pixi run train model=rlp \
  wm=assets/core/world_model/lejepa_tworoom dataset=<tworoom lance> \
  name=tworoom seed=$S state_key=pos_agent \
  planner.replay_prob=0 planner.expand_weight=0 \
  planner.freeze_critic_frac=0.8 \
  planner.steps=8000 planner.batch=128 value.steps=6000 \
  planner.actor_lr=1e-4 planner.actor_lr_final=1e-5 \
  planner.ckpt_every=2000
```

PLDM base: `planner.actor_lr=1e-3 planner.actor_lr_final=1e-4
planner.mean_weight=0.0` instead of the LeJEPA anchors.

Two further conflicts the auditor flagged and I have **not** resolved, because
resolving them needs a run, not a reading:

- `value.expectile` — the tree ships 0.03 (the Cube value); the producing
  harness hardcoded TwoRoom's offline teacher at **0.1**.
- `planner.n_step` — the tree ships 50; the producing harness drove **both**
  critics from one env var at **1**.

If you rerun TwoRoom, add `value.expectile=0.1 planner.n_step=1` and treat the
result as the check on both.

## 6. Reacher — the w=2 environment

```bash
pixi run train model=rlp \
  wm=assets/core/world_model/lejepa_reacher dataset=<reacher h5> \
  name=reacher seed=$S \
  value.window_frames=2 value.window_lag=5 \
  planner.expand_weight=0 \
  planner.mean_weight=0.3 \
  planner.actor_lr=1e-4 planner.actor_lr_final=1e-5 \
  planner.ckpt_every=2000
```

PLDM base: `planner.mean_weight=0.5` and leave the learning rates at the 3e-4
default. `window_lag=5` is mandatory whenever `window_frames >= 2` or the
pipeline raises.

`w=2` is the single biggest swing in the campaign: τ 0.05 goes 82.7 at w=1 →
96.3 at w=2. It is the minimal window that makes velocity observable in a
second-order system; w=3, the paper's choice, adds world-model noise.

`MUJOCO_GL=egl` for every render, never osmesa, and pin
`MUJOCO_EGL_DEVICE_ID` per pod.

## 7. Traps that produce a wrong number silently

1. **Checkpoint paths.** The trainer writes
   `logs/<date>/<time>/checkpoints/planner.pt`, not `$D/train_checkpoints/`.
   Snapshots are `planner_step2000.pt`, not `planner.pt.step2000.pt`. The
   selection loop must pass full paths and must `cp` the winner over
   `planner.pt`, because the report pass reads that filename unconditionally.
2. **`train_episodes` is applied at the cache stage only.** Re-running with
   `skip=[cache,subsample,actions]` against a cache built at a different cap
   silently trains on the wrong pool, and the cache filename does not encode
   the cap. Cache artifacts are keyed on `name` alone.
3. **Eval needs `RLP_DATA_HOME` set.** The action z-scoring stats are fitted
   from `data.stats` at eval time; an unset variable resolves elsewhere and
   changes the action convention.
4. **Cube eval ran in bf16** with `episode_range 8000:10000`. Omitting bf16
   does not reproduce the banked cells.
5. **Reacher eval** points at `dmc/reacher_random.h5`, which must be built
   first, and its pinned `episode_range 8000:10000` needs more than the
   default 1024 eval episodes.
6. **TwoRoom's shipped numbers are held-out** (`episode_range 8000:10000`)
   while `configs/eval/tworoom_pixels.yaml` ships `null`, the full 10k pool.
   The command sheet says the opposite; the sheet is wrong.
7. **Cube's offline teacher was shared across training seeds** in the
   producing harness. The hydra pipeline trains one per seed, so seed spread
   will differ from the banked table.

## 7a. Inert settings — passing them changes nothing

`planner.freeze_critic_frac=0` makes `freeze_at = 0`, so the in-loop
`critic_step()` never runs, and `pretrain = -1` with `init_value` set skips the
warmup loop too. No critic is ever updated in stage 5; the teacher is a verbatim
copy of the stage-4 checkpoint. Every knob whose only use site is inside
`critic_step`, its schedules, or the sampler it consumes is therefore dead:

`planner.expand_weight`, `planner.expand_traj`, `planner.gamma`,
`planner.n_step`, `planner.expectile`, `planner.expectile_final`,
`planner.td_batch`, `planner.td_p_cross`, `planner.td_max_delta`,
`planner.critic_lr`, `planner.critic_lr_final`, `planner.critic_wd`,
`planner.ema_tau`, `planner.huber_beta`, `planner.critic_ratio`,
`planner.pretrain`, `planner.near_frac`, `planner.near_max`.

Near-goal oversampling still matters, but only through **`value.near_frac`**,
which reaches the offline teacher. The launcher's `EXPAND` and the
`planner.near_frac` half of `NEAR_FRAC` have been dead in every run of this
family. This applies only where the critic is frozen from step 0; TwoRoom
(`freeze_critic_frac=0.8`) and Cube (0.5) do co-train and these knobs are live
there.

## 8. Closed knobs — do not re-sweep

Shared: `gamma` 0.98 (Reacher loses 7–14 points at 0.99/1.0), `vnorm` none,
`amax` 2.5, `K` 8 (24/32 are flat-to-worse and off-protocol), `depth` 2,
`ema_tau` 0.005.

PushT specifically: `mean_weight` 0.1, `actor_lr` 3e-4, actor steps 6000
(12k/18k worse), actor batch 256, `max_delta` 10 (negative alone; 6 is the
winner ONLY in combination with replay-off),
**teacher steps: the peak is 6000, not the shipped 12000** (3k and 24k both lose
7.4 points; at 6k the teacher's own sampling score also rises to 78.7),
`ac_weight` 0.5 — **load-bearing**, removing it costs ~7 points at p = 0.01,
and the physics-grounding term (E32), which is a null.
