# Replicating "Reinforcement Learned Planning with Latent World Models"

This is the command sheet for reproducing the paper's method and tables from
this repository. The paper calls the learned planner **rp1**; in the code and
checkpoints it is called **rp1** (Learned Iterative Planner) — they are the
same thing.

| Paper concept | Code |
|---|---|
| rp1 plan refiner `F_theta` (Eq. 10–13) | `src/rp1/core/planner/net.py`, trained by `src/rp1/training/rp1_ac.py` |
| Goal-conditioned quasimetric critic `V` (Eq. 4, App. B.1) | `src/rp1/core/value/head.py` (`QuasimetricHead`, an MRN), trained by `src/rp1/training/metric.py` (offline) and co-trained inside `rp1_ac.py` |
| Frozen world-model rollout `H_phi` (Eq. 1) | `src/rp1/core/rollout.py` |
| rp1 at plan time (9 forward + 8 backward unrolls) | `src/rp1/core/solver/rp1.py` |
| Baselines CEM / MPPI / Adam | `src/rp1/core/solver/{cem,mppi,gradient}.py`, configs `configs/core/solver/{cem,mppi,adam}.yaml` |
| Latent vs. value objective (Sec. 6, App. C) | `configs/core/value/latent.yaml` vs. `core/value=metric core.value.checkpoints=[<value_ckpt>]` |
| No-op floor (Cube "skill" normalization) | `configs/core/policy/no_move.yaml` |

Setup (pixi, datasets, LFS checkpoint) is covered in the top-level
[README](../../README.md). All commands below run from the repo root.

## 1. One-command rp1 training

`model=rp1` runs the paper's full training stack against a frozen pretrained
world model: latent caching (fs1 + frameskip-matched fs5), action-h5
extraction, offline quasimetric value learning, and actor-critic planner
training. Stage defaults are the paper's OGBench Cube recipe
(Tab. "Selected rp1 configurations for OGBench Cube").

```bash
pixi run prepare job=fetch_dataset preparation.dataset=ogb_cube
pixi run training model=rp1 \
    training.wm=assets/core/world_model/cube_lewm \
    training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance \
    training.name=cube_lewm training.planner.action_limit=1.6
```

Outputs land in the run directory (`logs/<date>/<time>/checkpoints/`):
`value_td` (offline critic), `value_ac` (co-trained teacher), `planner.pt`
(the rp1 refiner). Reusable caches go to `$RP1_DATA_HOME/caches/` and can be
reused across recipes with e.g. `skip=[cache,subsample,actions]`.

Per-environment recipe deltas (everything else is shared, see
`configs/training/rp1.yaml`):

| cell | override |
|---|---|
| Cube LeWM | `planner.action_limit=1.6` (default) |
| Cube PLDM | `planner.action_limit=4.5` |
| TwoRoom (both bases) | `planner.max_delta=12 planner.replay_prob=0 value.gamma=1.0 value.expectile=0.1 value.n_step=50 value.steps=6000` + per-cell `planner.action_limit`/`planner.mean_weight`/`planner.actor_lr` from App. C.1 |
| Reacher | the checkpoint-verified producing recipe (`reacher/HYPERS_20260802.md`, git history): `planner.steps=1000 planner.batch=128 planner.max_delta=12 planner.expand_weight=0 planner.replay_prob=0.5`, LeWM `planner.action_limit=2.2 planner.mean_weight=0.3 planner.actor_lr=1e-4 planner.actor_lr_final=1e-5`, PLDM `planner.action_limit=1.8 planner.mean_weight=0.5 planner.actor_lr=3e-4 planner.actor_lr_final=3e-5`; init value = 3-frame window quasimetric (expectile 0.05, γ 0.98); 6 train seeds, report draws 42–47. **The paper's App C.3 table does not match these artifacts — fix the paper, not the recipe.** |

## 2. Evaluation — producing a table cell

The evaluation driver is `rp1.inference.evaluate`; each invocation is one
(environment × base × planner × objective × horizon) cell. Horizons:
`benchmark.goal_offset_steps=25 planning.budget=50` (h25) or
`goal_offset_steps=100 budget=200` (h100). Reporting protocol: seeds 42/43/44,
50 episodes each, selection on 50/51 only (never report those).

All environments — **including Reacher** — run open loop: the full 5-chunk
plan (25 primitive steps) executes before replanning
(`planning.receding_horizon=5`, the default). The paper appendix's shared
protocol note claiming Reacher replans every chunk is an error in the
appendix, not the protocol.

```bash
# rp1 (9 rollouts/decision)
pixi run inference benchmark=lewm core/solver=rp1 core.solver.checkpoint.path=<planner.pt>

# CEM / MPPI (9,000 rollouts) and Adam (3,000 fwd + 3,000 bwd), latent objective
pixi run inference benchmark=lewm core/solver=cem
pixi run inference benchmark=lewm core/solver=mppi
pixi run inference benchmark=lewm core/solver=adam

# Same baselines under the learned value objective (App. C tables)
pixi run inference benchmark=lewm core/solver=cem \
    core/value=metric core.value.checkpoints=[<value_td>]

# No-op floor for the Cube skill normalization
pixi run inference benchmark=lewm core/policy=no_move
```

`model=lewm` / `model=pldm` select the Cube eval roots. For TwoRoom and
Reacher use the parametric roots and pass the environment's checkpoint
explicitly:

```bash
pixi run inference benchmark=tworoom_lewm core.world_model.checkpoint=<tworoom_ckpt> \
    core/solver=rp1 core.solver.checkpoint.path=<planner.pt>
pixi run inference benchmark=reacher_lewm core.world_model.checkpoint=<reacher_ckpt> \
    core/solver=cem
```

Rendering on Linux GPU nodes: always `MUJOCO_GL=egl` with a pinned
`MUJOCO_EGL_DEVICE_ID`, and serialize evaluations per node rather than
switching renderers.

## 3. World-model bases

**LeWM (cube)** is tracked in-tree via Git LFS
(`assets/core/world_model/cube_lewm`) and is bit-replicable from scratch —
see [REPLICATION_CUBE.md](cube/REPLICATION_CUBE.md) for the dataset
fingerprint, the seven known replication traps, and the exact training
command. For TwoRoom/Reacher bases, train LeWM on the corresponding play
dataset:

```bash
pixi run prepare job=collect_tworoom_mixed preparation.expert=0 preparation.random=10000 preparation.out=tworoom_play.lance
pixi run training model=lewm training/data=tworoom_lewm
pixi run training model=lewm training/data=reacher_lewm
```

**PLDM** cells use the authors' pretrained PLDM checkpoint converted 1:1 into
the LeWM key layout (both are vit_hf tiny/patch14/224; 303/303 keys map with
identical shapes, validated to ~2e-6 agreement — see
[docs/rp1/lip_results.md](../rp1/lip_results.md)). The converted cube
checkpoint is tracked in-tree at `assets/core/world_model/cube_pldm`, and the
converter is a maintained tool for other PLDM exports:

```bash
pixi run prepare job=convert_pldm preparation.src=<authors_pldm.pt> preparation.dst=<out.pt>
```

Pair the converted weights with a LeWM-target `config.json` (copy the one in
`assets/core/world_model/cube_pldm/`). Everything in Sections 1–2 then applies
unchanged with `wm=assets/core/world_model/cube_pldm` / `model=pldm`.

**Caveat on Reacher data**: the public reacher h5 pads every episode's
terminal step with NaN actions. Any normalization must use `nanmean`/`nanstd`
and assert finiteness — plain stats poison every action and training runs
silently at 100% GPU producing NaNs. The in-tree trainers handle this
(`rp1_ac.py` is nan-aware); custom preprocessing must too.

## 4. Dyna finetuning round (Tab. 3, block d)

**Evaluating row (d) is one command**: the Dyna-finetuned world models are
tracked in-tree (`assets/core/world_model/{lewm,pldm}_cube_dyna`, the
`dyna_wm_*` campaign artifacts' epoch-1 weights), so
`pixi run inference benchmark=lewm_dyna ...` / `benchmark=pldm_dyna ...` reproduces the
row-(d) cells with any planner. Retrain the row-(d) rp1 actors against the
finetuned base with `model=rp1 wm=assets/core/world_model/cube_lewm_dyna`.

*Regenerating* the finetuned world models is a procedure, not a single entry
point:

1. **Collect** on-policy episodes with the trained planner on h25 tasks from
   the training split (episodes 0–7999, no termination at goal), recording
   observations and actions (`pixi run inference core/solver=rp1 ...` with
   recording enabled).
2. **Mix** the collected episodes 50:50 with the original offline data.
3. **Finetune** the world model on the mixture for 2 epochs at LR 1e-5 and
   keep epoch 1 (`pixi run training model=lewm core.world_model.checkpoint=<base>`
   with the mixed dataset).
4. **Rebuild** the latent caches and retrain value + planner with the
   unchanged recipe (`model=rp1 wm=<finetuned>` — full rerun, no skips).
5. Reuse the finetuned model as-is for h100 evaluation.

The historical campaign harness for step 1–2 lives in git history on `main`
(`Dyna/`); it was deliberately not carried into this tree.

## 5. Data-split discipline — what the shipped numbers actually did

The split differs per environment; the configs encode the protocol each
table's numbers were actually produced under:

- **Cube** (all quoted rp1 rows, from the 2026-08-05/06 open-loop sweep
  onward): value and planner train on episodes 0–7999 (`train_episodes=8000`,
  a 1,608,000-row cache), eval tasks draw from 8000–9999
  (`benchmark.episode_range="8000:10000"`, pinned in the cube eval roots).
  Only the July 2026 Dyna-era measurements predate the split; none of those
  are quoted in the paper's tables.
- **Reacher**: same held-out split, pinned in `configs/inference/benchmark/reacher.yaml`.
- **TwoRoom**: the shipped numbers follow the original LeWM/DINO-WM contract —
  training and evaluation share the full 10k-episode pool (no split). All
  planner arms draw tasks from the same pool, so the within-table comparison
  is unaffected, but rp1 and value-objective rows are formally upper bounds.
  For the held-out variant, train with `train_episodes=8000` and evaluate with
  `benchmark.episode_range="8000:10000"`.

Hyperparameter selection uses eval seeds 50/51; report on 42/43/44 only.

**Seed protocol — mandatory for reportable numbers.** Every quoted rp1 cell
averages over **three independent actor/critic training seeds** (`seed=0,1,2`;
Reacher used six, 0–5) × the report eval draws. A single-seed run is a smoke
check, not a replication: actor-seed variance on Cube alone spans several
points (measured 2026-08-13: one LeWM actor seed gave 86.0 where the 3-seed
quote is 89.1). Train each seed with the same command varying only `seed=`,
then average the per-seed eval means.
