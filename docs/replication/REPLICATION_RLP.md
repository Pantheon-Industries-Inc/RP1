# Replicating "Reinforcement Learned Planning with Latent World Models"

This is the command sheet for reproducing the paper's method and tables from
this repository. The paper calls the learned planner **RLP**; in the code and
checkpoints it is called **LIP** (Learned Iterative Planner) — they are the
same thing.

| Paper concept | Code |
|---|---|
| RLP plan refiner `F_theta` (Eq. 10–13) | `src/rlp/core/planner/net.py`, trained by `src/rlp/train/lip_ac.py` |
| Goal-conditioned quasimetric critic `V` (Eq. 4, App. B.1) | `src/rlp/core/value/head.py` (`QuasimetricHead`, an MRN), trained by `src/rlp/train/metric.py` (offline) and co-trained inside `lip_ac.py` |
| Frozen world-model rollout `H_phi` (Eq. 1) | `src/rlp/core/rollout.py` |
| RLP at plan time (9 forward + 8 backward unrolls) | `src/rlp/core/solver/lip.py` |
| Baselines CEM / MPPI / Adam | `src/rlp/core/solver/{cem,mppi,gradient}.py`, configs `configs/core/solver/{cem,mppi,adam}.yaml` |
| Latent vs. value objective (Sec. 6, App. C) | `configs/core/value/latent.yaml` vs. `core/value=metric core.value.checkpoints=[<value_ckpt>]` |
| No-op floor (Cube "skill" normalization) | `configs/core/policy/no_move.yaml` |

Setup (pixi, datasets, LFS checkpoint) is covered in the top-level
[README](../../README.md). All commands below run from the repo root.

## 1. One-command RLP training

`model=rlp` runs the paper's full training stack against a frozen pretrained
world model: latent caching (fs1 + frameskip-matched fs5), action-h5
extraction, offline quasimetric value learning, and actor-critic planner
training. Stage defaults are the paper's OGBench Cube recipe
(Tab. "Selected RLP configurations for OGBench Cube").

```bash
pixi run tool tool=fetch_dataset dataset=ogb_cube
pixi run train model=rlp \
    wm=assets/core/world_model/lewm_cube \
    dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance \
    name=cube_lewm planner.amax=1.6
```

Outputs land in the run directory (`logs/<date>/<time>/checkpoints/`):
`value_td` (offline critic), `value_ac` (co-trained teacher), `planner.pt`
(the RLP refiner). Reusable caches go to `$RLP_DATA_HOME/caches/` and can be
reused across recipes with e.g. `skip=cache,subsample,actions`.

Per-environment recipe deltas (everything else is shared, see
`configs/train/rlp.yaml`):

| cell | override |
|---|---|
| Cube LeWM | `planner.amax=1.6` (default) |
| Cube PLDM | `planner.amax=4.5` |
| TwoRoom (both bases) | `planner.max_delta=12 planner.replay_prob=0 value.gamma=1.0 value.expectile=0.1 value.n_step=50 value.steps=6000` + per-cell `planner.amax`/`planner.mean_weight`/`planner.actor_lr` from App. C.1 |
| Reacher | `planner.max_delta=12 planner.batch=256 planner.steps=6000` + per-cell settings from App. C.3 (Tab. reacher-rlp-hypers) |

## 2. Evaluation — producing a table cell

The eval driver is `rlp.eval.world_model`; each invocation is one
(environment × base × planner × objective × horizon) cell. Horizons:
`evaluation.goal_offset_steps=25 evaluation.budget=50` (h25) or
`goal_offset_steps=100 budget=200` (h100). Reporting protocol: seeds 42/43/44,
50 episodes each, selection on 50/51 only (never report those).

```bash
# RLP (9 rollouts/decision)
pixi run eval model=lewm core/solver=lip core.solver.actor_path=<planner.pt>

# CEM / MPPI (9,000 rollouts) and Adam (3,000 fwd + 3,000 bwd), latent objective
pixi run eval model=lewm core/solver=cem
pixi run eval model=lewm core/solver=mppi
pixi run eval model=lewm core/solver=adam

# Same baselines under the learned value objective (App. C tables)
pixi run eval model=lewm core/solver=cem \
    core/value=metric core.value.checkpoints=[<value_td>]

# No-op floor for the Cube skill normalization
pixi run eval model=lewm core/policy=no_move
```

`model=lewm` / `model=pldm` select the Cube eval roots. For TwoRoom and
Reacher use the parametric roots and pass the environment's checkpoint
explicitly:

```bash
pixi run eval model=tworoom_lewm core.world_model.checkpoint=<tworoom_ckpt> \
    core/solver=lip core.solver.actor_path=<planner.pt>
pixi run eval model=reacher_lewm core.world_model.checkpoint=<reacher_ckpt> \
    core/solver=cem
```

Rendering on Linux GPU nodes: always `MUJOCO_GL=egl` with a pinned
`MUJOCO_EGL_DEVICE_ID`, and serialize evaluations per node rather than
switching renderers.

## 3. World-model bases

**LeWM (cube)** is tracked in-tree via Git LFS
(`assets/core/world_model/lewm_cube`) and is bit-replicable from scratch —
see [REPLICATION_CUBE.md](cube/REPLICATION_CUBE.md) for the dataset
fingerprint, the seven known replication traps, and the exact training
command. For TwoRoom/Reacher bases, train LeWM on the corresponding play
dataset:

```bash
pixi run tool tool=collect_tworoom_mixed expert=0 random=10000 out=tworoom_play.lance
pixi run train model=lewm data=tworoom_lewm     # TwoRoom base
pixi run train model=lewm data=reacher_lewm     # Reacher base
```

**PLDM** cells use the authors' pretrained PLDM checkpoint converted 1:1 into
the LeWM key layout (both are vit_hf tiny/patch14/224; 303/303 keys map with
identical shapes, validated to ~2e-6 agreement — see
[docs/lip/lip_results.md](../lip/lip_results.md)). The converted cube
checkpoint is tracked in-tree at `assets/core/world_model/pldm_cube`, and the
converter is a maintained tool for other PLDM exports:

```bash
pixi run tool tool=convert_pldm src=<authors_pldm.pt> dst=<out.pt>
```

Pair the converted weights with a LeWM-target `config.json` (copy the one in
`assets/core/world_model/pldm_cube/`). Everything in Sections 1–2 then applies
unchanged with `wm=assets/core/world_model/pldm_cube` / `model=pldm`.

**Caveat on Reacher data**: the public reacher h5 pads every episode's
terminal step with NaN actions. Any normalization must use `nanmean`/`nanstd`
and assert finiteness — plain stats poison every action and training runs
silently at 100% GPU producing NaNs. The in-tree trainers handle this
(`lip_ac.py` is nan-aware); custom preprocessing must too.

## 4. Dyna finetuning round (Tab. 3, block d)

The Dyna row is a procedure, not a single entry point:

1. **Collect** on-policy episodes with the trained planner on h25 tasks from
   the training split (episodes 0–7999, no termination at goal), recording
   observations and actions (`pixi run eval core/solver=lip ...` with
   recording enabled).
2. **Mix** the collected episodes 50:50 with the original offline data.
3. **Finetune** the world model on the mixture for 2 epochs at LR 1e-5 and
   keep epoch 1 (`pixi run train model=lewm core.world_model.checkpoint=<base>`
   with the mixed dataset).
4. **Rebuild** the latent caches and retrain value + planner with the
   unchanged recipe (`model=rlp wm=<finetuned>` — full rerun, no skips).
5. Reuse the finetuned model as-is for h100 evaluation.

The historical campaign harness for step 1–2 lives in git history on `main`
(`Dyna/`); it was deliberately not carried into this tree.

## 5. Data-split discipline

Everything trains on episodes 0–7999; all evaluation start/goal states draw
from episodes 8000–9999. Values and planners must never see the eval episodes
(the pre-split legacy numbers are upper bounds — do not mix generations).
Hyperparameter selection uses eval seeds 50/51; report on 42/43/44 only.
