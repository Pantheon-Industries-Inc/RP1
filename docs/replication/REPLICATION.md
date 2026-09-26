# Replicating the paper

How the paper's method and tables map to this repository's commands. Setup is in the
[README](../../README.md); every command runs from the repository root.

| Paper | Code |
|---|---|
| Plan refiner `F_theta` (Eq. 10–13) | `rp1.core.agent.planner.PlannerNet`, trained by `rp1.training.phases.agent.rp1_ac` |
| Goal-conditioned quasimetric value `V` (Eq. 4, App. B.1) | `rp1.core.agent.value.QuasimetricHead`, trained offline by `rp1.training.phases.agent.metric` and co-trained in `rp1_ac` |
| Frozen world-model rollout `H_phi` (Eq. 1) | `rp1.core.world_model.rollout` |
| rp1 at plan time | `rp1.core.agent.solver.RP1Solver` (`core/agent/solver=rp1`) |
| Baselines CEM / MPPI / Adam | `core/agent/solver=cem`, `mppi`, `adam` |
| Latent vs. value objective (Sec. 6, App. C) | `core/agent/value=latent` vs. `core/agent/value=metric` |
| No-move floor (Cube skill normalization) | `core/agent/policy=no_move` |
| DMPO and L2O-MPC baselines | [DMPO](../dmpo/README_dmpo.md), [L2O-MPC](../l2o/README_l2o.md) |

## 1. Training the agent

`posttrain` runs the full agent pipeline against a frozen world model: latent caches at one and at
`frameskip` primitive steps per row, the action h5, the offline quasimetric value, and actor-critic
training of the planner. Its defaults are the recipe shared by every environment.

```bash
pixi run prepare job=fetch_dataset preparation.dataset=ogb_cube
pixi run posttrain \
    training.wm=assets/core/world_model/cube_lewm \
    training.dataset=$RP1_DATA_HOME/datasets/ogb_cube_single.lance \
    training.name=cube_lewm training.planner.action_limit=1.6
```

The run's `checkpoints/` holds `value_td` (the offline value), `value_ac` (the co-trained teacher) and
`planner.pt`, which refers to `value_ac`. Caches go to `$RP1_DATA_HOME/caches/`, so
`training.stages=[value,planner]` iterates on the recipe without re-encoding.

What differs per environment:

| cell | overrides |
|---|---|
| Cube LeWM | `training.planner.action_limit=1.6` |
| Cube PLDM | `training.planner.action_limit=4.5` |
| TwoRoom | `training.planner.max_delta=12 training.planner.replay_prob=0 training.value.gamma=1.0 training.value.expectile=0.1 training.value.n_step=50 training.value.steps=6000`, with the action limit, mean weight and actor learning rate of each cell from App. C.1 |
| Reacher | `training.planner.steps=1000 training.planner.batch=128 training.planner.max_delta=12 training.planner.expand_weight=0 training.planner.replay_prob=0.5`; LeWM `action_limit=2.2 mean_weight=0.3 actor_lr=1e-4 actor_lr_final=1e-5`, PLDM `action_limit=1.8 mean_weight=0.5 actor_lr=3e-4 actor_lr_final=3e-5` (all under `training.planner.`); a three-frame window value (`training.value.window_frames=3 training.value.window_lag=5 training.value.expectile=0.05 training.value.gamma=0.98`) |

The vendored agents in `assets/core/agent/` are the paper's: Cube per world model and seed, TwoRoom per
world model, recipe variant (`a<action limit>_m<mean weight>_l<learning rate>`) and seed.

## 2. Evaluating a table cell

Each `evaluate` run is one environment × world model × planner × objective × horizon cell. The
benchmark sets the environment and world model; `benchmark.goal_offset_steps=25 planning.budget=50`
is h25 and `benchmark.goal_offset_steps=100 planning.budget=200` is h100. Report seeds are 42, 43 and
44 (`runtime.seed`), 50 episodes each; seeds 50 and 51 were used for selection only.

Every environment, Reacher included, runs open loop: the whole plan executes before the next decision
(`planning.receding_horizon=5`).

```bash
pixi run evaluate benchmark=cube_lewm core/agent/solver=rp1 core.agent.solver.checkpoint.path=<planner.pt>
pixi run evaluate benchmark=cube_lewm core/agent/solver=cem
pixi run evaluate benchmark=cube_lewm core/agent/solver=mppi
pixi run evaluate benchmark=cube_lewm core/agent/solver=adam
pixi run evaluate benchmark=cube_lewm core/agent/solver=cem \
    core/agent/value=metric core.agent.value.checkpoints=[<value_td>]
pixi run evaluate benchmark=cube_lewm core/agent/policy=no_move
```

The other benchmarks are `cube_pldm`, `tworoom_lewm`, `tworoom_pldm`, `reacher_lewm` and
`reacher_pldm`. On Linux GPU nodes MuJoCo renders with `MUJOCO_GL=egl`; pin `MUJOCO_EGL_DEVICE_ID` and
run one evaluation per GPU.

## 3. World models

The LeWM Cube world model can be trained from scratch; [cube/REPLICATION_CUBE.md](cube/REPLICATION_CUBE.md)
records the dataset fingerprint, the training command and the pitfalls. For TwoRoom and Reacher, train
on the environment's play data:

```bash
pixi run prepare job=collect_tworoom_mixed preparation.expert=0 preparation.random=10000 preparation.out=tworoom_play.lance
pixi run pretrain data=tworoom_lewm
pixi run pretrain data=reacher_lewm
```

The PLDM world models are the authors' checkpoints converted into the LeWM key layout, which shares the
architecture. `job=convert_pldm` converts another PLDM export; pair the result with the `config.json` of
`assets/core/world_model/cube_pldm`. `pixi run pretrain --config-name phases/world_model/pldm` trains
one from scratch with the authors' objective, on PushT by default (`data=` selects another dataset).

```bash
pixi run prepare job=convert_pldm preparation.src=<pldm.pt> preparation.dst=<out.pt>
```

The public Reacher h5 pads each episode's terminal step with NaN actions, so any normalization of its
actions must ignore NaNs; the trainers here do.

## 4. The Dyna round (Tab. 3, block d)

The Dyna-finetuned Cube world models are vendored, so `benchmark=cube_lewm_dyna` and
`benchmark=cube_pldm_dyna` evaluate row (d) with any planner, and `posttrain` with
`training.wm=assets/core/world_model/cube_lewm_dyna` retrains its agents. Regenerating the finetuned
world models is a procedure:

1. collect episodes with the trained planner on h25 tasks from the training split (episodes 0–7999),
   without terminating at the goal, recording observations and actions;
2. mix them 50:50 with the offline data;
3. finetune the world model on the mixture for two epochs at learning rate 1e-5 and keep the first epoch
   (`pixi run pretrain training.initial_weights=<base>` on the mixed dataset);
4. retrain the value and planner on the finetuned world model with the unchanged recipe, from fresh
   caches.

## 5. Data splits

- **Cube and Reacher**: the agent trains on episodes 0–7999 (`training.train_episodes=8000`) and
  evaluation draws tasks from episodes 8000–9999, as the benchmarks set.
- **TwoRoom**: the LeWM/DINO-WM protocol, where training and evaluation share the full pool. Every
  planner draws from the same pool, so comparisons within a table hold. For a held-out split, train with
  `training.train_episodes=8000` and evaluate with `benchmark.episode_range="8000:10000"`.

A reported rp1 cell averages three independently trained agents (`runtime.seed=0,1,2`; six for Reacher),
each evaluated on the report seeds. A single agent seed is a smoke test, not a replication.
