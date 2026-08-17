# DMPO — Deep Model Predictive Optimization as an RLP baseline

Sacks, Rana, Huang, Spitzer, Shi, Boots, *Deep Model Predictive
Optimization*, ICRA 2024 ([arXiv:2310.04590](https://arxiv.org/abs/2310.04590),
code [`jisacks/dmpo`](https://github.com/jisacks/dmpo)).

DMPO and RLP answer the same question — *replace the hand-designed planner with
a learned one* — from opposite ends. RLP learns a **refiner** that moves a plan
along the value gradient (9 world-model rollouts per decision). DMPO learns the
**reduction inside a sampling optimizer**: keep MPPI's sample-rollout-reduce
loop, and let an MLP turn the `N` rollout costs into the next sampling
distribution. It is the strongest available "learned optimizer" baseline for
the RLP tables, and it is a *residual* on MPPI, so the comparison is clean:
untrained, it reproduces the MPPI row exactly.

## Paper component → code

| Paper | Code |
|---|---|
| Learned update rule `m_phi` (Eq. 13–14: gated MPPI residual, multiplicative covariance) | `DMPONet.forward` — `src/rlp/core/planner/dmpo.py` |
| Learned warm start / shift model `Phi_phi` (Sec. IV-D) | `DMPONet.warm_start` |
| Fixed Halton sample set, current mean always sampled | `gaussian_halton`, `DMPONet.plans` |
| MPPI inner update (Eq. 5–6, min-max cost scaling, dynamic mirror descent step) | `DMPONet.mppi_mean` |
| `is_mppi` ablation | `core.solver.mppi_mode=true` |
| Deployment (rollouts + cost + inner loop) | `DMPOSolver` — `src/rlp/core/solver/dmpo.py` |
| Training | `src/rlp/train/dmpo.py`, config `configs/train/dmpo.yaml` |

Reference hyperparameters (one 256-unit ReLU hidden layer, last layer
`N(0, 1e-3)`, temperature 0.05, step size 0.8, cost scaling on, gate, learned
covariance, one inner iteration) are the defaults in
`configs/core/planner/dmpo.yaml`.

## Two trainers: `model=dmpo` and `model=dmpo_ppo`

| | `model=dmpo` (pathwise) | `model=dmpo_ppo` (offline DMPO) |
|---|---|---|
| objective | `V(z_T(mu_K), z_g)` of **one** decision | discounted return over `decisions` closed-loop decisions |
| algorithm | backprop through the frozen world model | PPO + GAE, forward-only rollouts |
| search heads | absent (deterministic update) | present — the sampled `(mu, Sigma)` is the policy action |
| critic | — | `DMPOCritic` over `(z_t, z_g, theta_{t-1})`, the paper's auxiliary state |
| trains the shift model | no (single decision) | yes (credit crosses decisions) |
| env steps | 0 | 0 |

`model=dmpo_ppo` is the paper's *algorithm*, closed inside the world model:
each imagined episode runs the whole MPC-in-the-loop policy for several
decisions, the reward is progress in the critic's cost-to-go
`V(z_t, z_g) - V(z_{t+1}, z_g)`, and PPO optimizes the discounted sum. Nothing
differentiates through the world model, exactly as on hardware. It is the
closer reproduction and the one to prefer when the DMPO row has to defend
itself as DMPO; `model=dmpo` remains the cheaper apples-to-apples comparison
against RLP, which is trained pathwise in the same way.

Both write the same checkpoint format, so `core/solver=dmpo` deploys either
(deployment always uses the distribution *locations* — the reference's
`use_mean`).

What neither trainer reproduces: the paper measures return on the **system**,
with the model only inside the inner loop, under domain randomization. That is
what lets its learned optimizer compensate for model error — the robustness
claim. Here training and the inner loop share one frozen world model, so model
error is invisible to training and only surfaces at evaluation. Closing that
gap needs environment rollouts.

## What differs from the paper, and why

1. **Pathwise gradients instead of PPO** (`model=dmpo`; `model=dmpo_ppo` closes
   this gap for the algorithm, though not for the on-system objective). DMPO is trained with PPO because its
   costs come from a real quadrotor: no analytic gradient exists. Here the
   world model is differentiable, so the same networks are trained by
   backpropagating `V(z_T(mu_K), z_g)` through the frozen world model — the
   convention this repository's own planner uses. Consequence: the actor's
   stochastic search heads (`mean_search_std`, `std_search_std`), which exist
   only to give PPO a policy gradient, are dropped. Everything on the forward
   path is the reference computation. This makes the DMPO row *comparable* to
   the RLP row (same data, same frozen critic, same objective, different
   learned planning procedure) but it is **not** a replication of the paper's
   quadrotor result.
2. **The gate is `tanh`, following the authors' code.** The paper's text
   describes a sigmoid gate in `[0, 1]`; `dmpo_policy.py` uses `tanh`.
   `core.planner.gate_activation=sigmoid` gives the paper-literal variant.
3. **Costs are the goal-conditioned critic, not a task cost.** DMPO plans
   against the same quasimetric value the RLP solver plans against (recorded in
   its checkpoint), so a DMPO-vs-RLP table isolates the planner. `amax` (the
   symmetric plan clip in z-scored action units) replaces the quadrotor's
   asymmetric thrust bounds.
4. **The learned warm start is inert under the shipped eval protocol.** All
   environments run open loop (`planning.receding_horizon=5` with a 5-block
   plan), so nothing survives the shift-forward and each decision cold-starts.
   Evaluate with `planning.receding_horizon=1` for the closed-loop regime DMPO
   was published in; train the shift model with `warm_start_every>0`.

## Cost per decision

| planner | world-model unrolls per decision |
|---|---|
| CEM / MPPI (repo defaults) | 9,000 forward (300 samples × 30 iterations) |
| Adam | 3,000 forward + 3,000 backward |
| **DMPO** (defaults) | **256 forward** (256 samples × 1 iteration), no backward |
| RLP / LIP | 9 forward + 8 backward |

The sample count is baked into the trained network — the actor reads the `N`
costs positionally — so it cannot be changed after training; the solver logs
and ignores a mismatching config value. The iteration count can be varied at
test time (`core.solver.iters`), as the paper does.

## Commands

DMPO trains against a frozen critic, so produce the caches and `value_td`
first (the RLP pipeline with the planner stage skipped), then train and
evaluate:

```bash
pixi run train model=rlp skip=[planner] \
    wm=assets/core/world_model/lewm_cube \
    dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance name=cube_lewm
```

```bash
pixi run train model=dmpo wm=assets/core/world_model/lewm_cube \
    cache=$RLP_DATA_HOME/caches/cube_lewm_fs5.pt \
    h5=$RLP_DATA_HOME/caches/cube_lewm_actions.h5 \
    init_value=logs/<date>/<time>/checkpoints/value_td \
    core.planner.action_limit=1.6
```

```bash
pixi run eval model=lewm core/solver=dmpo core.solver.actor_path=<dmpo.pt>
```

Offline DMPO (the PPO objective) swaps one command:

```bash
pixi run train model=dmpo_ppo wm=assets/core/world_model/lewm_cube \
    cache=$RLP_DATA_HOME/caches/cube_lewm_fs5.pt \
    h5=$RLP_DATA_HOME/caches/cube_lewm_actions.h5 \
    init_value=logs/<date>/<time>/checkpoints/value_td \
    core.planner.action_limit=1.6
```

Baselines for the same cell — the hand-written update DMPO learns a residual
on, under the same critic:

```bash
pixi run eval model=lewm core/solver=mppi core/value=metric core.value.checkpoints=[<value_td>]
```

```bash
pixi run eval model=lewm core/solver=dmpo core.solver.actor_path=<dmpo.pt> core.solver.mppi_mode=true
```

Window critics (Reacher's three-frame quasimetric) are supported on both
sides: the trainer and solver detect the context width from the critic's
`latent_dim` and score the last `context` imagined frames the way
`MetricCost` does at eval (`rlp.core.temporal.windowed_terminal_value`).

Reporting protocol is the repository's: hyperparameter selection on eval seeds
50/51, report on 42/43/44 × 50 episodes, and **three optimizer training seeds**
per quoted cell.

## Results

| environment | record |
|---|---|
| TwoRoom | [RESULTS_dmpo_tworoom.md](RESULTS_dmpo_tworoom.md) |
| Reacher | [RESULTS_dmpo_reacher.md](RESULTS_dmpo_reacher.md) |
| OGBench Cube | [RESULTS_dmpo_ogbench_cube.md](RESULTS_dmpo_ogbench_cube.md) |
| campaign protocol, wall-clock, provenance | [RESULTS_dmpo_20260816.md](RESULTS_dmpo_20260816.md) |

## Cluster campaign

`scripts/sky/dmpo_campaign.yaml` runs one cell (environment × base × value
window) per managed job: dataset → caches → `value_td` → DMPO per train seed →
h25 evaluation of DMPO and MPPI-under-the-same-value on report seeds. Launch
examples are in the file header.
