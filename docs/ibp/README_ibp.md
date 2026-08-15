# IBP — the Imagination-Based Planner as an RLP baseline

Pascanu, Li, Vinyals, Heess, Buesing, Racanière, Reichert, Weber, Wierstra,
Battaglia, *Learning model-based planning from scratch*, 2017
([arXiv:1707.06170](https://arxiv.org/abs/1707.06170)).

IBP and RLP answer the same question — *replace the hand-designed planner with
a learned one* — but IBP learns something the other baselines do not: **how much
to plan, and where**. A discrete manager chooses at every step between acting
and imagining, and which node of an imagination tree to imagine from; a
controller proposes actions; an LSTM memory aggregates every imagined step into
the plan context that conditions both. The agent is trained to plan
*economically*, trading task performance against a charge for imagination.

That makes it the natural complement to DMPO in the RLP tables. DMPO learns the
*reduction* inside a fixed-budget sampling optimizer (always 256 rollouts);
RLP learns a *refiner* on a fixed budget (always 9 forward + 8 backward); IBP
learns the *budget itself*, and its rollout count is an outcome of training
rather than a constant.

## Paper component → code

| Paper | Code |
|---|---|
| Manager `pi^M` over routes `U = {act, s_j0, s_jk}` (Sec. 3, tree mode) | `IBPNet.route_logits` — `src/rlp/core/planner/ibp.py` |
| Controller `pi^C` | `IBPNet.propose` |
| Memory `mu` (LSTM plan context) | `IBPNet.cell`, fed the paper's per-step tuple |
| Imagination tree construction / route execution | `IBPNet.search` |
| Imagination `I: S x A -> S x R` (an interaction network in the paper) | the **frozen pretrained world model**, injected as `imagine` |
| Manager REINFORCE + entropy bonus | `src/rlp/train/ibp.py` |
| Resource loss (`tau` per imagination step) | `core.planner.tau` |
| Deployment | `IBPSolver` — `src/rlp/core/solver/ibp.py` |

Paper hyperparameters carried over: Adam, controller LR `3e-4`, manager LR
`1e-4`, gradient clipping at norm 10, manager entropy regularization. The
imagination budget (`max_imagine`) is the paper's swept quantity — it reports
1–6 imagination steps on the spaceship task and 4–8 on 7×7 mazes.

## What differs from the paper, and why

1. **The imagination is the frozen pretrained world model**, not an interaction
   network learned on-line from the agent's own experience. IBP's model loss
   therefore has no counterpart here — nothing about the model is trained. This
   is what makes an IBP row comparable to the RLP and DMPO rows: identical
   data, identical frozen critic, identical objective, differing only in the
   learned planning procedure. It is **not** a replication of the paper's
   spaceship or maze results.
2. **The controller is trained by pathwise gradients**, not by the paper's
   policy gradient, because the world model here is differentiable. Same
   deviation, same reason, as the DMPO port. The *manager* is still REINFORCE —
   its routes are discrete, and that is the mechanism the paper's claims rest
   on.
3. **Costs are the goal-conditioned quasimetric critic**, not a task reward.
   `tau` is denominated in those same critic units, so the performance/resource
   trade-off the paper studies is intact but its numeric scale is not the
   paper's (`2e-4`/`4e-4` fuel cost).
4. **One IBP action is one plan block**, so a root-to-leaf chain of `horizon`
   blocks is exactly the plan the evaluation protocol executes.
5. **Greedy completion.** The protocol executes a full 5-block plan open loop,
   so when the manager commits before its chain is full, the controller
   completes the plan greedily. Those blocks are *not* scored during the
   search — the one place the executed plan can be worse than the committed
   node, which `SearchResult.committed_cost` exposes and
   `tests/core/test_ibp.py` guards.
6. **Single-frame critics only.** A tree node is one imagined frame, so a
   window critic (Reacher's three-frame quasimetric) has no window to read at
   the interior nodes. Trainer and solver both reject one; Reacher is therefore
   out of scope for this port as written.

## Cost per decision

| planner | world-model unrolls per decision |
|---|---|
| CEM / MPPI (repo defaults) | 9,000 forward (300 samples × 30 iterations) |
| Adam | 3,000 forward + 3,000 backward |
| DMPO (defaults) | 256 forward |
| RLP / LIP | 9 forward + 8 backward |
| **IBP** (`max_imagine=10`) | **≤ 15 forward**, learned — one block each, no backward |

IBP's unrolls are one *block* each, where the others' are full `horizon`-block
plans, so the gap in world-model compute is larger than the row counts suggest.
The realized count is logged per decision by the solver and returned in
`SearchResult.imagined` / `.unrolls`.

## Commands

IBP trains against a frozen critic, so produce the caches and `value_td` first
(the RLP pipeline with the planner stage skipped), then train and evaluate:

```bash
pixi run train model=rlp skip=[planner] wm=assets/core/world_model/lewm_cube dataset=$RLP_DATA_HOME/datasets/ogb_cube_single.lance name=cube_lewm
```

```bash
pixi run train model=ibp wm=assets/core/world_model/lewm_cube cache=$RLP_DATA_HOME/caches/cube_lewm_fs5.pt h5=$RLP_DATA_HOME/caches/cube_lewm_actions.h5 init_value=logs/<date>/<time>/checkpoints/value_td core.planner.action_limit=1.6
```

```bash
pixi run eval model=lewm core/solver=ibp core.solver.actor_path=<ibp.pt>
```

The imagination budget is a deployment knob, as in the paper's sweeps — the
trained value is the default, and `core.solver.max_imagine=<n>` overrides it:

```bash
pixi run eval model=lewm core/solver=ibp core.solver.actor_path=<ibp.pt> core.solver.max_imagine=4
```

Whole campaign (caches → critic → three planner seeds → h25 and h100 on report
draws 42/43/44), one job per environment × base:

```bash
sky jobs launch scripts/sky/ibp_campaign.yaml -n ibp-cube-lewm --priority p1 --env ENVNAME=cube --env BASE=lewm --env AMAX=1.6 --env EXPERIMENT_TAG=ibp-cube-lewm-20260815 -y
```
