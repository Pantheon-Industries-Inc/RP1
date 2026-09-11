# TwoRoom · RLP figures

Six figures (PNG + SVG each), rendered from the **real** tracked checkpoints in
`assets/` (LeWM/LeJEPA + PLDM world models, the `lip4` `fθ` planner, and the
trained value functions), run locally on CPU. No prose titles — panel labels
only, matching the paper's Fig-1. Palette = paper tex tones (rust `#c0491e`
plan/latent, steel-blue `#2f6f9f` learned, teal goal ring, gold star, red
origin, slate barrier). Heatmaps use a blue→grey cost-to-go scale (near = deep steel blue, far = pale grey); the goal ring in the filmstrips is a shaded disc drawn behind the plan so the path stays legible on top.

## Protocol (horizon / offset / success)

Every instance follows the eval contract in `configs/eval/_world_model.yaml`:

- **goal = state `goal_offset_steps = 25` primitive steps ahead of the origin**
  along an expert route — origin and goal are 25 steps apart, i.e. exactly **one
  plan horizon** (`horizon 5` blocks × `action_block 5`), so a single refined
  plan can reach it.
- **success = final agent within 16 px** of the goal (`TwoRoomEnv: terminated = dist < 16`).
- Instances are drawn under the **report seeds 42 / 43 / 44** and selected so the
  **executed** plan (see below) routes through the door and reaches the goal:

  | task seed | origin → goal | offset | executed end-dist | door crossing (y∈[35,63]) |
  |-----------|---------------|--------|-------------------|---------------------------|
  | 42 | (168,20) → (82,109) | 25 | 11.9 px ✓ | 45.0 |
  | 43 | (167,75) → (87,109) | 25 | 6.4 px ✓ | 54.8 |
  | 44 | (172,112) → (77,33) | 25 | 1.1 px ✓ | 45.2 |

## Plan-refinement time-series (3 filmstrips, no header)

`refine_seed{42,43,44}.png` — one row = refinement iterations 0,1,2,3,4,6,8. The
orange curve is the plan at that iteration **executed through the env's true
dynamics** (`agent += action·speed` with wall collisions), so it physically
routes *through the door* and never crosses a wall. Plan starts as a stub at the
agent (A₀=0) and, as `fθ` refines A₀→A₈, becomes a full door-routed path onto the
goal (LeWM base, actor seed 0). Earlier drafts drew the WM's *imagined* latent
path decoded by a linear probe — that cut across the wall (unphysical) and is
replaced here by the executed path.

**Action scale.** Executing a plan needs the action StandardScaler (z-scoring)
from the authors' `tworoom.h5`, which is not cached locally. It is reconstructed
by collecting the same mixed expert+random policy locally (mean≈0, std≈0.75) and
calibrating the scale to `×1.4` (effective std ≈ 1.05) so executed step lengths
match the expert's (median 5 px = full speed). At that scale single-shot success
over cross-room offset-25 draws is ~63%; the three shown are selected successes.

## Optimizer comparison — same value, same tasks (Adam / CEM / RLP fθ)

`refine_{adam,cem}_seed{42,43,44}.png` optimize the **same** learned terminal
value V(z_T(A), z_g) on the **same** tasks as the RLP filmstrips, from the same
A₀=0 init — so the only difference is the update rule:

- **RLP `fθ`** (`refine_seed*`): learned optimizer, **8** iterations. Executed end 12.0 / 6.6 / 1.3 px — all reach.
- **Adam** (AdamW lr 0.1, **30** steps, single start): 13.2 / 61.4 / 7.0 px — reaches 42 & 44 but **stalls in a local minimum on 43** (value only →12); the deployed solver's 100-restart variant would escape it.
- **CEM** (300 samples, top-30, **30** iters): drives the value lowest (→0.3) but the plans are jagged and executed ends land 20.2 / 18.8 / 19.8 px — value-optimal, execution-short (CEM exploits WM/value error more than gradient methods). The absolute shortfall is near the 16 px tolerance and depends on the local action-scale calibration; the qualitative pattern is the point.

Takeaway: the learned `fθ` reaches every task in 8 iterations where Adam needs
~30 (and can stall) and CEM converges fast in value but not in execution.

## Adam-stall analysis — `adam_value_landscape.png`

Why Adam fails on task seed 43, in the **original viridis** scale. The optimizer
descends the plan objective **J(A) = V(z_T(A), z_g)**; the figure shows a 2D slice
of that objective in plan space (plane spanned by the Adam- and RLP-solution
directions), plus 1D profiles and the convergence curves.

- **Landscape (left):** A₀ (V=35) sits on a ridge; the goal basin (RLP `fθ`, V=0.3)
  is deep but offset in a direction Adam never explores. Adam slides sideways into a
  **shallow basin (V=12.3)** and oscillates there.
- **Barrier (middle):** along the straight A₀→RLP line the value dips, then **rises
  to a ridge (~37) before dropping into the goal basin** — a barrier a local method
  can't cross. (The Adam→RLP line is monotone downhill, but Adam's gradient dynamics
  never turn toward it.)
- **Convergence (right):** Adam oscillates and plateaus at ~12–18 over 30 steps; the
  learned `fθ` reaches 0.3 in 8. This non-convexity is exactly what the learned
  optimizer is trained to navigate.

Reproduce: `python scripts/figures/tworoom_rlp/analyze_adam.py && python scripts/figures/tworoom_rlp/fig_adamland.py`

`adam_value_room.png` renders that same value objective as the **cost-to-go field over room positions** (viridis) with the executed Adam and RLP plans overlaid: both cross the door, but Adam's plan halts in the upper-left where V is still ≈12, while `fθ` continues down to the goal (V≈0.3).

## Cost-to-go heatmaps (3 figures, paper Fig-1 style)

For a fixed goal ★, every free cell is rendered→encoded→scored: **latent** =
‖z − z_goal‖, **learned** = V(z, z_goal), **geodesic** = BFS shortest path.
Each panel min–max normalised near→far; barrier is a vector overlay.

- `costtogo_main.png` — PLDM [latent|learned], LeWM [latent|learned], True [geodesic]. Task seed 42, actor seed 0.
- `costtogo_task_seeds.png` — rows = task seeds 42/43/44; cols = LeWM latent | LeWM learned | geodesic.
- `costtogo_actor_seeds.png` — task seed 42; geodesic vs LeWM learned for actor seeds 0/1/2.

### Correlation with the true geodesic (free cells)

| task seed | learned ~ geodesic | latent ~ geodesic |
|-----------|--------------------|-------------------|
| 42 | **0.985** | 0.085 |
| 43 | **0.983** | 0.045 |
| 44 | **0.990** | 0.426 |

Actor seeds 0/1/2 (task seed 42): learned~geodesic = 0.985 / 0.985 / 0.986.
The learned value recovers the geodesic (~0.98); raw latent distance is wall-blind.

## Caveats (honesty)

- Task instances come from `env.reset` draws keyed to seeds 42/43/44, not the
  authors' `tworoom.h5` episodes (not cached locally); models, geometry and
  dynamics are the real artifacts, only the task *sampling* and the action-scale
  calibration are local.
- A single open-loop 25-step plan is the tight case; the real eval gets budget 50
  with receding-horizon replanning. The shown instances are selected single-shot
  successes.

## Reproduce

```bash
python scripts/figures/tworoom_rlp/grids.py lejepa 5
python scripts/figures/tworoom_rlp/grids.py pldm 5
python scripts/figures/tworoom_rlp/build_data.py   # offset-25 seed 42/43/44 instances + executed paths
python scripts/figures/tworoom_rlp/fig_refine.py
python scripts/figures/tworoom_rlp/fig_ctg.py
```
