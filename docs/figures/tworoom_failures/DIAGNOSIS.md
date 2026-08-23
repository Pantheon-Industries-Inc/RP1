# TwoRoom h100 seed lottery — mechanism diagnosis

**TL;DR.** The failing PLDM deep-capacity seeds are not victims of a corrupted value
function, a wall-blind critic, door-gradient reversal, or unstable refinement. Their
**actors converged to a WM-hallucination basin**: they emit near-constant plans
(~80% of each plan's *true* net motion is one fixed direction, independent of state
and goal) whose real-world effect ignores the goal, while the frozen world model
imagines exactly those plans arriving at the goal (imagined-direction-vs-geodesic
cosine 0.86–0.91, *higher* than the good seeds'). The training loss — imagined E
through the frozen WM — cannot distinguish this basin from genuine planning, so which
basin a seed lands in is decided at initialization: a training-level lottery the
co-trained critic neither causes nor can repair. Attribution is behavioral and
decisive: **good actor × catastrophic seed's critic = 100%**, while
**catastrophic actor × good seed's critic = 19%**.

## Checkpoints diagnosed (from W&B, project `armin-sommer/RLP`, artifact type `lip_actor`)

| stack | run id | h100 eval (mean of task seeds 42/43/44) | h25 | role |
|---|---|---|---|---|
| PLDM s0 | `fbysy8bu` | 96.7 | 94.0 | healthy |
| PLDM s2 | `eksvgwk7` | 50.7 | 88.7 | cratered |
| PLDM s4 | `u11htbrm` | 26.7 | 78.0 | catastrophic |
| PLDM s5 | `lq96657a` | 100.0 | 96.7 | healthy |
| LeWM s0 | `ugbv30xb` | 93.3 | 100.0 | weak |
| LeWM s5 | `n3pg6e72` | 99.3 | 100.0 | healthy |

All `rlp-tw-unig-*-g98deepcap-*-20260822` (gamma 0.98, amax 2.5, K=10, v4_layers 3,
critic depth 3, vnorm none, squash hard). Each artifact carries the actor and its
deployed co-trained critic (EMA teacher). WMs are the frozen tracked checkpoints in
`assets/core/world_model/{pldm,lejepa}_tworoom`.

Full deep-capacity h100 sweep for context (not all diagnosed): PLDM
s0–s5 = 96.7 / 90.0 / **50.7** / 94.0 / **26.7** / 100.0 — 2 of 6 seeds
catastrophic; LeWM s0–s5 = **93.3** / 99.3 / 98.0 / 96.7 / 98.7 / 99.3 — one
mildly weak seed.

## Local replication harness

Everything below runs CPU-only from the repo assets plus the W&B checkpoints,
using the `scripts/figures/tworoom_rlp/` machinery (ridge probe latent→(x,y),
R² 0.99; cached value grids). 26 h100 instances (20 cross-room + 6 same-room
controls) were drawn exactly like the eval builds them: goal = state 100 primitive
steps ahead along a dataset-policy route (ExpertPolicy noise 2.0, repeat 0.05).
Closed loop replicates deployment: budget 200, replan when the 25-primitive-action
buffer empties (`receding_horizon: 5` blocks — **not** 5 primitive steps),
history_len 1 (current frame tiled ×3), zero action history, A(0)=0, K=10, last
iterate, success = within 16 px.

The harness reproduces the seed lottery quantitatively (cross-room instances are
the hard family, so bad seeds sit below their full-eval scores, which average over
many easy near-goal draws):

| stack | closed-loop total | cross-room | same-room | real eval h100 |
|---|---|---|---|---|
| PLDM s5 | 96% | 95% | 100% | 100.0 |
| PLDM s0 | 88% | 85% | 100% | 96.7 |
| PLDM s2 | 54% | 45% | 83% | 50.7 |
| PLDM s4 | 12% | 0% | 50% | 26.7 |
| LeWM s5 | 100% | 100% | 100% | 99.3 |
| LeWM s0 | 92% | 90% | 100% | 93.3 |

(A first harness draft replanned every 5 primitive steps; that collapses even the
good seed to 23% — worth knowing: the deployed cadence of executing the *full*
25-step plan between replans is load-bearing.)

## Hypotheses tested

### 1. Value landscape — CLEAN for every seed (null result)

`value_landscape_pldm_seeds.png`. Each seed's *deployed co-trained critic* evaluated
on the room lattice toward fixed cross-room goals, against the true BFS geodesic:

- Spearman rho(V, geodesic) over free cells: **0.983–0.992 for all four PLDM seeds**
  (goal in left room and right room alike); LeWM seeds identical (0.988–0.992).
  The offline assets teacher scores the same (0.985–0.988).
- Source-room restriction (the room without the goal): rho 0.979–0.990 for all
  seeds — the door funnel is present and correctly oriented in every critic.
- No wall-blind shortcuts: where the goal sits deep in the other room, correlation
  with straight-line Euclidean distance goes *negative* for every critic (as it
  must, since the true route detours through the door).

The bad seeds' critics are indistinguishable from the good seeds' on real states.
The gamma=0.98 discount compresses the range (V saturates toward 50 = 1/(1−gamma))
identically across seeds.

### 2. Refinement dynamics — no instability (null result)

`refinement_dynamics_pldm.png`. On the first replan of the 20 cross-room tasks
(identical start states for all seeds): E descends 31→4–7 for every seed; rms
grad_A V, the fraction of plan entries at the amax=2.5 clip (0.12–0.24 at k=10),
and the cosine of successive updates are statistically indistinguishable between
good and bad seeds. No oscillation, no clip saturation blow-up. The bad actors
optimize the imagined objective just as well as the good ones — *that is exactly
the problem*.

### 3. What actually differs — imagination-faithfulness of the plan family

Per replan on cross-room tasks (all replans; `mechanism_pldm.png`):

| stack (h100) | \|imag disp\| px | \|exec disp\| px | cos(exec, imag) | cos(exec, geodesic) | cos(imag, geodesic) | cos(planΣ, geodesic) | plan constancy |
|---|---|---|---|---|---|---|---|
| PLDM s5 (100) | 87 | 43 | **0.55** | **0.71** | 0.67 | **+0.68** | 0.39 |
| PLDM s0 (97) | 91 | 50 | **0.50** | **0.53** | 0.63 | **+0.51** | 0.26 |
| PLDM s2 (51) | 111 | 28 | 0.24 | 0.13 | **0.86** | **−0.46** | **0.77** |
| PLDM s4 (27) | 124 | 32 | 0.12 | 0.01 | **0.91** | **−0.57** | **0.80** |
| LeWM s0 (93) | 94 | 46 | 0.65 | 0.69 | 0.76 | +0.54 | 0.24 |
| LeWM s5 (99) | 83 | 52 | 0.85 | 0.83 | 0.75 | +0.80 | 0.28 |

planΣ = the plan's *true* net displacement (un-z-scored, env-clipped primitive
actions summed); plan constancy = |mean planΣ vector| / mean |planΣ| over all
replans, states and goals.

Reading:

- **Everyone hallucinates magnitude.** All seeds' imagined 25-step displacement is
  ~2–4× what the env permits (the WM extrapolates: plans at amax=2.5 z-units are
  1.9× beyond the largest action z-value in the data, and even in-distribution
  blocks imagine ~1.5–2× the real step). Probe-decoded imagined paths routinely
  cut across the wall — a known property of this stack (see the tworoom_rlp README).
- **Good seeds hallucinate *along the plan*.** Their plans' true net direction
  agrees with the geodesic (+0.51/+0.68) and execution follows imagination
  (0.50/0.55). Replanning every 25 steps then walks the agent through the door.
- **Bad seeds emit a near-constant adversarial pattern.** 77–80% of the plan's
  true net motion is one fixed vector (s4: down-left; s2: down-right — visible as
  the corner-seeking rust paths in `h100_paths_pldm_s4_vs_s5.png`), *independent of
  state and goal*. The true net direction points **away** from the goal
  (cos −0.46/−0.57) while the WM imagines the same plans arriving **at** the goal
  (cos +0.86/+0.91, imag endpoint within 16 px of the goal claimed in a minority of
  replans but E driven to 4–8 regardless). The small state/goal-dependent residual
  of the plan steers *the hallucination*, not the agent.
- Failure taxonomy (`failure_taxonomy.png`, 20 cross-room tasks): s4 = 18/20
  "stuck in source room" + 2/20 near-goal miss; s2 = 10/11 "stuck in source room";
  the agent *moves* 40–190 px but never approaches the door (min door distance
  35–150 px), and never presses into the wall (wall-contact fraction 0.00 — this is
  not hypothesis (ii) wall-blindness). The good seeds' rare failures are door-stalls
  or near-goal misses.
- LeWM's weak seed s0 is the same axis, milder: cos(exec, imag) 0.65 vs s5's 0.85.
  Its two failures are wandering episodes of the same kind.

### 4. Attribution — the actor owns the failure, the critic is exonerated

Closed-loop cross-swap (same WM, same tasks, deploy cadence), PLDM:

| actor \ critic | s4 critic | s5 critic | s2 critic | s0 critic |
|---|---|---|---|---|
| **s4 actor** | 12% (cross 0%) | 19% (cross 10%) | — | — |
| **s5 actor** | **100% (cross 100%)** | 96% (cross 95%) | — | — |
| **s2 actor** | — | — | 54% (cross 45%) | 38% (cross 40%) |
| **s0 actor** | — | — | **88% (cross 85%)** | 88% (cross 85%) |

(Diagonal cells reproduce the single-seed matrix exactly — the harness is
deterministic. The mild asymmetry for s2 — 54% with its own critic vs 38% with
s0's — is the expected co-adaptation of an actor to its own teacher's E scale;
the healthy actors are critic-agnostic: swapping either bad critic under s5/s0
changes nothing, 100/96 and 88/88.)

The catastrophic seed's critic supports **perfect** planning under a good actor;
the good critic cannot rescue the bad actor. The pathology is stored in the actor
weights f_theta.

Off-manifold corroboration: on 128 random-plan WM-imagined terminals per task
(8 cross-room tasks), all four critics agree with each other about equally
(pairwise Spearman 0.29–0.97, value-gradient cosine through the WM 0.45–0.88;
the s4~s5 agreement is as high as s0~s5), and *no* critic tracks the decoded true
position off-manifold (rho vs geodesic-of-decode spans −0.47…+0.90, task-dependent)
— imagined terminals are simply outside everyone's trained domain. There is no
differential critic corruption to find.

## Mechanism statement

Training optimizes E = V(z_T(A), z_g) purely through the frozen WM — no real
rollouts. Through this objective there exist (at least) two families of minima on
h100-scale tasks:

1. **Faithful family**: plans inside the WM's semi-faithful action envelope, whose
   imagined direction ≈ true direction. These execute toward the door; receding-
   horizon replanning does the rest.
2. **Hallucination family**: high-amplitude, nearly state-independent action
   patterns that act as "teleport keys" against the frozen WM — the WM imagines
   goal-ward arrival (and the geodesic-faithful critic therefore scores them ~as
   well or better), but their real net motion is a constant goal-independent drift.

Both are global-quality minima of the *training* objective; only deployment
separates them. Which family f_theta converges to is decided by the training seed —
hence: (a) failures are training-level (val mediocre at every snapshot: the actor
sits in the wrong basin from early on), (b) the co-trained critic is innocent,
(c) h25 is partially spared (the goal is one true plan-horizon away, so the
faithful family achieves genuinely low E and the two basins compete; s4 still
leaks to 78), while h100 is fully exposed (no physical 25-step plan can reach a
4-horizon goal, so the hallucinated minima dominate the imagined objective),
(d) deep capacity (v4_layers=3) worsens the lottery — a more expressive f_theta
finds the teleport keys more reliably (two of six seeds catastrophic vs the
depth-2 baseline's occasional mild weak seed).

## What would fix it (implications, not yet run)

- The failure is detectable *without* any environment rollout: plan constancy
  across (state, goal) pairs and cos(planΣ-direction, imagined-direction) separate
  bad from good seeds at a glance (0.77–0.80 vs 0.26–0.39; −0.3 vs +0.5). Either
  makes a cheap training-time canary or seed-selection statistic.
- Anything that pins imagination to reality on the actor's own plans should close
  the basin: a real-dynamics consistency term on the first block, or critic
  training on planner-visited imagined states (expand-weight was 0 here). Note
  that raw amplitude/clip statistics do NOT separate the basins (first-replan
  plan-rms and beyond-data-z fraction are the same for s5 and s2), so a plain
  amplitude penalty attacks the shared extrapolation regime, not the bad basin
  specifically — gate on the plan-constancy / direction-agreement canaries instead.
  Restart-based deployment cannot help — the basin is in the weights, not the
  init (every replan already starts from A=0).

## Figures

- `value_landscape_pldm_seeds.png` — geodesic vs each seed's deployed critic, two goals (null #1).
- `refinement_dynamics_pldm.png` — E / grad / clip / update-cosine per iteration (null #2).
- `h100_paths_pldm_s4_vs_s5.png`, `h100_paths_pldm_s2_vs_s0.png`, `h100_paths_lejepa_s0_vs_s5.png` — executed closed-loop paths, all 20 cross-room tasks: bad-seed corner-seeking vs good-seed door routing.
- `filmstrip_pldm_task9_s4_vs_s5.png`, `filmstrip_pldm_task5_s4_vs_s5.png`, `filmstrip_pldm_task9_s2_vs_s0.png` — per-replan probe-decoded imagined plans (rust) over the executed path (grey): the WM imagines goal-ward teleports while the bad seed's agent drifts to the corner.
- `mechanism_pldm.png` — the quantitative mechanism: direction-agreement bars, plan constancy, and the true net-displacement clouds of every refined plan.
- `failure_taxonomy.png` — failure classes per stack on the 20 cross-room tasks.

## Provenance / load paths

- Actor+critic pairs: W&B artifacts `g_tworoom_{pldm,lejepa}_unig_ctrl_a2.5_s{n}` (type `lip_actor`, runs listed above); local copies under the session scratchpad `tworoom_diag/ckpts/`.
- World models: `assets/core/world_model/pldm_tworoom`, `assets/core/world_model/lejepa_tworoom`.
- Offline teacher reference: `assets/core/planner/tworoom/{pldm,lejepa}/*/s0/g_*_value`.
- Grid latents: `scripts/figures/tworoom_rlp/_latgrid_{pldm,lejepa}.npz`; probe refit locally (R² 0.992/0.998).
- Action un-z-scoring: locally reconstructed dataset stats (mean≈0, std≈0.744 per component, the ×1.4 filmstrip calibration removed); closed-loop results are insensitive to this scale (96%→100% for s5, 12%→15% for s4 between std×1.0 and ×1.4).
- Analysis scripts (session scratchpad, `tworoom_diag/`): `diag_core.py` (checkpoint loaders incl. deep-capacity PlannerNet), `s1_tasks.py`, `s2_landscape.py`, `s3_closedloop.py`, `s4_crossswap.py`, `s5_offmanifold.py`, `s6_taxonomy.py`, `s7_direction.py`, `s8_wmresponse.py`, `f1`–`f5` figure scripts.

## Caveats

- Task instances are locally drawn (noisy-expert routes, cross-room-enriched), not
  the authors' `tworoom.h5` rows; the real eval also samples many easy near-goal
  instances, which is why bad seeds score higher there than on this hard set.
- The action StandardScaler is locally reconstructed (std 0.744); the two scales
  tested bracket it with no qualitative change.
- Executed-path collision dynamics use the real env; frames are rendered without
  the target dot, exactly as the existing tworoom_rlp figure pipeline does.
