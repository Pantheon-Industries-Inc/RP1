# LeWM LIPv4 rollouts — pre-Dyna vs post-Dyna

Rendered 2026-07-30 on pod `157.66.255.80:12519` (4×H200), branch `eval-sweep`.
Source of the numbers these clips belong to: [`../RESULTS_dyna_20260729.md`](../RESULTS_dyna_20260729.md).

## What you are looking at

Both arms are LIPv4 actors planning through a LeWM world model on ogbench
`cube-single`, 224 px, `MUJOCO_GL=egl`, `amax 1.6`.

| arm | world model | actor |
|---|---|---|
| **PRE-Dyna** | `v2WM` (frozen, `weights_epoch_22.pt`) | `lip4_dsp_pre_s{seed}.pt` |
| **POST-Dyna** | `dyna_full_5050` (v2WM fine-tuned on a 50/50 expert : on-policy mix, φ=0.08, K≈18) | `lip4_rep_full_s{seed}.pt` |

Two horizons:

- **`h25/`** — goal = frame t+25 of the demo, 50-step budget. This is the
  protocol the Dyna result was measured under.
- **`h100/`** — goal = frame t+100, 200-step budget. **Out of distribution for
  both arms**: they were trained and tuned at the 25-step offset. Included
  because long-horizon behaviour is worth seeing, not as a re-test of the
  finding.

## Dataset split

**Episode-disjoint, enforced at eval time.** Every stage that produced these
policies — latent caches, TD teacher, LIP actor-critic, Dyna collection, WM
fine-tune — used episodes **0–7999** only. These evals draw exclusively from
episodes **8000–9999** via `+eval.ep_range=8000:10000`, and the driver aborts
the cell unless the log confirms the filter applied. Start state and goal are
both held out.

One residual leak, unchanged from the campaign: **v2WM itself was pretrained on
all 10,000 episodes**, so the base representation has seen these frames. The
Dyna delta is still clean — both arms share that base — but absolute success
rates are upper bounds, not generalization estimates. See
[`../DATA_SPLIT_POLICY.md`](../DATA_SPLIT_POLICY.md).

## Results

| horizon | cell | PRE | POST | Δ |
|---|---|---|---|---|
| h25 | seed 2 / draw 44 | 80.0 | 92.0 | +12.0 |
| h25 | seed 1 / draw 42 | 90.0 | 98.0 | +8.0 |
| h100 | seed 2 / draw 44 | 82.0 | 90.0 | +8.0 |
| h100 | seed 1 / draw 42 | 78.0 | 88.0 | +10.0 |

Both h25 cells reproduced their campaign numbers **exactly**, per task and not
merely in aggregate — the same six/four task indices flip. h100 is new
measurement; note these are single cells at n=50, where the paired SE is ~4–5
points, so the h100 deltas are indicative, not established.

**h100 does not collapse.** The campaign's earlier long-horizon capture (h200,
`goal_offset 200 / budget 400`) was labelled "long-horizon collapse" with all 8
tasks failing; that does not extend down to h100, where both arms stay near
their h25 rates.

## Files

One folder per horizon and cell. Four files per task, no composited panels and
**no text burned into any frame** — everything identifying a clip is in its
filename:

- `taskNN_pre-dyna_<OUTCOME>.mp4` — PRE-Dyna rollout
- `taskNN_post-dyna_<OUTCOME>.mp4` — POST-Dyna rollout, same start state and goal
- `taskNN_expert-demo.mp4` — the demonstration the goal was taken from
- `taskNN_goal.png` — the goal frame itself

224 px, 10 fps, final frame held ~1 s (h25 = 60 frames, h100 = 210). Per-cell
outcome tables are in each folder's `INDEX.md`.

## Which clips to watch

The arms differ by a few points, i.e. a handful of tasks in 50 — most look
identical. Each folder holds the tasks that actually flip plus controls, so the
selection is visible rather than cherry-picked: **cures** (PRE fails, POST
succeeds), **both-succeed**, and **both-fail** (what Dyna did *not* fix).

Tasks were ranked by two measures taken from the demo's cube trajectory
(`privileged_block_0_pos`), computed by `scripts/task_stats.py`:

- **disp** — ‖cube(goal) − cube(start)‖, how far the cube must travel
- **lift** — max z − z(start), whether the cube leaves the table. A lift of
  ~0.25–0.31 m is a genuine grasp-carry-place; ~0 m is a push/slide.

### Longest-horizon cures at h100

**seed 1 / draw 42** — six cures, and the longest-horizon cure anywhere in this
set:

| task | disp | lift | |
|---|---|---|---|
| **22** | **0.546 m** | 0.28 m | **longest cure overall, and a full pick-and-place** |
| 48 | 0.435 m | 0.22 m | long carry, cured |
| 18 | 0.265 m | 0.19 m | |
| 33 / 46 / 42 | — | — | remaining cures |

**seed 2 / draw 44** — four cures:

| task | disp | lift | |
|---|---|---|---|
| 41 | 0.332 m | 0.14 m | longest cure in this cell |
| 45 | 0.305 m | 0.26 m | full pick-and-place |
| 29 | 0.288 m | 0.02 m | long push/slide |
| 31 | 0.264 m | 0.28 m | full pick-and-place |

Included alongside them: the biggest both-succeed pick-and-places (seed 2 task
48 at 0.55 m; 42, 22, 14; seed 1 tasks 5, 47, 40, 10), and the long tasks
**both** arms still fail — seed 1's 34 (0.444 m), 35 (0.417 m), 41 (0.386 m),
and seed 2's 43, 8, 12, which Dyna cures at h25 but neither arm solves at h100.

At h25 the largest-travel cures are 43 (0.324 m, lift 0.297) and 12 (0.274 m,
lift 0.309) — both full pick-and-places.

Success requires the cube to be at the goal **at the end** of the episode, not
merely to have been touched along the way; episodes run the full budget, so
watch the final frames.

## Reproducing

`scripts/` holds everything used here:

- `vidsetup.sh` — re-bootstraps the pinned stack on a recycled container (75 s).
  Afterwards install the swm `env` extras, then re-pin
  `"numpy<2" torch==2.4.1 torchvision==0.19.1 --index-url .../cu124`, or torch
  gets dragged to 2.13 and leftover `nvidia-*-cu13` wheels shadow cuDNN
  (`CUDNN_STATUS_NOT_INITIALIZED`; `torch.backends.cudnn.version()` must read
  90100, not 92000).
- `dynavid_cmp.sh` / `dynavid_h100.sh` — run the eval cells with `+video_dir`,
  sequentially, aborting any cell whose log does not confirm
  `ep_range 8000:10000`. ~150 s per h25 cell, ~215 s per h100 cell.
- `pick_tasks.py` — diffs two arms' `episode_successes` into gains,
  regressions and controls.
- `task_stats.py` — ranks tasks by cube displacement and lift.
- `export_individual.py` — splits `env_{i}.mp4` back into the four per-task
  streams. Note it takes the panel size (224) rather than inverting the canvas
  arithmetic: that rounds up to a multiple of 16 and so does **not** invert
  uniquely (222–225 all yield a 736×288 canvas), and searching silently crops
  a few px off every frame.
