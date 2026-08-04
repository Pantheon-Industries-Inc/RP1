# Reacher pod source snapshot — 2026-08-04

A verbatim copy of the patched sources that produced the Reacher §5.2 numbers.
Taken from the live reacher pod (`212.247.220.107:19243`), `/workspace`, at
2026-08-04 20:15 UTC.

**Why this exists.** Every handoff since 2026-07-30 has carried the same standing
risk: *"the patched source lives only on pod volumes. The repo's
`stable-worldmodel/` has no `_wpair`, no `pad_context`, and no
`train_window.py`. Losing both pods loses the trainers."* This snapshot closes
that. It is a **snapshot, not a merge** — it is deliberately parked in a
campaign-scoped directory rather than applied over the shared
`stable-worldmodel/` tree, because parallel sessions share this branch and
several of them have their own edits in flight there.

## Layout

```
stable_worldmodel/          the patched package (solver/, trm/, wm/, policy.py, world/)
scripts/                    scripts/plan/* — trainers, eval_wm.py, hydra configs
pyproject.toml              the package metadata the pod pip-installed with -e
workspace_drivers/          the 95 loose drivers and patch scripts at /workspace root,
                            including train_window.py, which is NOT inside the package
```

`workspace_drivers/train_window.py` is the 3-frame window quasimetric trainer —
the shared value behind Value+CEM, Value+Adam, RLP and PWM. It has never lived
in the package and is the single most load-bearing file here.

## What is in this snapshot that the repo tree does not have

- `_wpair` / window-pair handling in the metric trainers
- `--pad-context` in `train_lip_ac.py`
- `train_window.py` at all
- `--align-mode {terminal,dense,prefix,randhorizon}` in `train_lip_ac.py`
  (`workspace_drivers/patch_align_modes.py`)
- I3, the deadline-aware plan-cost readout, in **both** cost paths (below)

## I3, and the bug this snapshot fixes

I3 scores the imagined chunk that lands on the graded step instead of the plan
terminal, for the `receding_horizon=1` ablation where a 5-chunk plan overshoots
a 50-step budget. Two independent cost paths need it, which is what the previous
three days of edits missed:

| arm family | cost path | window stacking |
|---|---|---|
| Latent/Value + CEM/Adam | `_MetricCost.get_cost` in `scripts/plan/eval_wm.py` | the `_m >= 3` branch |
| RLP (`solver=lip`) | `LIPSolver._V` in `stable_worldmodel/solver/lip.py` | `window_pair` |

**LIPSolver never calls `model.get_cost`.** It scores with `self.lip_value`, the
value embedded in the actor checkpoint. Every earlier I3 edit went into
`_MetricCost`, so I3 was dead in exactly the arm it was written for — and a dead
I3 returns the unaligned baseline verbatim, which reads as a legitimate result.
The diagnostic that settled it: an *unconditional* print at the entry of
`_MetricCost.get_cost` produces no output in a LIP cell (`logs/i3dbg.log`).

Two further defects fixed in the same pass:

- **Frame axis.** `predicted_emb` is `T = off + P` frames, not `P`: the WM
  re-predicts its conditioning history, so with 3 history frames and a 5-block
  plan the axis is 6 long and the plan chunks are the *last* 5. The old guard
  compared `chunks_remaining` against `T`, which fires I3 one replan early —
  inside the reported rh=5 protocol. It now compares against `P` and offsets
  every index by `T - P`. `LIPSolver._V` needs no offset; its trajectory is the
  plan exactly.
- **Goal broadcast.** The prefix stack prepends an axis to `pred`; the generic
  `goal.unsqueeze(1)` below then produced `(J,1,B,D)` against `(J,B,S,D)` and
  every Latent+CEM cell crashed.

`I3_MODE` selects the readout — `deadline` (the chunk landing on the graded
step, default) or `min` (the best still-reachable chunk). Both are inert without
`plan_config.deadline`, and inert while `chunks_remaining >= P`.

**Inertness is verified, not asserted.** The reported rh=5 cell (RLP lejepa s0,
env seed 42) scores 66.0 @0.05 / 100.0 @0.1 with a deadline set and prints no
`[I3]` line — identical to the committed number. No reported figure can move.
