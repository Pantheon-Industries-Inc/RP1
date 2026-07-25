#!/usr/bin/env python3
"""Render RESULTS_tworoom_scores.md from results/summary_tworoom.csv."""
import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
rows = {}
for name, val in csv.reader((HERE / "results" / "summary_tworoom.csv").open()):
    try:
        rows[name] = float(val)
    except ValueError:
        pass

CELLS = [(surface, h) for surface in ("std", "hard") for h in ("h25", "h50")]
SEEDS = (42, 43, 44)


def card(prefix):
    """prefix -> {(surface,h,seed): score} using rows '<prefix>_<surface>_<h>_s<seed>'."""
    out = {}
    for surface, h in CELLS:
        for s in SEEDS:
            v = rows.get(f"{prefix}_{surface}_{h}_s{s}")
            if v is not None:
                out[(surface, h, s)] = v
    return out


def fmt(v):
    if v is None:
        return "—"
    return f"{v:g}"


def table(cards):
    """cards: list of (label, prefix). One row per planner+eval-seed? No:
    matrix rows = planner, columns = 4 cells x 3 seeds grouped."""
    lines = []
    header = "| planner | " + " | ".join(
        f"{surface} {h} s{s}" for surface, h in CELLS for s in SEEDS) + " | mean |"
    sep = "|" + "---|" * (len(CELLS) * len(SEEDS) + 2)
    lines += [header, sep]
    for label, prefix in cards:
        c = card(prefix)
        vals = [c.get((surface, h, s)) for surface, h in CELLS for s in SEEDS]
        got = [v for v in vals if v is not None]
        mean = f"{sum(got)/len(got):.1f}" if got else "—"
        lines.append("| " + label + " | " + " | ".join(fmt(v) for v in vals)
                     + f" | {mean} |")
    return "\n".join(lines)


def compact(cards):
    """per-cell means over eval seeds: 4 columns."""
    lines = ["| planner | std h25 | std h50 | hard h25 | hard h50 | overall |",
             "|---|---|---|---|---|---|"]
    for label, prefix in cards:
        c = card(prefix)
        cols, allv = [], []
        for surface, h in CELLS:
            vs = [c[(surface, h, s)] for s in SEEDS if (surface, h, s) in c]
            allv += vs
            cols.append(f"{sum(vs)/len(vs):.1f}" if vs else "—")
        overall = f"{sum(allv)/len(allv):.1f}" if allv else "—"
        # bold the perfect rows
        if allv and min(allv) == 100.0:
            cols = [f"**{x}**" for x in cols]
            overall = f"**{overall}**"
        lines.append(f"| {label} | " + " | ".join(cols) + f" | {overall} |")
    return "\n".join(lines)


PLANNERS_MAIN = [
    ("Random policy (floor)", "random"),
    ("Latent+CEM (300×30)", "latent"),
    ("LIP v1 full-input (old perfect actor, anchor)", "card_lv1ctrl"),
    ("min0 gated tandem `sched` s0", "card_m0sched-s0"),
    ("min0 gated tandem `sched` s1", "card_m0sched-s1"),
    ("min0 gated tandem `sched` s2", "card_m0sched-s2"),
    ("**LIPv4 final (amax 2.2, max-delta 12)** s0", "card_v4cmd12-s0"),
    ("**LIPv4 final (amax 2.2, max-delta 12)** s1", "card_v4cmd12-s1"),
    ("**LIPv4 final (amax 2.2, max-delta 12)** s2", "card_v4cmd12-s2"),
]

PLANNERS_SWEEP = [
    ("LIPv4 amax 2.0 s0", "card_v4a20-s0"),
    ("LIPv4 amax 2.0 s1", "card_v4a20-s1"),
    ("LIPv4 amax 2.0 s2", "card_v4a20-s2"),
    ("LIPv4 amax 2.5 s0", "card_v4a25-s0"),
    ("LIPv4 amax 2.5 s1", "card_v4a25-s1"),
    ("LIPv4 amax 2.5 s2", "card_v4a25-s2"),
    ("LIPv4 amax 3.0 s0", "card_v4a30-s0"),
    ("LIPv4 head-scale .01 s0", "card_v4hs-s0"),
    ("LIPv4 head-scale .01 s1", "card_v4hs-s1"),
    ("LIPv4 head-scale .01 s2", "card_v4hs-s2"),
    ("LIPv4 amax 1.8 s0", "card_v4bax18-s0"),
    ("LIPv4 amax 1.8 s1", "card_v4bax18-s1"),
    ("LIPv4 amax 1.8 s2", "card_v4bax18-s2"),
    ("LIPv4 amax 2.2 s0", "card_v4bax22-s0"),
    ("LIPv4 amax 2.2 s1", "card_v4bax22-s1"),
    ("LIPv4 amax 2.2 s2", "card_v4bax22-s2"),
    ("LIPv4 amax 2.2 s3", "card_v4cax22-s3"),
    ("LIPv4 amax 2.2 s4", "card_v4cax22-s4"),
    ("LIPv4 amax 2.2 s5", "card_v4cax22-s5"),
    ("LIPv4 a2.0 mean-weight .3 s0", "card_v4bmw03-s0"),
    ("LIPv4 a2.0 mean-weight .3 s1", "card_v4bmw03-s1"),
    ("LIPv4 a2.0 mean-weight .3 s2", "card_v4bmw03-s2"),
    ("LIPv4 a2.0 6k steps s0", "card_v4bst6k-s0"),
    ("LIPv4 a2.0 6k steps s1", "card_v4bst6k-s1"),
    ("LIPv4 a2.0 6k steps s2", "card_v4bst6k-s2"),
    ("LIPv4 a2.2 p-cross .5 s0", "card_v4cpc05-s0"),
    ("LIPv4 a2.2 p-cross .5 s1", "card_v4cpc05-s1"),
    ("LIPv4 a2.2 p-cross .5 s2", "card_v4cpc05-s2"),
    ("**LIPv4 a2.2 max-delta 12 s0**", "card_v4cmd12-s0"),
    ("**LIPv4 a2.2 max-delta 12 s1**", "card_v4cmd12-s1"),
    ("**LIPv4 a2.2 max-delta 12 s2**", "card_v4cmd12-s2"),
    ("min0 tandem ng s0 (no-gate, no schedules)", "card_m0ng-s0"),
]

out = f"""# TwoRoom scores — LIPv4 / min0 / Latent+CEM / baselines

**Date:** 2026-07-15 · **Protocol:** replay eval from `tworoom_play.lance`
(seed-7 regeneration), n=50 episodes/cell, success = within 16 px of goal.
Cells: surface ∈ {{std, hard(cross-wall)}} × horizon ∈ {{h25 (offset 25, budget
50), h50 (offset 50, budget 100)}} × eval seed ∈ {{42, 43, 44}} = 12 cells =
600 episodes per card. LIP evals: pure learned planner (no sampling),
`restarts=1` unless marked; Latent+CEM: CEM 300 samples × 30 iters on WM
rollouts with latent-L2 cost; random: uniform actions. World model (LeWM
ViT-tiny 192-d, frameskip-5) frozen and identical for all planners.

## 1. Headline: per-cell means over the 3 eval seeds

{compact(PLANNERS_MAIN)}

**LIPv4 final recipe = 100.0 on every cell for all three training seeds
(3600/3600 episodes), plain deploy** — single pass of the gate-free
minimal-input learned planner, `restarts=1`, no sampling, no test-time tricks.
Recipe: `--arch v4 --amax 2.2 --max-delta 12` + tandem warm-start + schedules
(§4 notes). The `max-delta 12` ingredient (HER goals to 60 primitive steps)
systematically removes the cross-wall wall-trap failures that made
single-seed perfection a coin flip in earlier variants.

## 2. Full per-cell matrix (every eval seed)

{table(PLANNERS_MAIN)}

## 3. LIPv4 sweep detail (plain deploy)

{table(PLANNERS_SWEEP)}

## 4. Notes

- **Latent+CEM** (same WM, no learned value): means 84.0 / 58.0 / 82.7 / 68.0
  — it plans competently at h25 but loses 26–42 points to the learned-value
  planners at h50, and needs ~9000 rollout-equivalents per replan vs ~16 for
  LIPv4 (~450×).
- **Random floor**: 20.7 / 10.0 / 13.3 / 2.0.
- The old campaign's dataset died with its pod; the seed-7 regeneration is
  statistically equivalent but re-rolls the eval task draws, so old-campaign
  numbers are not cell-for-cell comparable (see WRITEUP §1.4). The old perfect
  v1 actor (anchor row) scores 10×100 + 2×98 on the new draws.
- **LIPv4 final recipe**: `train_lip_ac.py --arch v4 --amax 2.2
  --max-delta 12` + tandem warm (`--init-value` TD τ0.1/n50) + schedules
  (τ 0.1→0.03, critic-lr 1e-3→1e-4 cos, actor-lr 3e-4→3e-5 cos), 8k steps,
  K=8, H=5. 340,530-param MLP, input `[A, ∇V, E]` (101-d), no gate, no raw
  latents. Actors: `trm_v4c_md12_s{0,1,2}.pt`.
- Sweep findings (§3): amax is monotone-helpful tighter (2.2-2.0 best,
  3.0 worst) — clipping substitutes for the gate's damping; head-scale
  small-init hurts the MLP; without max-delta 12 the per-seed perfect-card
  rate is ~40-60% (borderline wall-trap episodes), with it 3/3.
- All numbers here are the plain deploy: single pass of the learned planner,
  `restarts=1`, no sampling, no restart selection (deploy-time
  restart-selection variants were explored and rejected by directive; their
  rows remain in summary_tworoom.csv for the record).
- Failure forensics on `a20` s2's two 98-cells: both failing episodes are
  cross-wall wall-traps (greedy descent pins the agent against the wall at
  the goal's mirror position instead of detouring through the door), not
  16px near-misses — actor-specific, other seeds solve them.
- Full provenance: `WRITEUP_tworoom_min0.md`; raw per-cell data
  `results/summary_tworoom.csv`; actors + teacher values in `actors/`,
  `metrics/`.
"""

(HERE / "RESULTS_tworoom_scores.md").write_text(out)
print("wrote", HERE / "RESULTS_tworoom_scores.md")
