# amax (boundary clip) sweep — cube-single, v2WM, h25

## USE `--amax 1.6`

Campaign default for LIPv4 on cube. Replaces the historical `--amax 3.5`, which
costs ~7 points and triggers a catastrophic-seed failure mode.

`train_lip_ac.py`'s `--amax` **default is left at 2.5** on purpose — that file is
shared with the reacher/tworoom campaigns and silently changing a default would
alter their runs. Pass `--amax 1.6` explicitly on cube.

## The card — 3 seeds × 3 draws (42/43/44), 50 tasks each

| amax | s0 | s1 | s2 | **3-seed** | seed spread |
|---|---|---|---|---|---|
| 1.0 | 78.7 | 81.3 | 78.0 | 79.3 | 3.3 |
| 1.2 | 82.0 | 80.7 | 84.7 | 82.4 | 4.0 |
| 1.4 | 86.7 | 82.7 | 87.3 | 85.6 | 4.7 |
| **1.6** | 85.3 | 84.7 | 89.3 | **86.4** | 4.7 |
| 1.8 | 85.3 | 83.3 | 90.0 | 86.2 | 6.7 |
| 2.2 | 84.0 | 84.7 | 88.7 | 85.8 | 4.7 |
| 2.6 | 78.7 | — | 86.7 | — | |
| 3.0 | 67.3 | — | 70.7 | — | |
| 3.5 (old default) | 68.0 | 81.3 | 88.7 | 79.3 | **20.7** |

Raw cells: `results/summary_amaxsweep.csv` on the pod, keys `amsw_a<amax>_s<seed>_e<draw>`.
Drivers: `dyna_harness/amax_sweep.sh` (coarse), `amax_sweep2.sh` (fine 1.0–1.8).

## What it says

1. **Broad flat plateau 1.4–2.2** — 85.6 / 86.4 / 86.2 / 85.8, a 0.8-pt range across
   four values, i.e. statistically indistinguishable. Both tails fall off hard: 79.3 at
   1.0 (too tight, starves the actor) and 79.3 at 3.5 (too loose, lets the actor push
   into the WM's fabrication regime). Inverted-U confirmed on all three seeds.
2. **The real win is variance, not the mean.** Seed spread collapses 20.7 → 4.7. The
   catastrophic-seed mode (seed 0 = 68.0 at amax 3.5) disappears entirely. That is the
   result worth quoting.
3. **amax tuning is exhausted.** 79.3 → 86.4 = +7.1 pts, and it was already banked at
   2.2. No further compute on amax is justified.

### Why 1.6 rather than the held-out winner

Honest held-out selection (select on draw 42, report on 43+44 — see
`DATA_SPLIT_POLICY.md` §2, tool `dyna_harness/amax_holdout_select.sh`) picks **1.4** and
reports **86.3**. The naive best-all-3-draw picks 1.6 at 86.4. They nearly agree *because
the plateau is flat* — selection bias is small when candidates are equivalent. 1.6 is
chosen as mid-plateau: most robust to being slightly off, and it is not the endpoint of
the swept range in either direction.

Do not read the within-plateau ordering as signal. 1.4 sits below 1.6 because of a
**single cell** (seed 1, draw 44 = 68 vs 78 at both 1.6 and 1.8); one 50-task cell moves a
3-seed mean ~2 pts.

## Two caveats on these numbers

**Draw 43 is saturated and carries almost no signal.** Across the 15 fine cells:

| draw | mean | spread |
|---|---|---|
| 42 | 82.7 | 12 |
| 43 | **94.1** | **6** ← saturated |
| 44 | 75.2 | **24** ← does the discriminating |

So a 3-draw mean is roughly "draw 44 plus a constant". Any future cube card should
consider dropping 43 or adding harder draws instead.

**Everything here ran under `MUJOCO_GL=osmesa`, which may be out-of-domain.** The reacher
campaign found on 2026-07-27 that osmesa renders differ from the authors' h5 renders, and
switching to egl was worth **+7.3 pts** there (lejepa TD+CEM 84.0 → 91.3). Cube has not
been checked. If the same holds, the absolute numbers here are depressed and the optimum
could shift — the *relative* card is still internally consistent (verified: no pod code
changed during the 07:48–09:34 run). **Re-check 1.6 under egl before treating 86.4 as the
ceiling.** egl crashes under concurrent training, so run those evals on an idle pod.
