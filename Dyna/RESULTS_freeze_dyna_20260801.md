# Critic-freeze ladder, expand dose-response, and per-base Dyna

>**SUPERSEDED as the reference config (2026-08-03).** The numbers here are at
>`expand-weight 1.0`. The current SOTA is **`expand-weight 3.0`** — LeWM 90.00,
>PLDM 84.00 — recorded in [`HANDOFF_20260728.md`](HANDOFF_20260728.md#sota--expand-weight-30-current-best-config-2026-08-03).
>This document remains valid for the Dyna deltas and the sweep history, which
>were measured at `expand 1.0` and have not been repeated at 3.0.


**2026-08-01, H200 pod (8×), EGL throughout.** Held-out evaluation on episodes
`8000:10000` for every number below; collection and all training on `0:7999`.
3 training seeds × 3 eval draws (42/43/44), 50 tasks per draw.

Headline: **PLDM closed the gap to LeWM.** 72.8 → 91.33 across two
interventions, against LeWM's 92.00. The long-standing framing of this campaign
— LeWM strong, PLDM the weak base, ~89.6 the ceiling — no longer holds.

---

## 1. Why this ran: an unnoticed config difference

The LeWM transfer test scored its baseline **89.56** against a historical anchor
of **87.8** (3 seeds) / 87.1 (6 seeds). Diffing the two training commands, one
flag differed:

| | historical `dsp_pre` | transfer-test `lewmbase` |
|---|---|---|
| `--freeze-critic-frac` | *omitted* → default **0.8** | explicitly **0.05** |
| critic freezes at step | 4800 of 6000 | **300** of 6000 |

`0.05` was never a performance choice. It was adopted so all 16 factorial cells
would share one static teacher and `E_final` would stay a common yardstick, then
rode into the LeWM test unexamined. Three explanations were possible: the flag,
the post-volume-loss rebuild of the cache and TD teacher, or 3-seed noise.

### What the flag actually controls

`freeze_at` gates three things at once
([`train_lip_ac.py:456-459`](../stable-worldmodel/scripts/plan/train_lip_ac.py#L456)),
not just the freeze point:

* the critic stops updating after `freeze_at`
* the expectile anneal 0.1 → 0.03 completes **over** `freeze_at` steps
* the critic LR cosine 1e-3 → 1e-4 completes **over** `freeze_at` steps

Since `--init-value` already supplies a fully trained 6000-step TD teacher,
frac 0.05 means *"use the pretrained teacher nearly as-is"* and frac 0.8 means
*"let it drift 4800 steps under LIP's own rollouts"*. Opposite ends of a real
design axis; only the two ends had ever been measured, on different assets.

**And critically:** `--expand-weight` acts only inside `critic_step()`
([`train_lip_ac.py:345-350`](../stable-worldmodel/scripts/plan/train_lip_ac.py#L345)),
which stops being called at `freeze_at`. So every prior measurement of the
factorial winner — including PLDM's +10.6 — was taken with the expand term
**active for 300 of 6000 steps and off for the other 95%.**

`--replay-prob` is *not* affected: it acts in `actor_step()`, which runs all
6000 steps regardless. So sweeping the fraction with the winner flags on is, to
first order, an **expand-weight dose-response curve with replay held fixed**.

---

## 2. Settings

Held identical across every arm unless stated:

```
--horizon 5 --iters 8 --steps 6000 --n-step 50 --arch v4
--actor-lr 3e-4 --actor-lr-final 3e-5
--expectile 0.1 --expectile-final 0.03
--critic-lr 1e-3 --critic-lr-final 1e-4
--mean-weight 0.1 (default)  --p-cross 0.3 (default)  --max-delta 10 (default)
```

Per-base, never shared:

| | LeWM | PLDM |
|---|---|---|
| WM | `v2WM` (`weights_epoch_22.pt`) | `PLDM_OgBench_lewm` (`weights.pt`) |
| `--amax` | **1.6** | **4.5** |
| fs1 / fs5 cache | `v2_tr8000_*` | `pldm_tr8000_*` |
| TD teacher | `v2_tr8000_TD.pt` | `pldm_TD.pt` |

Swept:

* `--freeze-critic-frac` ∈ {0.05, 0.3334, 0.5, 0.8} → freeze_at {300, 2000, 3000, 4800}
* winner bundle on/off: `--batch 256 --replay-prob 0.5 --expand-weight 1.0`

`p-cross` was retired at its default 0.3 — the TD sweep measured 0.6 = +0.2 and
0.0 = −3.3, so the default sits at the optimum.

Every arm gated on the trainer's own `"critic+teacher frozen"` line matching
`int(frac*6000)`, so a mis-typed fraction dies rather than silently producing a
rung that isn't what it claims.

Eval protocol, all cells:

```
eval_wm.py --config-name cube seed=<draw> ++bf16=true eval.img_size=224
  eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=8000:10000"
```

---

## 3. The isolation question is settled

`frz48` is the historical recipe re-run on today's rebuilt assets:

```
frz48 (historical recipe, rebuilt assets)   87.78 ± 0.89
historical dsp_pre                          87.80
                                           --------
                                             -0.02
```

**The rebuild is faithful; the 89.6-vs-87.8 gap was the freeze flag**, worth
**+1.78** on LeWM (sd 1.92, t=1.60, df=2 — not significant at 3 seeds, but the
isolation itself is unambiguous).

The earlier claim that the anchor "passed" was comparing two different
configurations. It has now actually been checked.

---

## 4. Freeze ladders (plain, no winner flags)

| freeze_at | frac | LeWM | spread | PLDM | spread |
|---|---|---|---|---|---|
| 300 | 0.05 | **89.56** | 0.7 | 73.56 | 4.00 |
| 2000 | 0.3334 | 85.78 | — | **75.11** | 2.00 |
| 3000 | 0.5 | 87.78 | — | 72.67 | 6.00 |
| 4800 | 0.8 *(default)* | 87.78 | 0.89 | **75.11** | 7.33 |

Paired 0.05 vs 0.8:

* **LeWM** +1.78, sd 1.92, t=1.60 — not significant
* **PLDM** −1.56, sd 5.35, t=−0.50 — not significant

PLDM's full-ladder span is 2.44 pts. This reproduces the banked contrast with
seeds: `fx_none` at frac 0.05 = 72.7 vs the 20-seed grid at frac 0.8 = 72.83,
a gap of −0.13. **On PLDM the freeze fraction is a null.**

The LeWM curve is non-monotone (300 best, 2000 worst), so "freeze early" is not
a clean law even there — but the 300 rung is the best cell on that base.

---

## 5. The cross: expand *was* underdosed — on PLDM

Winner bundle = `--batch 256 --replay-prob 0.5 --expand-weight 1.0`.

### LeWM (amax 1.6)

| freeze_at | expand dose | plain | winner | Δ | winner spread |
|---|---|---|---|---|---|
| 300 | 5% | 89.56 | 90.00 | +0.44 | 2.0 |
| 2000 | 33% | 85.78 | 88.44 | +2.67 | 3.3 |
| 3000 | 50% | 87.78 | 88.67 | +0.89 | 1.3 |
| 4800 | 80% | 87.78 | 88.44 | +0.67 | 2.7 |

Winner 4800 vs 300, paired: −1.56, sd 2.04, t=−1.32 — not significant.

### PLDM (amax 4.5)

| freeze_at | expand dose | plain | winner | Δ | winner spread |
|---|---|---|---|---|---|
| 300 | 5% | 73.56 | 79.56 | +6.00 | 6.0 |
| 2000 | 33% | 75.11 | 84.44 | +9.33 | 2.7 |
| 3000 | 50% | 72.67 | **85.33** | **+12.67** | 2.0 |
| 4800 | 80% | 75.11 | 84.22 | +9.11 | 3.3 |

Winner 4800 vs 300, paired: +4.67, sd 4.00, t=2.02 — not significant at 3 seeds,
but **three independent rungs all sit 3–6.7 pts above the 5% rung**, which is
worth more than that single test suggests.

**Reading.** On PLDM the winner benefit roughly doubles once the expand term is
allowed to run for a third of training or more. The factorial's 83.3 was
measured at 1/16 dose *and* on a single seed; `win@3000` is **85.33 at 3 seeds**.
So 83.3 was not the ceiling, and the term was being measured at a fraction of
its strength.

On LeWM the benefit stays small at every dose — consistent with the transfer
test's verdict that this is weak-base repair. But the *size* of that repair was
being understated by about a factor of two.

Because replay's dose does not scale with the fraction, the trend is
attributable to expand.

---

## 6. Selecting one shared config

Constraint: LIP-side knobs may differ per base, but TD/critic and data-side
knobs must be identical on both. A knob that only works on one base is a base
repair, not a method.

Chosen by a **pre-registered formula** written before the data existed — this
campaign has produced a phantom +2.0 from one lucky seed and twice had `E_final`
select a config that planned worse. Each base's 8 cell means are standardised
*within* that base (the bases sit ~89 vs ~73, so raw deltas would let one base
dictate), then ranked by `min(z_LeWM, z_PLDM)`. The `min` is what encodes "holds
for both". Ties: higher sum of z, then plain over winner, then smaller
`freeze_at`. Guard: must also beat the trainer default in raw points on **both**
bases, else fall back to the default. The selector was unit-tested on synthetic
sweeps for all three paths (winner cell / plain cell / incomplete-data fallback).

| cell | LeWM | z_L | PLDM | z_P | **min z** | ≥ default |
|---|---|---|---|---|---|---|
| plain@300 | 89.56 | +0.97 | 73.56 | −0.98 | −0.98 | no |
| plain@2000 | 85.78 | −1.97 | 75.11 | −0.69 | −1.97 | no |
| plain@3000 | 87.78 | −0.41 | 72.67 | −1.15 | −1.15 | no |
| plain@4800 | 87.78 | −0.41 | 75.11 | −0.69 | −0.69 | yes |
| win@300 | 90.00 | +1.32 | 79.56 | +0.15 | +0.15 | yes |
| win@2000 | 88.44 | +0.11 | 84.44 | +1.07 | +0.11 | yes |
| **win@3000** | 88.67 | +0.28 | **85.33** | **+1.24** | **+0.28** | yes |
| win@4800 | 88.44 | +0.11 | 84.22 | +1.03 | +0.11 | yes |

**Selected: `win @ freeze_at 3000`** →
`--freeze-critic-frac 0.5 --batch 256 --replay-prob 0.5 --expand-weight 1.0`,
amax per base.

Note the rule doing its job: **`plain@300` — LeWM's single best cell at 89.56 —
was excluded** because it falls below the default on PLDM. The selection cost
LeWM about 0.9 at PRE (88.67 vs its own best 89.56). That is the price of a
configuration that holds for both.

---

## 7. Dyna, per base

Two **independent** runs — separate collections, mixes, fine-tunes, caches, TD
teachers, actors, evals and cards. Nothing pooled across bases; only the knob
*values* were shared. A failure in one could not stop the other.

Protocol byte-identical to the banked PLDM round-1/round-2 loop so deltas stay
comparable: episode-disjoint collection on 0–7999 with `terminate_at_goal=False`,
50/50 expert:on-policy uniform-K mix, fine-tune lr 1e-5 for 2 epochs with epoch 1
pre-registered, then fresh caches → TD → 3 LIP actors → 9 held-out cells.
PRE is the selected cell's own already-measured 3-seed score, so the paired
t-test is properly paired on training seed.

| base | seed | PRE | POST | Δ |
|---|---|---|---|---|
| **LeWM** | 0 | 88.00 | 92.00 | +4.00 |
| | 1 | 88.67 | 92.00 | +3.33 |
| | 2 | 89.33 | 92.00 | +2.67 |
| | **mean** | **88.67** | **92.00** | **+3.33** — sd 0.67, **t=8.66, significant** |
| **PLDM** | 0 | 84.67 | 89.33 | +4.67 |
| | 1 | 86.67 | 93.33 | +6.67 |
| | 2 | 84.67 | 91.33 | +6.67 |
| | **mean** | **85.33** | **91.33** | **+6.00** — sd 1.15, **t=9.00, significant** |

All six seeds positive. Banked references: LeWM +4.7 (t=7.00, replicated 3×,
plateau ~92); PLDM round 1 +8.0 (t=2.11, n=3, not significant).

LeWM's POST is **92.00 on every seed** — the ~92 plateau reproduced a fourth
time. Dyna **stacked on top of** the winner config rather than overlapping with
it, answering an open question: the two are not redundant corrections.

### Loop mechanics (PLDM lane, representative)

```
collection    36/36 clean calls               21 min
mix           1,608,000 expert rows
              90,000 on-policy, K=18
              effective on-policy frac 0.502   1 min
fine-tune     lr-probe: epoch 0 lr=0 → 5.08e-06
              epoch 1 checkpoint taken        3h 31m
caches + TD                                   25 min
3 POST actors                                 48 min
9 eval cells                                   4 min
```

---

## 8. The headline

```
PLDM:  72.8  (LIP, 20 seeds)   →  85.33 (win@3000)  →  91.33  (+Dyna)
LeWM:  87.8  (LIP historical)  →  88.67 (shared cfg) →  92.00  (+Dyna)
```

**91.33 vs 92.00 is well inside noise.** PLDM has caught up.

---

## 9. Caveats

* **Everything is 3 seeds.** That resolves ~3 pts at best. The 20-seed PLDM grid
  spread was 68.0–79.3, so these means carry real uncertainty. The convergence
  claim in §8 rests on 3 seeds per base and wants a 6-seed confirm.
* **`win@3000` carries winner's curse** — it was the best of 8 cells on PLDM.
  Its 85.33 is the selection-time value; the Dyna PRE re-uses those same actors,
  so any upward bias in PRE makes the +6.00 Dyna delta *conservative*, not
  inflated.
* **The winner is a bundle.** `batch 256` + `replay-prob 0.5` +
  `expand-weight 1.0` were never separated at the chosen freeze point. The
  banked factorial says they don't function apart (`expand10` alone = 70.7, the
  worst of 16 cells; `replay50` alone = 74.0; together with batch256 = 83.3,
  interaction +5.50), which is why they were bundled — but the decomposition at
  freeze_at 3000 is untested.
* **`amax` was originally selected on the eval draws** (~3 pts of optimism),
  a pre-existing campaign-wide caveat inherited here.
* **v2WM was pre-trained on all 10k episodes.** The train/eval split is enforced
  everywhere downstream, but the base WM itself still leaks. Fixing that needs a
  base-WM retrain on a subset.

---

## 10. Files

| | |
|---|---|
| `dyna_harness/lewm_freeze_ladder.sh` | LeWM plain ladder + isolation verdict |
| `dyna_harness/pldm_freeze_ladder.sh` | PLDM plain ladder |
| `dyna_harness/win_freeze_cross.sh` | winner × freeze, both bases, 2×4 card |
| `dyna_harness/dyna_perbase.sh` | pre-registered selector + two independent Dyna runs |

Pod logs: `driver_lewmfrz.log`, `driver_pldmfrz.log`, `driver_wincross.log`,
`driver_dynaperbase.log`, `driver_dyna_jl.log`, `driver_dyna_jp.log`.
Results CSV: `/workspace/results/summary_pldmgrid_egl.csv`.
Fine-tuned WMs: `/workspace/models/dyna_pb_jl`, `/workspace/models/dyna_pb_jp`.
