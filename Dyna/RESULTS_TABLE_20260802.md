# Cube results — one shared configuration, both bases

>**SUPERSEDED as the reference config (2026-08-03).** The numbers here are at
>`expand-weight 1.0`. The current SOTA is **`expand-weight 3.0`** — LeWM 90.00,
>PLDM 84.00 — recorded in [`HANDOFF_20260728.md`](HANDOFF_20260728.md#sota--expand-weight-30-current-best-config-2026-08-03).
>This document remains valid for the Dyna deltas and the sweep history, which
>were measured at `expand 1.0` and have not been repeated at 3.0.


**2026-08-02, 8×H200, EGL.** Held-out evaluation on episodes `8000:10000` for
every number; all training and collection on `0:7999`. 50 tasks per draw,
draws 42/43/44.

Everything below is measured at **one shared configuration**, so the tables are
directly comparable across bases, planners and horizons.

```
shared (identical on both bases)
  --arch v4 --horizon 5 --iters 8 --steps 6000
  --freeze-critic-frac 0.5     (critic live for 3000 of 6000 actor steps)
  --batch 256 --replay-prob 0.5 --expand-weight 1.0
  --gamma 0.98 --n-step 50 --p-cross 0.3 --mean-weight 0.1
  --expectile 0.1 -> 0.03      --critic-lr 1e-3 -> 1e-4
  --actor-lr 3e-4 -> 3e-5
  TD teacher: quasimetric (MRN), expectile 0.03, gamma 0.98, n-step 50, 12k steps

per base (the only settings that differ)
  amax   LeWM 1.6   |   PLDM 4.5
```

---

## 1. Headline

```
LeWM   87.80  historical LIP  ->  89.11  base  ->  92.00  + Dyna
PLDM   72.83  20-seed LIP     ->  85.11  base  ->  91.33  + Dyna
```

PLDM closed a 15-point gap to LeWM. **At the tuned horizon (h25) the two are
statistically indistinguishable** (91.33 vs 92.00). This does **not** survive the
horizon shift — see §3.

Base figures are 6 seeds (LeWM) and 3 seeds (PLDM); Dyna figures are 3 seeds.

---

## 2. Planner ladder, in-distribution (h25)

Goal = frame t+25, 50-step budget — the protocol everything was tuned at.

| planner | LeWM PRE | LeWM POST | PLDM PRE | PLDM POST |
|---|---|---|---|---|
| latent + CEM | 76.00 | 78.00 | 66.67 | 72.00 |
| TD + CEM | 79.33 | 85.33 | 73.33 | 84.00 |
| PWM (reactive, 0 WM rollouts) | 84.00 | 81.33 | 62.22 | 61.11 |
| PWM (open-loop, 5 WM rollouts) | 83.11 | 81.56 | 62.00 | 59.56 |
| **LIPv4** | **88.67** | **92.00** | **85.33** | **91.33** |

**Dyna delta by planner** — PRE → POST on the same world model:

| planner | LeWM | PLDM |
|---|---|---|
| latent + CEM | +2.00 | +5.33 |
| TD + CEM | +6.00 | **+10.67** |
| PWM | −1.56 | −2.44 |
| LIPv4 | +3.33 | +6.00 |

`latent+CEM` uses no learned value and no learned actor. It rises on both bases,
which is the cleanest available evidence that the Dyna fine-tune improved the
**world model itself** rather than merely making it easier for a learned actor
to exploit. PWM is the only planner Dyna makes worse — and the only one that
uses no world model at deploy time, so it cannot consume the improvement.

CEM rows: 3 draws, no training seed. PWM / LIP rows: 3 seeds × 3 draws.

---

## 3. Planner ladder, out of distribution (h100)

Goal = frame t+100, 200-step budget. **Nothing was ever tuned here.**

| planner | LeWM PRE | LeWM POST | PLDM PRE | PLDM POST |
|---|---|---|---|---|
| latent + CEM | 60.00 | 70.67 | 55.33 | 60.00 |
| TD + CEM | 74.00 | 82.00 | 63.33 | 79.33 |
| PWM (reactive) | 81.56 | **81.56** | 53.78 | 58.00 |
| **LIPv4** | 81.78 | **85.33** | 83.11 | **80.22** |

Three things change out of distribution:

1. **The convergence does not hold.** LeWM POST 85.33 vs PLDM POST 80.22 — 5.1
   points apart, where at h25 they were 0.67 apart.
2. **PLDM POST LIP is the only negative Dyna cell in the campaign** (83.11 →
   80.22). Two candidate causes, not yet separated: Dyna collected its
   on-policy data at h25, *and* the value function's discount saturates below
   the task's scale (§6).
3. **PWM is horizon-invariant where LIP is not.** LeWM POST PWM is 81.56 at both
   horizons — identical. LIP loses ~6.7. At h100 on LeWM PRE, a single MLP
   forward pass (81.56) matches LIPv4's 8 refinement iterations (81.78).

---

## 4. Dyna, per base

Two independent runs. Episode-disjoint collection on 0–7999 with
`terminate_at_goal=False`, 50/50 expert:on-policy uniform-K mix, fine-tune
lr 1e-5 epoch 1 (pre-registered), fresh caches → TD → 3 LIP actors.

| base | seed 0 | seed 1 | seed 2 | mean | Δ | t (df=2) | |
|---|---|---|---|---|---|---|---|
| **LeWM** PRE | 88.00 | 88.67 | 89.33 | 88.67 | | | |
| **LeWM** POST | 92.00 | 92.00 | 92.00 | **92.00** | **+3.33** | 8.66 | **significant** |
| **PLDM** PRE | 84.67 | 86.67 | 84.67 | 85.33 | | | |
| **PLDM** POST | 89.33 | 93.33 | 91.33 | **91.33** | **+6.00** | 9.00 | **significant** |

All six seeds positive. LeWM POST is 92.00 on every seed — the ~92 plateau
reproduced for a fourth time across independent world models.

---

## 5. Hyperparameter search: what moved and what did not

~180 LIP actor trains, 96 TD trains, 12 PWM trains, ~1,400 eval cells.

### Effects that survived re-measurement

| finding | size | evidence |
|---|---|---|
| **Dyna** | +3.33 / +6.00 | t=8.66 / 9.00, all 6 seeds positive |
| **winner bundle** (`batch 256 + replay 0.5 + expand 1.0`) | +12.3 on PLDM | replicated across 4 freeze levels |
| **expand-weight dose** | +6.00 → +12.67 on PLDM | monotone across 3 independent freeze levels |
| **critic-freeze isolation** | +1.78 on LeWM | `frz4800` = 87.78 landed on the 87.80 historical anchor |

### Axes closed — the inherited value is correct

| knob | levels tested | outcome |
|---|---|---|
| `amax` | 1.2 / 1.6 / 2.0 / 2.4 (LeWM), 3.5 / 4.5 / 5.5 / 6.5 (PLDM) | every alternative worse on both bases |
| `critic-lr` | 3e-3, 1e-3 (default), 3e-4, flat | negative on both bases |
| `p-cross` | 0.0 / 0.3 / 0.6 | 0.0 = −3.3; default already optimal |
| `gamma` (at h25) | 1.0 / 0.99 / 0.98 / 0.96 / 0.95 / 0.90 | flat; optima scattered, no structure |

### Findings that did **not** replicate

| claim | at 3 seeds | on re-measurement |
|---|---|---|
| `gamma 0.98` teacher | +3.3 | +0.33 |
| `steps 12000` teacher | +2.7 | folded into the above null |
| gamma PRE/POST split | 4/4 consistent signs | scattered under a full 6-point curve |
| `expand 3.0` on LeWM | 90.22, t=3.50 | **89.89 at 6 seeds, t=1.28** |
| `replay 0.25` on LeWM | 90.00, t=3.46 | **89.44 at 6 seeds, t=0.65** |
| `mean-weight 0.03` on PLDM | best in the old factorial | now PLDM's **worst** arm, t=−3.59 |

**Every finding selected as the maximum of a multi-arm sweep at 3 seeds has
failed on re-measurement. Every finding with a mechanism stated in advance has
held.** That is the most transferable result here.

### The negative result worth stating plainly

**Twenty-one consecutive arms failed to beat the deployed config on PLDM** — 10
from the LR/freeze sweep, 5 from replay/expand levels, 6 from the per-base
sweep, across three independent sweeps. PLDM's configuration is converged.
On LeWM everything lives in an **89–90 band** with nothing statistically
separated from the 89.11 baseline.

---

## 6. Caveats

* **3 seeds unless stated.** That resolves ~3 points. The 20-seed PLDM LIP
  spread was 68.0–79.3, so single-arm means carry real uncertainty.
* **No configuration has ever exceeded 90 at 6 seeds.** Five have at 3 seeds;
  the two that were re-measured both fell back.
* **The h100 value function is out of range.** `gamma 0.98` gives an effective
  horizon of `1/(1−γ) = 50` primitive steps, but h100 has a 200-step budget, so
  the critic cannot distinguish 80 steps from 150. Every h100 number in §3 was
  measured through that ceiling. A gamma × n-step sweep scored *at* h100 is
  running.
* **`amax` was originally selected on these same eval draws** (~3 points of
  optimism) — a pre-existing campaign-wide caveat.
* **v2WM was pre-trained on all 10k episodes.** The train/eval split is enforced
  everywhere downstream, but the base representation has seen these frames, so
  absolute rates are upper bounds. Dyna deltas are unaffected — both arms share
  the base.
* **Our PWM is not a faithful PWM baseline.** Paper PWM trains online; ours is
  fully offline and was given neither online data nor the offline substitutes
  (`replay`, `expand`) that LIP has. It also ran at each base's LIP-tuned `amax`
  rather than its own default of 2.2. Every PWM gap here is therefore an
  **upper bound** on LIPv4's real advantage.
* **All learning is offline.** No trainer touches an environment; every rollout
  is imagined through a frozen world model. Dyna's collection phase is the only
  real environment data, and it enters as an outer loop, never a gradient.
