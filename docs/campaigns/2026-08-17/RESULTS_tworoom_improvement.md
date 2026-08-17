# TwoRoom RLP improvement campaign — 2026-08-17

Target: raise RLP on TwoRoom above its shipped held-out numbers, on the two
cells where it does not clearly lead — **LeWM/LeJEPA h100** (94.2) and
**PLDM h25** (98.2). Held-out protocol throughout (value + planner train on
episodes 0–7999, evaluation draws tasks from 8000–9999), 50 episodes per cell.

## 1. value+CEM remeasured (complete)

Six shards, `scripts/sky/tworoom_value_cem.yaml`, `SPLIT=1`, jobs 7699–7704,
tags `vcem2-tw-{lejepa,pldm}-e{42,43,44}-20260817`. One shard per
(base × eval draw); each builds its own caches and its own `value_td` critic,
then plans with CEM (9,000 rollouts/decision) against it.

| cell | draws 42 / 43 / 44 | mean | sd | prior record | Δ |
|---|---|---|---|---|---|
| LeJEPA h25 | 100 / 100 / 100 | **100.0** | 0.0 | 100.0 | +0.0 |
| LeJEPA h100 | 98 / 94 / 92 | **94.7** | 3.1 | 96.0 | −1.3 |
| PLDM h25 | 100 / 100 / 100 | **100.0** | 0.0 | 100.0 | +0.0 |
| PLDM h100 | 94 / 88 / 84 | **88.7** | 5.0 | 89.3 | −0.6 |

Two things follow, and the second corrects the framing this campaign started
from.

1. **PLDM h25 value+CEM is 100.0, reproduced exactly** (3/3 draws at ceiling).
   The paper's 98.7 for this cell is genuinely lower than what the held-out
   pipeline measures. RLP's 98.2 therefore has ~1.8 points of real headroom
   against a sampling planner using the *same* critic and world model — an
   optimizer gap, not a representation gap.
2. **LeJEPA h100 value+CEM is 94.7, not 96.0.** Against RLP's 94.2 that is a
   tie, not a loss (sd across draws is 3.1, and h100 spreads 92–98 across
   draws on the same base). The claim that value+CEM beats RLP on that cell
   rests on the banked 96.0 and does not survive remeasurement. What *does*
   survive is **DMPO's 99.1** on the identical held-out protocol — that is the
   number establishing h100 headroom, and it is ~4.9 above RLP.

The h25 cells are at ceiling on both bases and reproduce exactly; the h100
cells are the noisy ones, which is itself consistent with long-horizon
TwoRoom being where planner quality actually separates.

## 2. Why RLP's h100 is weak — mechanism (verified in code, not yet in an experiment)

The LIPv4 actor's first layer is `(512, 101)` = `[A(50), grad_A V(50), E(1)]`
([overlays/lip.py:91](../../../scripts/sky/overlays/lip.py)), confirmed against
the shipped checkpoint's `net.0.weight`. LIPv4 sets `use_z0=False` and
`use_zg=False`, so **the raw critic value `E` is the only channel carrying goal
distance**, and it is consumed unnormalised
([overlays/lip.py:113](../../../scripts/sky/overlays/lip.py)).

With `gamma=1.0` the quasimetric is steps-to-go, so h100 presents `E` roughly
4× larger than h25 and well outside the band the actor trains in
(`max_delta=12` blocks). The measured actor response `‖dA‖` is linear in `E`,
and the clamp is applied to the *cumulative plan*
(`(A + dA).clamp(-amax, amax)`), so a larger `E` drives a growing fraction of
plan entries onto the `±amax` boundary — a non-uniform projection that rotates
the plan and kills its training gradient.

DMPO and CEM consume `V` only after z-scoring or ranking, so both are
structurally invariant to this. That is consistent with DMPO holding 99.1 at
h100 while training its own actor at the *same* `max_delta` — which is why the
original "train on farther goals" hypothesis is not the primary explanation,
though it is still tested below as a rival.

Unit-tested locally: with `vnorm=log`, the actor's sensitivity to an `E` shift
of 25→100 drops **58×**, and `in_dim` stays 101 so state_dicts remain
interchangeable.

## 3. Arms in flight

Six jobs, `scripts/sky/launch_tworoom_20260817.sh`, one job per train seed
(so every job carries the control and trains its own TD teacher; the
replicated control also measures job-to-job drift). Selection draws 50/51,
reporting draws 42/43/44, both horizons for every arm. **K is fixed at 8
throughout — refinement depth is not a variable in this campaign.**

### 3a. LeJEPA — `GRID=escale` (7 arms × 3 seeds)

| arm | amax | max_delta | vnorm | tests |
|---|---|---|---|---|
| `escale_ctrl` | 1.8 | 12 | none | shipped recipe, retrained in-job |
| `escale_vlog` | 1.8 | 12 | log | **pre-registered primary**: `log1p(E)` |
| `escale_vinv` | 1.8 | 12 | loggn | `log1p(E)` + RMS-normalised `grad_A V` |
| `escale_vlog_a2.6` | 2.6 | 12 | log | clamp headroom after compression |
| `escale_vlog_a1.4` | 1.4 | 12 | log | the other side of the clamp |
| `escale_md20` | 1.8 | 20 | none | rival fix: widen the trained goal band |
| `escale_vlogmd20` | 1.8 | 20 | log | both mechanisms at once |

`escale_vlog` is pre-registered as primary because the LeJEPA arms are
reported on 42/43/44; the rest are secondary.

### 3b. PLDM — `GRID=pldm_h25_probe` (7 arms × 3 seeds)

A single 7-point dose-response on `amax` at the shipped operating point
(mean-weight 0.0, actor-lr 1e-3, max_delta 12, K=8): **1.0 / 1.2 / 1.4 / 1.6 /
1.8 (= ctrl) / 2.0 / 2.2**. Everything below 1.8 is unexplored — the live
`terminal_search` grid sweeps 2.4–3.2, entirely on the far side of the
deployed winner, which itself entered via `INCLUDE_WINNERS` rather than from a
sweep.

## 4. Guards

- Every cell asserts the trained checkpoint's own `train_args` (`iters`,
  `amax`, `mean_weight`, `actor_lr`, `max_delta`, `vnorm`) against its grid row
  before an eval is spent on it, so a silent default-fallback cannot be
  reported as the arm it was named after.
- The solver prints the conditioning it loaded and raises if it cannot honour
  it; the in-repo port raises on a `vnorm` checkpoint rather than feeding raw
  `E` (`in_dim` matches, so it would otherwise fail as a mystery-weak cell).
- Fresh `CACHE_VERSION` and `EXPERIMENT_TAG` per job; `grep -c 'REUSED
  persistent actor'` must be 0 on a first launch.
- Any arm that wins h100 while dropping h25 below 99.3 is rejected on that
  basis alone.

## 5. Open

- Results for §3 (jobs not yet launched at time of writing — the harness
  clones a second repo with a PAT that must be supplied at launch).
- A promoted PLDM winner needs a 3-seed confirmation before it is quoted
  against the banked 98.2.
- If `vnorm=log` works, Cube h100 is the immediate next test — the same raw-`E`
  conditioning is in every RLP actor.
