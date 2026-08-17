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

`scripts/sky/launch_tworoom_20260817.sh`, one job per train seed (so every job
carries its grid's control and trains its own TD teacher; the replicated
control also measures job-to-job drift). Train seeds 0-3, reporting draws
42/43/44. **K is fixed at 8 throughout — refinement depth is not a
variable in this campaign.**

### 3a. Horizon-matched training

**h25 and h100 actors are trained separately.** Every arm carries its own goal
band (`max_delta`) and is scored at its own single horizon, so an h100 number
is never read off an h25-tuned refiner. `max_delta=12` covers the h25 goal
(5 blocks); `max_delta=20` reaches the 100 primitive steps an h100 goal sits
at. This turns the goal band from a confound into a factor.

### 3b. LeJEPA — `GRID=escale` (10 arms x 4 seeds)

h25-trained (`max_delta` 12, scored at 25):

| arm | amax | vnorm |
|---|---|---|
| `es25_ctrl_a1.8` | 1.8 | none |
| `es25_vlog_a1.8` | 1.8 | log |
| `es25_vinv_a1.8` | 1.8 | loggn |
| `es25_vlog_a1.4` | 1.4 | log |

h100-trained (scored at 100) — a 2x2 on goal band x conditioning, plus two
extras. `es100_ctrl_md12_a1.8` reproduces exactly the condition behind the
banked 94.2, so the factorial is anchored:

| arm | max_delta | amax | vnorm |
|---|---|---|---|
| `es100_ctrl_md12_a1.8` | 12 | 1.8 | none |
| `es100_ctrl_md20_a1.8` | 20 | 1.8 | none |
| `es100_vlog_md12_a1.8` | 12 | 1.8 | log |
| `es100_vlog_md20_a1.8` | 20 | 1.8 | log |
| `es100_vinv_md20_a1.8` | 20 | 1.8 | loggn |
| `es100_vlog_md20_a2.6` | 20 | 2.6 | log |

### 3c. PLDM — `GRID=pldm_h25_probe` (12 arms x 4 seeds)

h25-trained (`max_delta` 12, scored at 25) — the target cell. A 7-point amax
dose-response (**1.0 / 1.2 / 1.4 / 1.6 / 1.8 = ctrl / 2.0 / 2.2**) plus the
conditioning lever at the control clip (`pl25_vlog_a1.8`, `pl25_vinv_a1.8`).
Everything below 1.8 is unexplored: the live `terminal_search` grid sweeps
2.4-3.2, entirely on the far side of the deployed winner, which itself entered
via `INCLUDE_WINNERS` rather than from a sweep.

h100-trained (scored at 100) — not the target (RLP already leads value+CEM
there, 96.0 vs 88.7), but with horizon-matched training the h100 number needs
its own arms: `pl100_ctrl_md12_a1.8` (anchors the banked 96.0),
`pl100_ctrl_md20_a1.8`, `pl100_vlog_md20_a1.8`.

### 3d. Scale

8 jobs x H200:4 = **32 GPUs**, one job per train seed per grid. Train seeds
0-3 x reporting draws 42/43/44 x 50 episodes = **600 reported episodes per
arm** (SE ~0.6 pt at p=0.98).

There are **no separate selection draws**, so an argmax over arms would be
selection on the reported draws. The analysis therefore does not pick a
winner by argmax: each arm is reported against its own in-job control (the
comparison is pre-registered and paired at the episode level), and the PLDM
amax ladder is read as a **dose-response shape** — a monotone curve across
seven clip values is evidence in a way a single best point is not. Any arm
promoted on this basis needs a confirmation run on fresh draws before it is
quoted as a headline number.

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

## 4b. Provenance

| stage | jobs | tag |
|---|---|---|
| value+CEM remeasure | 7699–7704 | `vcem2-tw-{lejepa,pldm}-e{42,43,44}-20260817` |
| smokes (control path) | 7726 / 7727 | `rlp-tw-{escale,pldmamax}-s0-20260817-smoke` |
| smoke (`vnorm=log` path) | 7736 | `rlp-tw-vlog-smoke-20260817` |
| escale, seeds 0–3 | 7752–7755 | `rlp-tw-escale-s{0..3}-20260817` |
| pldm, seeds 0–3 | 7756–7759 | `rlp-tw-pldmamax-s{0..3}-20260817` |

Jobs 7740–7751 were an earlier 6-seed / 5-draw launch of the same grids,
cancelled ~20 min in (during cache build, before any actor trained) when the
protocol was narrowed to seeds 0–3 and draws 42/43/44. No results came from
them.

Smoke gate cleared on all three: `[grid] -> N configs` (never 0),
`[args-ok] ... iters=8 ... md=... vnorm=...` matching the row on every cell,
`[summary]` non-null, no `[args-mismatch]` and no `REUSED persistent actor`.
Job 7736 specifically confirms `--vnorm log` threads trainer → checkpoint →
guard, and that a horizon-separated row evaluates at exactly one offset
(`{"rh5_h25_s42": …}` with no h100 key).

## 5. Open

- Results for §3 (jobs 7752–7759 in flight at time of writing, ~4.5 h).
- Both banked numbers this campaign is measured against (LeJEPA 94.2, PLDM
  98.2) came from a *single* actor scored at both horizons. The h100 arms here
  are h100-trained, so the fair anchor for them is the in-job `*_ctrl_md12`
  arm rather than the banked value.
- If `vnorm=log` works, Cube h100 is the immediate next test — the same raw-`E`
  conditioning is in every RLP actor.
