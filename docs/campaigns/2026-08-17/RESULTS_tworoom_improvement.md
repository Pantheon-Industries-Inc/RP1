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

---

## 6. PLDM h25 — results (complete, 2026-08-18)

Jobs 7756–7759, `GRID=pldm_h25_probe`, train seeds 0–3, reporting draws
42/43/44, 50 episodes, held-out, RH=5. K=8 throughout.

### 6a. The amax ladder is flat — amax is not the lever

| amax | 1.0 | 1.2 | 1.4 | 1.6 | **1.8 (ctrl)** | 2.0 | 2.2 |
|---|---|---|---|---|---|---|---|
| mean | 98.50 | 97.17 | 98.67 | 98.83 | **98.17** | 98.33 | 98.33 |
| sd | 0.84 | 1.00 | 0.77 | 0.33 | 0.33 | 0.67 | 0.86 |

The in-job control reproduces the banked 98.2 almost exactly (98.17), which
validates the pipeline. But the ladder is **non-monotone** — 1.2 dips to 97.17,
below both neighbours — so there is no dose-response shape. A 1.66-point spread
across seven values with per-arm sd 0.3–1.0, read as the max of eight
comparisons, is noise. amax 1.6's +0.66 over control is **not** claimed.

Conditioning is also null at h25 (`vlog` 98.33, `vinv` 98.67 vs ctrl 98.17) —
expected, since at a 5-block goal `E` is already inside the training band.

### 6b. Cost vs. search coverage (job 7878, 3 actors × 3 draws, n=9)

Crosses planner × critic on the vendored actors, no retraining.

| arm | n | mean | sd | per-actor |
|---|---|---|---|---|
| **A** CEM + TD teacher | 3 | **100.00** | 0.00 | — |
| **B** CEM + co-trained critic | 9 | **99.11** | 1.45 | 98.0/99.3/100.0 |
| **C** RLP + co-trained (shipped) | 9 | 97.11 | 3.62 | 98.0/97.3/96.0 |
| **D** RLP + TD teacher | 9 | **98.00** | 2.24 | 99.3/97.3/97.3 |
| **E** RLP + value-init 16 | 9 | 92.22 | 2.91 | 94.7/91.3/90.7 |
| **E** RLP + value-init 64 | 9 | 95.33 | 4.12 | 94.7/94.7/96.7 |
| **E** RLP + value-init 256 | 9 | 94.44 | 2.79 | 94.0/93.3/96.0 |
| **I** RLP + value-init 64 + TD | 9 | 94.89 | 2.26 | 95.3/93.3/96.0 |

1. **Search coverage is refuted.** Every value-init arm is BELOW the zero-init
   control, at all three candidate counts and on both critics, consistently
   across all three actor seeds. The refiner is trained to refine from A=0, so
   a value-selected start is off-distribution — the same train/deploy mismatch
   family as the raw-`E` problem, not a fix for it. `_value_init` should stay
   off unless an actor is *trained* with it.
2. **The cost is real, worth ~1 point.** CEM loses 0.89 on the co-trained
   critic (100.00 → 99.11) and RLP gains 0.89 on the teacher (97.11 → 98.00) —
   same magnitude, opposite directions. `freeze_critic_frac=0.8` co-training
   mildly degrades the cost. **`D` is a free +0.9 at zero deploy cost: point
   `core.solver.value_path` at the offline teacher.**
3. **~2 points are genuinely the optimizer.** On the identical good critic RLP
   is 98.00 vs CEM's 100.00, and none of amax, `vnorm` or value-init touched
   that residual.

**Correction to §1.** This campaign opened by asserting RLP and value+CEM
shared a critic, so the gap "had to be" the optimizer. That was wrong:
`LIPSolver` deploys `load_metric(ck["value"])` — the co-trained critic — while
value+CEM used the untouched TD teacher. Arms A–D are the corrected comparison.

### 6c. Where the remaining ~2 points are not

Four levers tested on this cell, three null and one worth ~1 point:

| lever | result |
|---|---|
| action clip `amax` (7 values) | flat, non-monotone — null |
| value conditioning `vnorm` | null at h25 (in-band by construction) |
| deploy-time search coverage (value-init ×3) | **harmful**, −2 to −5 |
| critic: co-trained → offline teacher | **+0.9**, free |

Reaching CEM's 100.0 therefore needs a change to how the refiner is *trained*
(e.g. freezing the critic outright, or training with the deployment init), not
to how it is clipped, conditioned or initialised at deploy.

## 7. Ops — the seed-linked training hang

`escale` seeds 0/1/2 hung **6 times out of 6** across six distinct nodes
(GPU UUIDs verified different each time), while seed 3 completed 10/10 cells on
its first attempt. The hang signature: training reaches `step 500` of 8000,
writes ~2.2 KB of log, then nothing — no error, no non-zero exit, so no
`FAILDIR` entry. The stalled processes hold all 4 slots, the mkdir allocator
blocks forever, and the remaining 6 cells never launch. Two jobs sat like this
for 7 h and 1.5 h respectively before being killed.

Not a bad node (six different ones) and not the vnorm change (the arm that
hangs is `es25_ctrl_a1.8`, `vnorm=none`, the unmodified path, which seed 3 ran
fine). Root cause unresolved; it tracks the train seed. Worked around by
running seeds 4–7 instead — actor seeds are exchangeable, and the final LeJEPA
table will state which seeds it used rather than implying 0–3.

**Detection lesson:** a hang is invisible to the harness's failure accounting,
which only catches non-zero exits. The cheap signal is per-job
(cells dispatched, cells completed): a job sitting at 4 dispatched / 0
completed for >1 h is stalled. That check would have caught this at ~40 min
instead of 7 h.

## 8. LeJEPA h100 — the 2x2, seed 3 (PROVISIONAL, 1 of 4 seeds)

Job 7755, `GRID=escale`, train seed 3, draws 42/43/44, 50 episodes, held-out,
RH=5, K=8. Horizon-matched: every arm trained for the horizon it is scored at.

| arm | max_delta | vnorm | h100 | per-draw |
|---|---|---|---|---|
| `es100_ctrl_md12` | 12 | none | 88.67 | 86/84/96 |
| `es100_ctrl_md20` | 20 | none | 88.67 | 86/86/94 |
| `es100_vlog_md12` | 12 | log | 94.67 | 96/90/98 |
| **`es100_vlog_md20`** | 20 | log | **97.33** | 100/96/96 |
| `es100_vinv_md20` | 20 | loggn | 92.67 | 94/88/96 |
| `es100_vlog_md20_a2.6` | 20 | log | 91.33 | 88/88/98 |

Main effects, from the 2x2 on (goal band) x (conditioning):

| effect | value |
|---|---|
| `max_delta` 12 → 20, alone | **+0.00** |
| `vnorm` none → log, alone | **+6.00** |
| both | **+8.67** (⇒ interaction **+2.67**) |

**The goal-band hypothesis is null on its own and only pays off through the
interaction.** That is precisely what the mechanism predicts: widening the
trained goal band cannot help while the actor's sole distance channel (raw `E`)
is saturated out of band at h100; compress `E` and the wider band becomes
usable. The two are sequential, not rival.

`loggn` (92.67) is worse than plain `log` (94.67) — normalising the gradient as
well overshoots. A looser clip under `log` (a2.6, 91.33) is worse than a1.8, so
the clip is not a lever here either.

**h25 does not regress**: all four h25 arms score 100.0/100.0/100.0.

### Caveats

- **One seed.** n=150 episodes per arm. Seeds 4–7 in flight (7904–7907).
  Do not quote +8.67 until at least three seeds agree.
- **Anchor on the in-job control, not the banked 94.2.** `es100_ctrl_md12`
  reads 88.67 because these are separately-trained horizon-matched actors (the
  banked number came from ONE actor scored at both horizons), plus the
  unresolved cross-pipeline level discrepancy in §5. The defensible claim is
  the within-job **+8.67**; against the banked 94.2 it is +3.1. Either way
  97.33 clears remeasured value+CEM (94.7) and closes most of the gap to
  DMPO (99.1).

## 9. Protocol note (2026-08-18)

**All reported cells must use train seeds 0/1/2** (user decision). Consequences:
- The §8 seed-3 2x2 is **diagnostic evidence only**, never a reportable number.
- The seeds-4–7 fleet (7904–7907) and g98 seeds 8/9 (7911/7912) were cancelled.
- `rlp-tw-g98-s3` (7913) is kept as a **diagnostic pair** for §8's γ=1.0 seed-3
  data — same seed, same grid, γ the only variable — and is likewise not
  reportable.
- Seeds 0/1/2 are exactly the seeds that hang (§7), so the harness now carries
  a silent-hang watchdog: a train log untouched >25 min while its trainer
  lives gets SIGUSR1 (faulthandler dumps all Python stacks into that log),
  60 s grace, SIGKILL — the cell fails loudly, the slot frees, the grid
  continues, and the next stall leaves an autopsy. Trainer calls also carry a
  4 h hard timeout.
