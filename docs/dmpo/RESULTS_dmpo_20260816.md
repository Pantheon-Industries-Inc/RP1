# DMPO baseline campaign — results record (2026-08-14 … 2026-08-17)

Offline DMPO (Sacks et al., ICRA 2024; `model=dmpo`, `core/solver=dmpo`; method
mapping and deviations in [README_dmpo.md](README_dmpo.md)) evaluated against
the same-critic MPPI baseline on all three environments and both bases.

**Campaign headline.** DMPO beats the hand-written MPPI update it learns a
residual on in **13 of 14 comparable cells**, at 1/35th the rollouts, and loses
to RLP in **13 of 14**. The two exceptions are informative rather than
symmetric: the MPPI loss is Cube PLDM, itself confounded by an inherited action
clip; the RLP loss is TwoRoom LeJEPA h100, a genuinely weak RLP cell that
value+CEM also fails to beat.

## Shared protocol

| | |
|---|---|
| training | `model=rlp skip=[planner]` (latent caches + offline quasimetric critic) → `model=dmpo` per seed → `rlp.eval.world_model` |
| seeds | 3 optimizer seeds (0/1/2) × eval draws 42/43/44 × 50 episodes = **n=9 per DMPO cell**; MPPI rows are 3 eval draws against the identical `value_td` critic and world model |
| split | held-out everywhere — value/optimizer train on episodes 0–7999, eval tasks draw 8000–9999 |
| harness | `scripts/sky/dmpo_campaign.yaml` |
| rollouts/decision | **DMPO 256 forward** (256 samples × 1 learned iteration, no backward); **MPPI/CEM 9,000 forward** (300 × 30); **RLP 9 forward** — 8 refinement iterations, each reusing its single unroll for both the value gradient and the actor's features, plus one selection unroll — with 8 backward passes over those same graphs |

**Action bounds.** DMPO clips plans to the environment's own per-dimension
action limits, derived from the dataset's action statistics
(`core.planner.action_range=1.0`). Earlier cells inherited RLP's symmetric
`amax`, which is a *tuned residual trust region* and not a bound; reusing it
produced one 38-point artifact (see the TwoRoom retraction). Cells still on the
old clip are marked ◇.

---

# 1. TwoRoom

**Status: current** — held-out, environment action bounds (±1.4 in z-scored
units), full 3×3 grid. Jobs 7730/7731 (bounds), 7689/7690 (split), 21 result
cells each. Data is the authors' fetched pool
(`tool=fetch_dataset dataset=tworoom`, 10,000 episodes, h5), evaluated through
the `tworoom_{lewm,pldm}_h5` roots — that pool stores the agent position as
`pos_agent` and carries no `state` column. Horizons h25
(`goal_offset_steps=25 budget=50`) and h100 (`=100`/`=500`).

| planner | roll./decision | LeJEPA h25 | LeJEPA h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** | 256 | 96.9 | **100.0** | 97.8 | 92.7 |
| value MPPI (same critic) | 9,000 | 83.3 | — | 76.0 | — |
| RLP (held-out record) | 9 | **100.0** | 94.2 | **98.2** | **96.0** |
| value CEM (remeasured 2026-08-17) | 9,000 | 100.0 | 94.7 | 100.0 | 88.7 |

1. **DMPO beats same-critic MPPI by +13.6 / +21.8** at h25 using 1/35th the
   rollouts.
2. **RLP leads three of four cells** — h25 on both bases, PLDM h100 by 3.3.
3. **TwoRoom LeJEPA h100 is a weak RLP cell**: RLP 94.2 against DMPO's 100.0.
   value+CEM *ties* RLP there (94.7 remeasured over six shards; the banked 96.0
   did not reproduce), so DMPO's +5.8 is the only clear evidence the cell has
   headroom. Mechanism, established in
   [../campaigns/2026-08-17/RESULTS_tworoom_improvement.md](../campaigns/2026-08-17/RESULTS_tworoom_improvement.md):
   the LIPv4 actor consumes the raw critic value `E` as its only goal-distance
   channel, unnormalised, and with `gamma=1.0` h100 presents an `E` ~4× outside
   its training band. **DMPO is structurally immune** because it z-scores the
   cost vector before the actor sees it — which is why it holds h100 while
   training at the same `max_delta=12`.

### Retracted: the "PLDM collapses at h100" finding

An earlier version of this record reported **PLDM h100 = 54.4** as a
base-specific long-horizon collapse. That was a **clipping artifact**: those
cells clipped to a symmetric `amax=2.5` inherited from the RLP recipe while
TwoRoom's real action range is ±1.4, so DMPO planned with actions the
environment cannot execute *and the world model never saw in training*, and the
out-of-distribution rollout error compounded over h100's 8 replans. Correcting
the bound recovers **54.4 → 92.7** (+38.3) and moves every other TwoRoom cell
by ≤1 point.

Superseded numbers: 96.7 / 99.1 (LeJEPA h25/h100), 97.3 / 54.4 (PLDM), MPPI
84.7 / 80.0. `TWOROOM_SPLIT=0` reproduces the authors' full-pool contract,
whose numbers are upper bounds (the leak was worth ~2–3 points at h25) and are
**not** comparable to the held-out RLP rows.

---

# 2. Reacher

**Status: PLDM current; LeJEPA cells (◇) still on the superseded symmetric
clip.** Jobs 7733/7735 (PLDM, env bounds ±1.73), 7018/7020/7022/7023 (LeJEPA,
`amax=2.2`). Data `tool=fetch_dataset dataset=reacher` (2,010,000 rows, h5),
h25, open loop (`receding_horizon=5`).

**Scoring**: *first-hit* success, latched on the first step whose worst joint is
within τ of the goal (`environment.success_threshold`) — the qpos-match task
hardcodes 0.05 rad and cannot express τ=0.1 through termination. Window-1 cells
train and evaluate against the single-frame critic; window-3 against the
3-frame window quasimetric.

| | | LeJEPA τ=.1 | τ=.05 | PLDM τ=.1 | τ=.05 |
|---|---|---|---|---|---|
| **(a) window-1** | **DMPO** (256 roll.) | 92.4 ◇ | 67.8 ◇ | **90.0** | **62.9** |
| | value MPPI (9k, same critic) | 64.7 | 40.0 | 68.0 | 46.7 |
| | RLP w1 ablation (paper†) | **98.7** | **88.7** | **97.8** | **82.0** |
| **(b) window-3** | **DMPO** (256 roll.) | 92.7 ◇ | 75.8 ◇ | **95.1** | **82.9** |
| | value MPPI (9k, same critic) | 83.3 | 66.0 | 88.0 | 68.0 |
| | RLP (paper) | **99.9** | **97.1** | **99.4** | **91.2** |

◇ measured with `amax=2.2` rather than the environment's per-dimension bounds.
The PLDM cells moved ≤1 point when re-measured with correct bounds (w1
90.4→90.0 / 62.4→62.9; w3 95.6→95.1 / 83.8→82.9), so the LeJEPA numbers are
expected to be stable — not yet confirmed.

1. **DMPO beats same-critic MPPI in all eight columns**, by +9 to +28 — the
   largest margins in the campaign. This is where the learned reduction most
   clearly earns its keep.
2. **RLP leads all eight columns**, by 6–21 points, widest at the tight
   tolerance (w3 LeJEPA: 97.1 vs 75.8).
3. **The window ordering reproduces for DMPO**: w3 > w1 at both bases and both
   tolerances, and the gain concentrates at τ=.05 (PLDM +20.0), matching the
   paper's ~+9 for a 3-frame critic at the tight tolerance.
4. **Scoring cross-check**: the in-job value-MPPI rows land within a few points
   of the paper's App C.3 value-MPPI rows (74.0/42.0 LeJEPA-w1, 86.0/66.0
   LeJEPA-w3), and τ=.05 reproduces the environment's own 0.05 rad termination
   criterion to within noise — the first-hit machinery added for this campaign
   (`rlp.environment.World.threshold_hits`) is calibrated.

---

# 3. OGBench Cube

**Status: superseded action clip; reruns pending.** These cells clip to a
symmetric `amax` inherited from the RLP recipe (LeWM 1.6, PLDM 4.5). Cube's
real bounds in z-scored units are [-3.50, +3.43], [-2.54, +2.55],
[-1.56, +1.55], [-2.55, +2.55], [-4.64, +3.37] — so **`amax=4.5` binds on only
1 of 5 dimensions (effectively no clip) while `amax=1.6` binds on 4 of 5**. The
two bases were searched over qualitatively different action sets and the
LeWM-vs-PLDM comparison below is confounded. Jobs 6758/6759 (train + h25),
7025/7026/7152/7153 (h25 + h100). Horizons h25 and h100 (`=100`/`=200`).

| planner | roll./decision | LeWM h25 | LeWM h100 | PLDM h25 | PLDM h100 |
|---|---|---|---|---|---|
| **DMPO** ◇ | 256 | **72.9** | **55.8** | 61.8 | 51.6 |
| value MPPI (same critic) | 9,000 | 62.7 | 50.0 * | **66.0** | 50.0 * |
| RLP (repo record, h25) | 9 | **86.9 ± 3.0** | — | **82.0** | — |
| no-op floor (record) | — | 54.0 | — | 54.0 | — |

\* 1–2 eval seeds only, measured before h100 MPPI was dropped from the grid (a
cell is 8 replans × 9,000 rollouts × 50 episodes); footnote-grade.

Per-seed h25 spread — LeWM 72.0 / 73.3 / 73.3, PLDM 60.7 / 62.7 / 62.0: tight
on both bases, so seed variance is not what separates the cells.

1. **LeWM: DMPO beats same-critic MPPI by +10.2** (72.9 vs 62.7) at 1/35th the
   rollouts. **PLDM: DMPO loses by 4.2** (61.8 vs 66.0) — the campaign's only
   MPPI loss, and the prime suspect is the clip confound above (PLDM ran
   effectively unclipped at `amax=4.5` with `init_std=1.0`).
2. **RLP leads both bases by ~14–20 points** at h25 on 9 rollouts per decision
   against DMPO's 256.
3. **Both DMPO cells clear the no-op floor** (54.0) at h25; at h100 LeWM stays
   above it (55.8) while PLDM sits at it (51.6), so treat the h100 numbers as
   near-floor until re-measured.
4. Horizon costs DMPO ~10–17 points on both bases — but on TwoRoom the
   equivalent drop proved to be largely a clipping artifact, which is a further
   reason not to read into these before the rerun.

---

# 4. Wall-clock per decision (job 7868)

Cube LeWM on one H200, `evaluation.num_episodes=50` (the protocol's batch), h100
budget 500 so each arm yields 20 decisions. **Steady median** excludes the first
solve, which carries warmup and — in graphed mode — the multi-second CUDA-graph
capture; a mean over 8 solves is dominated by that outlier and is how an earlier
version of this table wrongly showed graphing as a 3.9× regression.

| planner | mode | first solve | steady median | ms/env | rollouts/decision |
|---|---|---|---|---|---|
| **RLP / LIP** | graphed | 2182 | **80.1** | **1.60** | 9 fwd (+8 bwd) |
| RLP / LIP | eager | 747 | 319.5 | 6.39 | 9 fwd (+8 bwd) |
| **DMPO** | graphed | 2364 | 151.5 | 3.03 | 256 fwd |
| DMPO | eager | 651 | 159.1 | 3.18 | 256 fwd |
| CEM | eager, `batch_size=50` | 10342 | 1976.9 | 39.54 | 9,000 fwd |
| MPPI | eager, `batch_size=50` | 10390 | 4766.8 | 95.34 | 9,000 fwd |

1. **Graph capture buys RLP 4.0× and DMPO 1.05×.** RLP's decision is 8
   sequential iterations of a 5-step unroll plus backward — launch-bound, which
   is what capture removes. DMPO is one wide forward batch of `B*N` plans (5
   dependent steps), with almost no launch overhead to recover. Graphed DMPO is
   bit-exact against eager (`max_difference = 0.000e+00` on every
   `graphed=verify` call).
2. **Best-config latency: RLP 1.60 vs DMPO 3.03 ms/env** — RLP 1.9× faster on
   top of 28× fewer rollouts. Eager-only the ordering *reverses* (DMPO 3.18 vs
   RLP 6.39), so any latency claim must state the mode.
3. **Both learned planners are an order of magnitude faster than the sampling
   baselines**: 13–25× versus CEM, 31–60× versus MPPI.
4. The shipped `configs/core/solver/{cem,mppi}.yaml` set `batch_size: 1`, i.e.
   50 sequential single-env solves; the rows above use `batch_size=50`. As
   shipped, CEM costs 144 ms/env and MPPI 222 ms/env (job 7761) — a ~2.3×
   config penalty that applies to every baseline timing in this repository.

---

# 5. Scope, gaps, and provenance

**What this campaign measures.** *Offline* DMPO: the paper's update rule and
inner loop, trained by pathwise gradients against the shared critic through the
frozen world model — the same regime RLP trains in. It measures optimizer
quality under a given cost, **not** the paper's return-driven model-error
compensation, which requires on-system interaction (see
[README_dmpo.md](README_dmpo.md)). `model=dmpo_ppo` implements the paper's PPO
objective closed inside the world model and is available but unused here.

**Rollout budgets are not matched, and cannot be.** RLP spends 9 forward
unrolls per decision against DMPO's 256, so DMPO loses while spending 28× more
rollouts. A rollout-matched DMPO row is ill-posed: 9 samples in Cube's
125-dimensional plan space is fewer samples than dimensions, below what a
Gaussian sampler can estimate at all. That asymmetry — a gradient refiner needs
O(10) unrolls where a sampler needs O(100+) — is structural, not a tuning gap.

**No hyperparameter selection was performed for DMPO.** Architecture and
inner-loop settings (256 samples, 1 iteration, temperature 0.05, step size 0.8,
gate, multiplicative covariance, 256-unit hidden layer, `N(0,1e-3)` last layer)
are the authors' `quadrotor_dmpo_zigzagyaw.yml` values; trainer settings (2000
steps, batch 32, lr 1e-4→1e-5) are defaults. RLP's rows went through selection
on eval seeds 50/51; DMPO's did not. The bias runs against DMPO, but an
`amax`/`init_std` grid on seeds 50/51 remains the obvious next check —
especially for Cube PLDM, the one MPPI loss.

**Open items.** Cube LeWM/PLDM and Reacher LeJEPA w1/w3 need re-measurement
under the environment's action bounds; those reruns (`EVAL_PAR=2`) are pending a
SkyPilot client upgrade, the API server having begun dropping job submissions.

## Provenance

| cell | jobs (train + eval) | tag |
|---|---|---|
| Cube LeWM ◇ | 6758 → 7025/7152 | `dmpo-cube-lewm-20260814` |
| Cube PLDM ◇ | 6759 → 7026/7153 | `dmpo-cube-pldm-20260814` |
| Reacher LeJEPA w1/w3 ◇ | 7018 / 7022 | `dmpo-re-lejepa-w{1,3}-20260815` |
| Reacher PLDM w1/w3 | 7733 / 7735 | `dmpo-re-pldm-w{1,3}-ab-20260817` |
| TwoRoom LeJEPA/PLDM | 7730 / 7731 | `dmpo-tw-{lejepa,pldm}-ab-20260817` |
| TwoRoom (split, superseded clip) | 7689 / 7690 | `dmpo-tw-{lejepa,pldm}-split-20260817` |
| TwoRoom (full pool, superseded) | 7340 / 7341 | `dmpo-tw-{lejepa,pldm}-20260815` |
| Wall-clock benchmark | 7868 | — |

Per-cell result files live under
`/checkpoints/armin@pantheon.inc/<tag>/results_*.txt` on the cluster volume.

## Ops notes (fixes committed)

Pipefail-killed dataset discovery (`|| true`); TwoRoom pool regeneration
infeasible in-job (3,443/10,000 episodes in 11 h, decelerating) so the authors'
h5 is fetched instead, with dedicated eval roots for its `pos_agent` column;
first-hit tolerance scoring for Reacher's τ=0.1 column; resumable eval cells;
4-wide parallel evals requiring per-process `OMP_NUM_THREADS` caps (default
threading made 4-wide *slower than sequential* — zero cells in 3.2 h); a
zero-cell run now fails the job instead of printing `DMPO_CAMPAIGN_DONE`; and
`EVAL_PAR=8` starves heavy environments (Cube, Reacher-LeJEPA cells stalled ~36
min then were killed with no traceback). The final harness — H200:4 / 80 cores,
8 cells wide, three optimizer seeds trained concurrently one per GPU — runs a
complete TwoRoom cell (caches, critic, three seeds, 21 eval cells) in **21
minutes**, against ~11 h for the 1-GPU sequential version.
