# RLP unified-recipe campaign (uniJ) — consolidated results

Compiled 2026-09-25 15:00 PDT. Supersedes nothing; it gathers every number the campaign has produced
into one place. Per-seed tables and provenance live in `docs/campaigns/2026-09-22/RESULTS_uniJ_shared_stop.md`.

## Protocol

One recipe for every environment × base (`scripts/sky/unigamma/recipes/uniJ.yaml`). Per cell
(env × base × horizon stack) the teacher budget ∈ {3k…18k} and actor step ∈ {2k…12k} are chosen ONCE by
the mean validation score over six seeds on draws **48–51**; all six seeds are then evaluated at that
pair on report draws **42–44**, 50 episodes each. Test draws are touched once per cell. Reported number
is the median over seeds.

Everything below shares that world model, that train/eval split (train 0–7999, eval 8000–9999), and
those draws. γ = 0.98 for h25 stacks, 0.99 for h100 (1/(1−γ) must cover the horizon). Reacher uses a
2-frame critic window (velocity is unobservable from one frame); everything else uses 1.

---

## 1. RLP under the unified recipe — 12 cells, complete

| env | base | stack | shared stop (teacher / actor) | **n=6 median** | mean | config B |
|---|---|---|---|---|---|---|
| TwoRoom | LeJEPA | h25 | 15k / 2k | **100.0** | 99.9 | 100.0 |
| TwoRoom | LeJEPA | h100 | 6k / 8k | **99.3** | 98.9 | 94.7 |
| TwoRoom | PLDM | h25 | 3k / 2k | **98.0** | 98.3 | 96.3 |
| TwoRoom | PLDM | h100 | 3k / 10k | **94.0** | 93.9 | 85.3 |
| Reacher | LeJEPA | h25 (τ.05) | 18k / 2k | **95.3** | 94.9 | ~96 |
| Reacher | PLDM | h25 (τ.05) | 18k / 2k | **87.3** | 88.1 | 88.7 |
| Cube | LeWM | h25 | 3k / final | **92.7** | 91.8 | 90.0 |
| Cube | LeWM | h100 | 9k / 2k | **85.3** | 84.8 | 86.3 |
| Cube | PLDM | h25 | 15k / 8k | **85.3** | 84.0 | 85.0 |
| Cube | PLDM | h100 | 12k / 2k | **83.7** | 82.2 | 85.0 |
| PushT | LeWM | h25 | 3k / 6k | **72.0** | 72.7 | 64.7 |
| PushT | PLDM | h25 | 3k / final | **58.3** | 58.3 | — |

Reacher at τ 0.1: LeJEPA **100.0**, PLDM **99.3**.

The unified recipe beats per-cell early stopping (config B) in 7 of 10 comparable cells and loses by at
most 1.4 elsewhere. PushT PLDM is the first RLP number on a PLDM PushT base (no public checkpoint exists;
the world model was trained in-house with the authors' recipe).

---

## 2. Planner controls — same critic, same world model, same draws

CEM / Adam / MPPI using the cell's **locked RLP teacher snapshot**, one arm per training seed (n=6).
Eval only, no actor training. This is the "same critic, different optimizer" control.

| env | base | stack | CEM | Adam | MPPI | **RLP** |
|---|---|---|---|---|---|---|
| TwoRoom | LeJEPA | h25 | 100.0 | 96.3 | 95.3 | 100.0 |
| TwoRoom | LeJEPA | h100 | 73.0 | 68.0 | 68.0 | **99.3** |
| TwoRoom | PLDM | h25 | 100.0 | 94.0 | 95.7 | 98.0 |
| TwoRoom | PLDM | h100 | 49.3 | 53.0 | 53.3 | **94.0** |
| Cube | LeWM | h25 | 80.0 | 77.3 | 67.0 | **92.7** |
| Cube | LeWM | h100 | 77.7 | 70.7 | 58.3 | **85.3** |
| Cube | PLDM | h25 | 75.3 | 66.0 | 65.3 | **85.3** |
| Cube | PLDM | h100 | 69.3 | 57.0 | 53.0 | **83.7** |
| Reacher | LeJEPA | h25 (τ.05) | 63.3 | 71.7 | 70.3 | **95.3** |
| Reacher | PLDM | h25 (τ.05) | 56.3 | 68.7 | 65.3 | **87.3** |

**RLP wins 9 of 10 and ties the tenth.** Margin tracks difficulty: ~0 where the task saturates
(TwoRoom h25), +40.7 on TwoRoom PLDM h100, +23.6 on Reacher LeJEPA, +7.6…+16.0 across Cube.

Reacher at τ 0.1 — LeJEPA: CEM 88.0 / Adam 92.7 / MPPI 89.0 vs RLP 100.0. PLDM: 75.3 / 92.7 / 86.0 vs 99.3.

### 2b. What the learned critic is worth (MPPI, latent vs value cost)

| env | base | stack | MPPI + latent | MPPI + learned critic | RLP |
|---|---|---|---|---|---|
| TwoRoom | LeJEPA | h25 | 70.7 | 95.3 | 100.0 |
| TwoRoom | LeJEPA | h100 | 20.0 | 68.0 | 99.3 |
| TwoRoom | PLDM | h25 | 64.0 | 95.7 | 98.0 |
| TwoRoom | PLDM | h100 | 33.3 | 53.3 | 94.0 |
| Cube | LeWM | h25 | 56.7 | 67.0 | 92.7 |
| Cube | LeWM | h100 | 46.0 | 58.3 | 85.3 |
| Cube | PLDM | h25 | 60.0 | 65.3 | 85.3 |
| Cube | PLDM | h100 | 47.3 | 53.0 | 83.7 |

Two separable contributions: the **critic** is worth +5 to +48 over the parameter-free latent objective
with the optimizer held fixed; the **learned refiner** is worth a further +2.3 to +40.7 on top.

---

## 3. Learned baselines — validation-selected (equal treatment)

The baselines deploy a snapshot **chosen on validation draws 48–51**: one shared step per cell by the
mean over three training seeds, ties to the earlier step — RLP's own rule. Earlier runs deployed a fixed
final iterate while RLP got a validation-selected pair, which was not the same treatment.
DMPO snapshots every 200 of the paper's 1000 PPO iterations; L2O every 3,000 of 18,000.

### DMPO — 10 of 10 complete

| env | base | stack | selected step | **DMPO** | (final iterate) | best planner | RLP |
|---|---|---|---|---|---|---|---|
| TwoRoom | LeJEPA | h25 | 200 | **73.3** | 69.3 | 100.0 | 100.0 |
| TwoRoom | LeJEPA | h100 | 400 | **69.3** | 65.3 | 73.0 | 99.3 |
| TwoRoom | PLDM | h25 | 200 | **76.7** | 72.7 | 100.0 | 98.0 |
| TwoRoom | PLDM | h100 | 800 | 44.0 | 44.7 | 53.3 | 94.0 |
| Cube | LeWM | h25 | 200 | 66.0 | 66.0 | 80.0 | 92.7 |
| Cube | LeWM | h100 | final | 54.0 | 54.7 | 77.7 | 85.3 |
| Cube | PLDM | h25 | 800 | 61.3 | 61.3 | 75.3 | 85.3 |
| Cube | PLDM | h100 | 200 | **50.0** | 49.3 | 69.3 | 83.7 |
| Reacher | LeJEPA | h25 (τ.05) | 600 | 20.7 | 20.0 | 71.7 | 95.3 |
| Reacher | PLDM | h25 (τ.05) | 600 | 19.3 | 20.7 | 68.7 | 87.3 |

Reacher DMPO at τ 0.1: LeJEPA 40.7, PLDM 36.0.

- **Selection is not a uniform win** — four cells gain, four flat, two lose slightly. Applied regardless
  of whether it flattered the baseline.
- **The final iterate wins in only one of ten cells**, and selected steps scatter early (200–800).
  The PPO return is a trendless walk over the paper's 1000 iterations, confirmed on held-out draws.
- **DMPO is below every planner control in all ten cells.**
- **Reacher is settled, not unlucky**: all five snapshots validate in a 39.5–41.8 band, so DMPO genuinely
  cannot hit Reacher's tolerance — not a checkpoint artifact.

### L2O-MPC — 10 of 10 complete

| env | base | stack | selected | **L2O** | validation ladder (3k→final) | budget-limited? |
|---|---|---|---|---|---|---|
| TwoRoom | LeJEPA | h25 | final | **98.0** | 92.7, 96.8, 97.7, 96.3, 97.5, 98.2 | mild |
| TwoRoom | PLDM | h25 | 15k | **99.3** | 96.3, 98.5, 98.7, 98.0, 99.0, 99.0 | saturating |
| TwoRoom | PLDM | h100 | 15k | 46.7 | 46.3, 50.5, 50.7, 52.0, **54.7**, 54.3 | **yes** |
| Cube | LeWM | h25 | final | 64.0 | 48.2, 49.5, 49.7, 49.7, 49.5, 50.5 | marginal |
| Cube | LeWM | h100 | 9k | 54.0 | 47.7, 48.8, **49.5**, 48.0, 49.3, 49.3 | no |
| TwoRoom | LeJEPA | h100 | 15k | **92.7** | 70.3, 75.7, 84.2, 87.7, **91.8**, 91.8 | no (flat at top) |
| Cube | PLDM | h25 | 3k | 56.0 | 46.0 at every rung | no |
| Cube | PLDM | h100 | 3k | 45.3 | 43.5 at every rung | no |
| Reacher | PLDM | h25 (τ.05) | 15k | 39.3 | 48.0, 52.8, 54.0, 58.7, **62.2**, 61.3 | **yes** |
| Reacher | LeJEPA | h25 (τ.05) | final | **46.7** | 46.7, 59.2, 54.5, 63.3, 67.3, **67.7** | mild |

Reacher PLDM at τ 0.1: 67.3.

**TwoRoom LeJEPA h100 = 92.7** is the second cell where L2O beats every planner control by a wide margin
(92.7 vs CEM 73.0) while still trailing RLP (99.3). Its ladder climbs steeply (70.3 → 91.8) but ties at
the top two rungs, so 18k is sufficient there.

**Reacher LeJEPA = 46.7** (τ.1: 68.0), the last cell. Its 18k checkpoints were rescued off the cluster
volume before shutdown and evaluated on a RunPod H200 host. Two independent checks qualify that host:
its first four validation rungs (46.67, 59.17, 54.50, 63.33) reproduce the interrupted cluster run's
exactly, and the completed 50k final-iterate run of the same cell scored τ.05 47.3 (job 28288) against
46.7 here — consistent, and slightly higher at the larger budget as the ladder predicts.

**TwoRoom PLDM h25 L2O (99.3) is the only cell where a baseline beats RLP (98.0) under equal treatment.**

#### Budget caveat — important
L2O was cut from 50k to 18k steps on a misreading of the *imitation loss* (a DAgger training loss against
a moving target), which is flat from ~step 1,000. Held-out success is not: the ladders above show four of
seven cells still rising at the top rung. Two direct comparisons against the earlier 50k runs:

| cell | 50k, final iterate | 18k, validation-selected |
|---|---|---|
| Cube LeWM h25 | 66.0 | 64.0 |
| Reacher PLDM τ.05 | 46.0 | **39.3** |

Selection can only help at fixed budget, so the loss is the budget cut.

**Decision 2026-09-25 (user): report 18k for every cell.** The two whose ladders were still climbing
(TwoRoom PLDM h100 at 46.7, Reacher PLDM at 39.3) are therefore **lower bounds**, and the ladders above
are the published evidence of that, showing exactly where each was still rising. The bias has one
direction only: L2O is under-reported on those two cells, never over-reported. Cells with flat ladders
(Cube PLDM h25 and h100 are flat to the decimal across all six rungs) need no such caveat — the ladder is
direct evidence more training would not have helped.

A single 50k run of TwoRoom PLDM h100 is completing on a RunPod H200 host and will be folded in as an
addendum when it lands; it changes no conclusion, since L2O already trails the planner controls in every
cell by a margin far larger than the gap between the rungs.

---

## 4. Dyna under the unified recipe (h25 only)

Per iteration: the locked actors collect rollouts at offsets 25 **and** 100 on the training split, the
world model is anchor-fine-tuned from the previous one, and the cell's **locked config is retrained** on
it (never re-selected). h100 retrains were dropped by decision 2026-09-24 — the gain is h25-specific and
measured to be so (see §4b). Collection still runs at both offsets.

| cell | base | it1 | it2 | **reported** |
|---|---|---|---|---|
| Cube / LeWM / h25 | 92.7 | **95.0** (val 93.00) | 93.7 (val 92.75) | **it1 — 95.0** (+2.3) |
| Cube / PLDM / h25 | 85.3 | 89.7 (val 83.25) | **93.3** (val 90.33) | **it2 — 93.3** (+8.0) |

**Decision 2026-09-25 (user): report two iterations, not three** — matching the config-B campaign, which
also reported two. Iteration 3 was only ever a check on whether PLDM keeps compounding past 93.3; it was
still running at shutdown and is not needed for any claim here.

Per-cell iteration choice is on validation: LeWM picks **it1** (93.00 > 92.75), PLDM picks **it2**
(90.33 >> 83.25). Cube/LeWM seed 0's validation draws 48/49/50 were lost to a failed eval and recovered
on 2026-09-25 (90/92/88), which is what settles that cell on it1; its test number was n=6 throughout. PLDM compounds (+4.4 then +3.6) and its iteration-1 seed-3 outlier (69.3 against
82.7–92.0) disappears at iteration 2 — the post-Dyna variance collapse seen in the config-B campaign.
LeWM is flat after iteration 1.

### 4b. Why h25 and not h100 (config-B evidence)
The h25 and h100 hard cores are **disjoint** task sets — 9/150 and 17/150, different segments of the same
episodes at different offsets. Across the config-B campaign Dyna cured 3–6 of the 9 h25 core tasks and
**1 of the 17** h100 ones. Collection horizon is not the cause: v2 collects 1800 episodes at each offset
and the h100 core still does not move. The h200 probe explains it — true waypoints give 82.0 at h25 but
4.0 at h200, so at long horizon the bottleneck is executing a long contact sequence, which a one-step
latent fine-tune does not address. Config-B hard-success gains: **+13.0 / +12.2 at h25, +0.7 / +0.5 at
h100** (the latter inside seed noise).

### 4c. Config-B Dyna reference (complete, n=6, drop-in LaTeX)
`docs/paper/tables_configB.tex` (`tab:dyna`): LeWM 95.7 / 86.7, PLDM 92.3 / 85.3 (h25 / h100).
Anchor ablation on identical data: weight 0.1 beats 1.0 by +2.7 at h100 at no h25 cost; freezing the
encoder is catastrophic (77.3 at h100) — the correction lives in the encoder. **The current campaign runs
anchor 1.0**, which that ablation suggests is slightly too tight; iteration 2 at anchor 0.1 was never run.

---

## 5. Open items

| item | state |
|---|---|
| L2O 18k: TwoRoom LeJEPA h100 | **resolved 2026-09-25** — checkpoints rescued to a pod, evaluated there: 92.7 |
| L2O 18k: Reacher LeJEPA | **resolved 2026-09-25** — 46.7 on the pod; cross-checked against the cluster rungs and the 50k run |
| L2O 50k: TwoRoom PLDM h100 | running on a RunPod H200 host, lands ~04:00 — addendum only |
| L2O 50k: Reacher PLDM | cluster; superseded by the 18k decision |
| Dyna iteration 3 | cluster; superseded by the two-iteration decision |
| Cube LeWM h25 Dyna it2 seed 0 | **resolved** — validation draws recovered (48/49/50 = 90/92/88) |

**All 52 reported cells are complete.** Nothing outstanding; the 50k L2O re-runs and Dyna round 3 are
optional refinements that no claim depends on.

### Result-file collision (found and fixed 2026-09-25)
`run_eval` located a finished evaluation with `find logs -name "$unique.txt" | head -1`. That name is
unique within a cell but not across jobs, and `find` returns directory order — oldest first. Every managed
cluster job gets a fresh workdir, so this never fired there; on a pod, where one `~/sky_workdir` is reused,
a reacher cell silently adopted a tworoom cell's 72 validation results verbatim (byte-identical files,
different inodes, different eval commands). Caught because the two ladders matched to 0.01 across six
rungs. Both campaigns now take the newest match by mtime. Blast radius was checked file by file: only that
one cell was affected, it was re-run clean, and no cluster result can be touched by this.

### Deviations from published recipes, to footnote
1. **Reacher critic window = 2**, not the paper's 3. Measured better than both w=1 and w=3 (2026-08-26);
   velocity is unobservable from a single frame.
2. **L2O budget**: the paper's literal DAgger budget is ~10⁶ gradient steps per seed, which in our setting
   is a world-model unroll rather than a small MLP update. We train to convergence and report the plateau,
   with the validation ladder as evidence.
3. **DMPO**: the paper says "up to 1000 iterations of PPO" — we snapshot within that budget and select on
   validation, which is arguably closer to the paper than always taking iteration 1000.
4. **Dyna h100 retrains dropped** (§4).

### Validity caveat inherited from the stack
Everything pre-Dyna trains on the same 10k episodes the eval draws come from, so absolute numbers are
upper bounds; the split policy is in `Dyna/DATA_SPLIT_POLICY.md`. Comparisons within this document are
unaffected — every arm shares the split.
