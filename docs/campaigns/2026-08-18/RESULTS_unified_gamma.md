# Unified-γ cross-environment campaign — 2026-08-18

Question: can RLP's critic discount be unified across environments, and at
what cost per cell? The unified-config proposal argued γ=0.98/n=50 everywhere
("ordering-invariance; finite unreachable target"); TwoRoom falsified it
(§10/§10a of [RESULTS_tworoom_improvement.md](../2026-08-17/RESULTS_tworoom_improvement.md):
h100 collapses to 66–73 because V saturates at 1/(1−γ)=50 and grad V goes
flat). The revised proposal — γ as a horizon budget, γ ≈ 1 − 1.5/d_max, i.e.
**γ=0.99** with worst-case d_max≈150 — is the headline arm here.

## Design (user decisions pinned 2026-08-18)

- **Train everything ONCE at amax=2.5, max_delta=12** — no per-cell clip, no
  horizon-matched actor pairs. One actor per (γ × vnorm × seed), evaluated at
  both horizons from the same checkpoint. amax is adapted only at deployment
  (post-hoc eval pass with the checkpoint's top-level `amax` rewritten; the
  solver reads deploy amax exclusively from the checkpoint —
  `overlays/lip.py`, no config path exists).
- **Arms: γ ∈ {0.98, 0.99, 1.0} × vnorm ∈ {none, log}, n-step 50.** One γ
  point is each environment's bespoke value (TwoRoom 1.0; Cube/Reacher 0.98),
  so the fleet is a 3-point dose–response per environment with the bespoke
  value as an internal anchor.
- **Cells:** TwoRoom LeWM/LeJEPA (h25+h100, held-out SPLIT=1), OGBench Cube
  LeWM (h25+h100, held-out by construction), Reacher LeWM/LeJEPA (latched
  first-hit tau 0.1 and 0.05). Train seeds 0/1/2, report draws 42/43/44,
  50 episodes/cell.
- Everything else stays at each cell's established recipe (mw/lr anchors,
  steps/batch, teacher expectile/steps), so γ and the clip are the only
  deltas against banked per-cell numbers.

Infrastructure: `scripts/sky/unigamma/launch_unig_fleet.sh` (smoke → fleet),
`GRID=unig` in `tworoom_g98_rescue.yaml` (TwoRoom + Cube; `CU_GAMMA` threads
the cube discount — `TR_GAMMA` never reached cube), and
`reacher_gamma.yaml` (`RS_GAMMA`/`RS_VNORM` threading; the vnorm actor port
`overlays_reacher_vnorm/` is now wired). WM checkpoints come from the
volume stage (`stage_lfs.yaml` + `stage_probe.yaml`, `cp -rL`); the GitHub
LFS budget is exhausted and the W&B-artifact restore is fallback only.

## Banked context (already measured, TwoRoom LeWM/LeJEPA h100 @ a1.8/md20)

From jobs 7913–7919, 8022–8024 (collector: `collect_unig.yaml`,
TAG_GLOB `rlp-tw-g9*-20260818`):

| critic | raw E (ctrl) | vnorm=log |
|---|---|---|
| γ=1.0 / n50 (bespoke) | 88.67 (seed 3) | **97.33** (seed 3) |
| γ=0.98 / n50 (unified proposal) | 66.22 ± 21.7 | 81.78 ± 3.7 |
| γ=0.98 / n100 (≡ n200, see below) | 85.11 ± 3.3 | 87.33 ± 2.7 |
| γ=0.996 / n50 ("0.98 per block") | 87.11 ± 1.0 | 91.33 ± 2.4 |

Three observations that shape expectations for the fleet:

1. **The 0.98 collapse is a window problem as much as a discount problem.**
   Widening the exact-label window (n100) recovers +19 points and cuts seed
   sd 7×; moving the ceiling out via per-block discounting (γ=0.996, ceiling
   250 steps) recovers +21. Both rescue arms stay 6–10 points below the
   γ=1.0+vlog record — the discount still costs real accuracy at range.
2. **n=100 and n=200 trained bit-identical actors** (all 18 per-draw scores
   equal): TwoRoom's episode length caps sampled goal deltas at ~100
   primitive steps, so the n-step window saturates. n>100 is not a separate
   arm on this environment.
3. γ=0.99/n50 (ceiling 100) sits between the falsified 0.98 (ceiling 50) and
   the rescue 0.996 (ceiling 250). The horizon-budget rule predicts it
   should survive h100-with-drift; the §10 saturation mechanism predicts its
   d=100 slope (0.37/step) clears the critic noise floor that 0.98's
   0.13/step did not. This is exactly what the fleet measures.

PLDM matrix arms (banked, same protocol, PLDM base): γ=0.996 h100 ctrl
88–92 across seeds; γ=0.98/n100 ~85–87; γ=1.0 reference partial (job
cancelled mid-run; 88.0 on the completed seed). Not this campaign's target
cells but consistent with the LeJEPA ordering.

## Fleet (in flight)

18 jobs × H200:4 = 72 GPUs, p1: 9 TwoRoom (γ × seed sharded; jobs
8269–8281), 3 Cube (γ, seeds packed; 8305–8307), 6 Reacher (γ × vnorm,
seeds in-job; 8277–8293). Smoke canaries (one per environment, γ=0.99;
8257/8258/8260, all SUCCEEDED) gated each slice; the reacher smoke's
checkpoint was torch.loaded to prove vnorm=log/amax=2.5/gamma=0.99
round-tripped (reacher has no [args-ok] guard).

Tags: `rlp-tw-unig-{g98,g99,g100}-s{0,1,2}-20260818`,
`rlp-cu-unig-{g98,g99,g100}-20260818`,
`rlp-re-unig-{g98,g99,g100}-{none,log}-20260818`.

## Results

### TwoRoom LeWM (h25 / h100) — jobs 8269–8281, complete

One actor per arm at amax=2.5/md12, evaluated at both horizons; seeds 0/1/2,
draws 42/43/44 (450 episodes per cell):

| arm | h25 | h100 | h100 per-seed |
|---|---|---|---|
| γ=0.98 ctrl | 100.0 | **88.0** ± 9.2 | 93.3 / 93.3 / 77.3 |
| γ=0.98 vlog | 99.8 | 74.9 ± 8.4 | 73.3 / 84.0 / 67.3 |
| γ=0.99 ctrl | 100.0 | **88.0** ± 8.7 | 92.0 / 78.0 / 94.0 |
| γ=0.99 vlog | 100.0 | 85.8 ± **0.8** | 86.7 / 85.3 / 85.3 |
| γ=1.0 ctrl | 100.0 | **87.8** ± 3.9 | 90.7 / 83.3 / 89.3 |
| γ=1.0 vlog | 99.8 | 88.4 ± 3.4 | 84.7 / 91.3 / 89.3 |

**Headline: at amax=2.5 the three discounts are indistinguishable at h100
(88.0 / 88.0 / 87.8 raw-E), and h25 is at ceiling everywhere.** The §10/§10a
γ=0.98 collapse (66–73) does not reproduce at the wider clip — with means
±5 at 3 seeds, the arms are within noise of each other.

**Reinterpretation of the §10 falsification: it was a γ × clip interaction,
not pure critic saturation.** The §2 mechanism (actor response ‖dA‖ linear
in E; the clamp applied to the cumulative plan) predicts failure when E's
drive pushes plan entries onto the ±amax boundary. At γ=0.98 the critic's
E ≈ 43–50 everywhere far; at amax=1.8 that drive pins the plan (collapse);
at amax=2.5 the boundary is out of reach and the same critic plans fine.
Consequences:
- "TwoRoom's bespoke γ=1.0 is load-bearing" holds **only at the bespoke
  a1.8 clip**. Under the unified clip, γ genuinely unifies on TwoRoom.
- `vnorm=log`'s banked +8.7 (and the 6.7× variance cut) was likewise
  clip-specific: at a2.5 it is neutral at γ=1.0 (88.4 vs 87.8), **harmful
  at γ=0.98 (74.9 vs 88.0)** — log1p on an already-discount-compressed E
  over-compresses. γ=0.99+vlog is the variance pick (sd 0.8 vs 8.7 raw).
- The unified config costs ~9 pts at h100 against the banked tuned best
  (97.3 at a1.8/md20/vlog/γ1.0, horizon-matched training) and 0 at h25.
  How much of that is the md12 band vs the clip is not resolved here — the
  banked 97.3 carried an md20 goal band that covers the 100-step goal.

### Reacher LeWM (tau 0.1 / 0.05) — jobs 8277–8293, complete

One actor per arm at amax=2.5 (mw 0.3, lr 1e-4, expand 0, replay 0.5 — the
paper LeWM shape at the unified clip), seeds 0/1/2, draws 42/43/44.
**Convention: held-at-end** (held10 ≈ tau 0.1 worst-joint sweep, held05 =
qpos-in-ball residency ≈ dm_control's 0.05 geom) — `GRID=cross` routes actor
evals through `evone`, not the latched-first-hit passes (those are gated to
the latched grids). Do not compare these numbers against latched-protocol
tables.

| arm | tau 0.05 (held) | tau 0.1 (held) | seed means (0.05) |
|---|---|---|---|
| γ=0.98 ctrl (bespoke γ) | **55.1** ± 6.5 | **94.0** ± 0.7 | 51.3 / 62.7 / 51.3 |
| γ=0.98 vlog | 57.6 ± 6.2 | 94.7 ± 5.3 | 54.0 / 54.0 / 64.7 |
| γ=0.99 ctrl | 40.9 ± 9.7 | 85.3 ± 12.2 | 30.7 / 50.0 / 42.0 |
| γ=0.99 vlog | 40.9 ± 23.2 | 83.1 ± 17.4 | 14.7 / 58.7 / 49.3 |
| γ=1.0 ctrl | 46.7 ± 9.8 | 87.6 ± 7.3 | 35.3 / 52.7 / 52.0 |
| γ=1.0 vlog | 34.2 ± 23.0 | 76.9 ± 22.8 | 8.7 / 40.7 / 53.3 |

**Reacher runs opposite to TwoRoom: the bespoke γ=0.98 wins outright** —
raising γ costs 7–9 points at tau 0.1, ~10–14 at tau 0.05, and inflates seed
variance (seed 0 craters to 8.7–14.7 under γ≥0.99+vlog). Mechanism reading:
Reacher has no far field for a discount to protect (d_max ≈ tens of steps),
so γ→1 buys nothing on target spacing and loses the per-hop bootstrap
contraction — the co-trained critic drifts, and precision at the goal
(exactly what tau 0.05 scores) degrades first. The horizon-budget rule
γ ≈ 1 − 1.5/d_max with a small d_max in fact prescribes γ ≤ 0.98 here; the
unification failure is in forcing γ *up*, not down. vnorm=log is neutral at
γ=0.98 and amplifies the γ≥0.99 instability.

### OGBench Cube LeWM (h25 / h100) — jobs 8477–8479 (+watchdog fix)

The first cube wave (8305–8307) failed on a harness bug, not science: the
silent-hang watchdog's `pgrep -f '<cell>.pt'` also matched eval commands
(they embed the actor filename), and cube's node-serialized evals queue past
`TRAIN_STALL_SEC` — healthy queued evals were stack-dumped and killed. Fixed
(match `train_lip_ac.py` only) and rerun; all actors had trained fine and
were reused.

| arm | h25 | h100 |
|---|---|---|
| γ=0.98 ctrl (bespoke γ) | **87.0** ± 1.4 | **81.3** ± 0.9 |
| γ=0.98 vlog | 89.0 ± 0.5 | 81.3 ± 2.8 |
| γ=0.99 ctrl | 86.0 ± 1.2 | 78.4 ± 2.1 |
| γ=0.99 vlog | 87.1 ± 1.5 | 80.0 ± 0.7 |
| γ=1.0 ctrl | 86.7 ± 1.3 | 80.0 ± 0.7 |
| γ=1.0 vlog | 86.9 ± 1.4 | 81.6 ± 1.4 |

(γ=0.98 row: seeds 0/1 at collection time; seed 2 in flight in the rerun.)

**Cube is γ-flat under the unified clip** (spread ≤ 3 pts, bespoke 0.98
nominally best), and γ=1.0 shows no collapse at h100 — cube's operating
range sits inside every convention's informative band, so the exact branch
dominates. Against the banked tuned reference (89.1 / 82.4 at amax 1.6,
md 10/20, 12k-critic), the unified config costs only ~2 / 1 pts — far
gentler than TwoRoom's −9.

### TD-target seam (user observation): real in target space, cosmetic in behaviour

The n-step target is **non-monotone at the window boundary for γ<1**: the
sampler labels in-window goals with RAW delta (`dist[b] = float(delta)`,
`trm/samplers.py`) while the out-of-window branch bootstraps with DISCOUNTED
accumulation — at γ=0.98/n=50 the trained target is V(50)=50 vs V(51)≈32.2,
a −17.8 cliff exactly at the seam (γ=0.99: −9.9; γ=1: none). Note the cliff
is γ-correlated, so the legacy γ dose–response carried it as a potential
confound. Two fixes implemented behind `--boundary` in both teachers and
both co-train trainers (`overlays_boundary/`): **smooth** = `boot = n_eff +
γ^n_eff·d_next` (continuous seam, per-window discounting, ceiling
n/(1−γⁿ) = 78.6 at 0.98/n50, 126.6 at 0.99); **disc** = discount the exact
branch too (the textbook discounted quasimetric).

Results (ctrl arms, unified config; jobs 8511–8543):

| cell | legacy | smooth | disc |
|---|---|---|---|
| TwoRoom h100, γ=0.98 | **88.0** ± 9.2 | 85.8 ± 5.0 | 85.8 ± 1.9 |
| TwoRoom h100, γ=0.99 | **88.0** ± 8.7 | 82.7 ± 2.3 | 86.7 ± 1.8 |
| Reacher τ0.05/τ0.1, γ=0.98 | **55.1**/**94.0** | 55.1/90.0 | 51.8/90.2 |
| Reacher τ0.05/τ0.1, γ=0.99 | 40.9/85.3 | 32.0/77.1 | **43.3**/**85.8** |

(TwoRoom h25 is 100.0 everywhere; γ=1.0 needs no arms — both fixes reduce
to the identical target at γ=1.)

**Verdict: the seam is behaviorally cosmetic — no fix beats legacy anywhere,
and smooth is mildly harmful** (worst at γ=0.99 on Reacher). Reading:
(1) the expectile-Huber fit plus the MRN's subadditivity already smooth over
the cliff in function space; (2) the eval-relevant queries rarely straddle
the seam (Reacher's operating range is in-window; TwoRoom's refiner descends
local grad V at plan sites); (3) smooth's continuity is bought by extending
the target ceiling (50 → 78.6/126.6), which re-widens the raw-E range the
actor must digest — re-importing the conditioning problem it was meant to
avoid. The boundary fix also does NOT rescue Reacher's γ=0.99 gap (disc
85.8 vs legacy 85.3 at τ0.1), which pins Reacher's γ-sensitivity on the
anchoring/contraction mechanism, not the cliff. **Keep `boundary=legacy`;
the γ-correlated-confound worry about the fleet table is discharged.**

### md20 wave — the discount-as-compressor hypothesis, refuted; md20 still adopted

γ=0.98 / raw E / md=20 / a2.5 (jobs 8565–8575), vs the md12 fleet:

| cell | md12 (fleet) | md20 | md20+smooth | md20+vlog |
|---|---|---|---|---|
| TwoRoom h100 | 88.0 ± 9.2 | 87.1 ± 6.7 | 85.6 ± 6.0 | 75.6 ± 9.7 |
| Cube h100 | 81.1 ± 0.8 | 81.1 ± 1.4 | — | 81.6 ± 1.0 |
| Reacher τ0.05 / τ0.1 | 55.1 ± 6.5 / 94.0 ± 0.7 | **55.1 ± 1.7 / 95.8 ± 2.1** | — | **58.9 ± 1.0** / 95.6 |

The TwoRoom push-up hypothesis is **refuted**: the wide band does not
unlock at γ=0.98 with raw E (87.1 ≈ 88.0) — the banked band×conditioning
synergy (+8.7) was specific to vlog at γ=1.0/a1.8 and does not transfer.
But md20 is **free on Cube and positive on Reacher** (τ0.05 seed sd 6.5 →
1.7; τ0.1 +1.8), so the unified config adopts **md=20** as the
horizon-covering constant. (Reacher vlog at 0.98/md20 posts the best τ0.05
measured, 58.9 ± 1.0 — but TwoRoom vetoes vlog at γ=0.98 twice over, so
`none` stays the unified choice.)

### Deploy-amax pass — a clean null: the deploy clip is a free knob

Frozen γ=0.98 fleet actors, top-level `amax` rewritten (jobs 8572–8583):

| cell | deploy 1.6/1.8/2.2 (recipe) | deploy 2.5 (native) | deploy 3.0 |
|---|---|---|---|
| TwoRoom h100 | 88.7 ± 12.1 (@1.8) | 88.0 ± 9.2 | 87.6 ± 8.3 |
| Cube h100 | 80.0 ± 1.2 (@1.6) | 81.1 ± 0.8 | 81.1 ± 1.5 |
| Reacher τ0.05 / τ0.1 | 54.4 / 94.4 (@2.2) | 55.1 / 94.0 | 54.9 / 94.0 |

At train-amax 2.5, deployment clip anywhere in [1.6, 3.0] moves nothing
beyond noise. The per-seed decomposition is the telling part: TwoRoom's
weak actor (seed 2: 74.7 / 77.3 / 78.0 across clips) and strong actors
(seeds 0/1: 95.3–96.0 at deploy 1.8) keep their identity at every clip —
**the h100 spread is actor-training quality, not a deployment-projection
artifact**. Note the strong seeds at 95–96 already touch the banked-best
level; the frontier is seed-to-seed training stability, not any of γ /
band / clip / boundary.

## Conclusion — the unified configuration and what it costs

**Supported single config: γ=0.98, n=50, vnorm=none, boundary=legacy,
amax=2.5 (train; deploy free in [1.6, 3.0]), max_delta=20.** Scores (this
campaign's protocol, seeds 0/1/2 × draws 42/43/44):

| cell | unified | bespoke banked | Δ |
|---|---|---|---|
| TwoRoom h25 / h100 | 100.0 / ~88 | 100.0 / 97.3† | 0 / −9† |
| Cube h25 / h100 | 87.3 / 81.1 | 89.1 / 82.4 | −1.8 / −1.3 |
| Reacher τ0.1 / τ0.05 (**latched**, paper convention) | **99.6 / 93.3** | 98.7 / 88.7 (paper) | **+0.9 / +4.6** |
| Reacher τ0.1 / τ0.05 (held, in-campaign) | 95.8 / 55.1 | — | — |

The latched pass (job 8590, frozen unified md20 actors re-evaluated under
`RS_LATCHED=1`) shows the earlier held-at-end row was a convention
artifact: under the paper's first-hit metric the unified config **beats the
paper's tuned Reacher recipe, +4.6 at the tight tolerance** (per-seed
τ0.05: 94.7/94.0/91.3 — the md20 band's precision gain carries through).
Protocol caveat: 3 train seeds × draws 42–44 vs the paper's 6 × 42–47 —
extend seeds before quoting in the paper.

† the banked 97.3 is a seed-3 diagnostic (γ=1.0+vlog+md20+a1.8,
horizon-matched); the protocol-seed anchor for that recipe was never
banked. Unified seeds 0/1 reach 95–96 at deploy 1.8 — the gap is seed
variance, not configuration.

### PLDM columns (completion wave) — no collapse anywhere, and a base-keyed vlog pattern

Unified (γ=0.98/md20/a2.5/raw-E) vs the paper's per-cell PLDM rows:

| cell | unified ctrl | unified vlog | paper |
|---|---|---|---|
| TwoRoom PLDM h25 / h100 | **99.1** / 86.2 ± 8.7 | 96.7 / 88.9 ± **0.8** | 98.2 / 96.0 |
| Cube PLDM h25 / h100 (easy) | 79.8 / **80.7** | **83.1 / 83.6** | 82.9 / 77.1 |
| Reacher PLDM τ0.1 / τ0.05 (latched) | 97.6 / 80.7 | — | 97.8 / 82.0 |

- The predicted-riskiest cell (cube PLDM, bespoke clip 4.5 → 2.5) did NOT
  collapse: −3.1 at h25, **+3.6 at h100** (hard: −7 / **+6.6**).
- TwoRoom PLDM repeats the LeWM signature exactly: h25 above paper, h100
  −10 on ctrl with the weak-seed spread (77.3/86.7/94.7 — strong seed a
  point under paper), vlog tightening sd to 0.8.
- Reacher PLDM is par (−0.2/−1.3), trained at the paper anchor m0.5/lr3e-4.
- **vlog at γ=0.98 helps BOTH PLDM cells and hurts BOTH LeWM-family cells**
  — E-conditioning need is a property of the world-model base, not the
  environment. A single base-keyed rule ("vnorm=log iff PLDM") would beat or
  match the paper in nearly every column, at the cost of one principled
  fork in the otherwise-unified config; under the strict single config
  (vnorm=none) the PLDM h100 cells carry the same selection story as LeWM.

### Beat portfolio (jobs 8596–8607) — scale falsified, replay free, selection survives

Three candidate stabilizers at the unified base (γ=0.98/md20/a2.5):

| arm | TwoRoom h100 | note |
|---|---|---|
| ctrl (reference, md12) | 88.0 ± 9.2 | |
| S: vnorm=scale (E·(1−γ)) | **74.2 ± 3.4** | stable but stably WORSE |
| R: replay 0.5 | 85.1 ± 9.8 | ≈ free; same weak-seed pattern |
| SR: scale+replay+og | 70.4 ± 4.7 | dominated by scale's damage |

**The linear interface normalization is falsified alongside log**: at γ=0.98
the refiner needs the raw E magnitude — every compression of the
already-discount-compressed channel (log −13, scale −14) trades the same
~14 points of mean for its variance cut. The conditioning story therefore
explains the *variance*, but no input transform converts it into mean at
γ<1; the mean-preserving stabilizer is selection, not normalization.
On Reacher, scale is par (latched 99.8 / 92.2 vs raw-E 99.6 / 93.3) — no
cross-env harm. Replay 0.5 can be adopted for uniformity at ~no cost
(−2.9, within noise) but does not stabilize.

### Seed selection on val draws (user-approved protocol extension)

Train seeds unchanged (0/1/2); each cell's seed is picked on draws **50/51**
(the repo's sanctioned selection draws, never reported) and the pick's
already-banked 42–44 number is quoted. Val pass (jobs 8608–8613, eval-only):
the weak actor ranks LAST on val in both families (md12: val 95/96/86 vs
report 93.3/93.3/77.3; md20: 95/79/90 vs 93.3/80.0/88.0), so selection
avoids it without touching report draws. **Selected-of-3 TwoRoom h100 =
93.3 vs paper 94.2 — the unified deficit shrinks from −6/−7 to −0.9.**
Fairness note: the banked shipping protocol itself selected on draw 42 and
reported 43/44, so the paper number is also a selected quantity; this
protocol (selection on held-back draws, report untouched) is the stricter
of the two. Reported as its own row — the estimand is selected-of-3, not
mean-of-3.

**Extending selection to the deploy clip closes it (jobs 8631–8633).** The
deploy clip is a deployment choice like the seed; over 6 val candidates
(3 seeds × deploy ∈ {2.5, 1.8}, all scored only on 50/51) the winner is
s0@1.8 (val 97.0), whose untouched report number is **95.3 — the strict
unified config beats the paper's 94.2 on TwoRoom h100**. Per-cell selection
keeps h25 at 100.0. Also: `vnorm=scale` on Cube came in mildly positive
(88.9 ± 1.4 / 82.0 ± 0.7 vs raw-E 87.3 / 81.1) — scale's −14 is
TwoRoom-specific, matching the base/env-dependent conditioning pattern.

### Granular deploy sweep, deploy-K probe, tanh squash (2026-08-19/20 wave)

- **Deploy-K is a clean negative**: rewriting checkpoint `iters` to 12/16
  hurts on val (LeWM h100 92.3 → 87.7/85.3; PLDM likewise). The K=8-trained
  refiner does not extrapolate to deeper application — test-time compute
  does not scale this planner.
- **Fine deploy-clip grid (val h100 means, LeWM)**: 1.4 → **94.7** > 1.8 ≈
  2.0–2.5 ≈ 92 > 2.8 → 90.3, and the weak seed recovers at 1.4 (val 92).
  PLDM flat (85–87.7). Report evals at 1.4/2.0 in flight to check whether
  the tight-clip gain is a mean effect.
- **Clip selection at 2 val draws is noisy**: PLDM's val-argmax (s2@1.8, val
  92 vs 91 native) reported 91.3 — WORSE than native s2's 94.7. **Seed-only
  selection is the robust rule** (LeWM 93.3, PLDM 94.7, both ≈ −1 vs paper);
  the LeWM seed+clip beat (95.3) stands but clip-selection needs more val
  draws to be a protocol.
- **tanh squash** (gradient flows through the box): **harmful on LeWM**
  (h100 68.7/73.3 — the raw-interface lesson a third time) but on PLDM
  `tanh×vlog` posts 93.3/**98.7**/61.3 — a seed above the paper's 96.0.
  PLDM-squash val pass in flight for honest selection. Gradient flow
  matters exactly where saturation pressure is highest (PLDM's demo actions
  reach 4.5σ against the 2.5 box).

Mechanism summary: (1) γ is a **horizon budget** in both directions —
TwoRoom/Cube are γ-indifferent once the clip stops interacting (the 08-17
falsification was γ×clip), while Reacher genuinely wants γ=0.98 because its
ceiling 1/(1−γ)=50 anchors the never-exactly-labeled cross-episode pairs at
the env diameter; (2) the TD-seam non-monotonicity is real in target space
and cosmetic in behaviour; (3) band and deploy-clip are free knobs at the
wide training clip; (4) the remaining unified-vs-tuned gap on TwoRoom h100
is per-seed actor-training variance — the open frontier is stabilizing the
raw-E actor's out-of-band extrapolation (vlog does it at γ=1, nothing
measured does it at γ=0.98 without costing mean).
