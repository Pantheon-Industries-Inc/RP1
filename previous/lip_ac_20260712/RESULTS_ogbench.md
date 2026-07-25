# LIP-AC on OGBench lewm-cube — Results (2026-07-12)

**TL;DR:** One tandem actor-critic run (warm-started critic, no per-horizon sweep) **matches
both horizon-specific sequential champions**: h25 88/92/76 (mean 85.3 vs champion 87.3),
h50 74/86/62 (mean **74.0 — exact tie** with the dedicated-h50-swept champion, +4.0 over the
h50 *confirmed* champion lipH50). The headline mechanism: tandem training **degraded every
teacher's CEM score** (82 → 64–78 at h25) while producing students at/above champion level —
direct evidence that value-ranking quality (what CEM needs) and value-gradient quality (what
LIP consumes) are different, tradeable properties. This is the complement of the PushT
reversal, shown from the other side.

## Setup

Fresh pod (4×H100), full bootstrap; harness validated by anchors reproducing the sequential
winners **exactly**: LIP v1 88.0 (writeup 88), TD+CEM 82.0 (writeup 82) on h25/s42.
Protocol identical to the full-protocol writeup (50 tasks/draw, draws = eval seeds 42/43/44,
success = block within 4cm, bf16, 224px). Trainer: `scripts/plan/train_lip_ac.py`
(critic = n-step expectile TD on the stride-1 cache, teacher = EMA(critic) serving bootstrap
targets and the actor, actor = train_lip.py objective; all arms: critic τ0.03/n50/1024-batch,
actor K8/lr3e-4/8k steps, EMA τ.005, 1:1 interleave). Caches rebuilt from stored h5 pixels
(GPU-batched, eval-matched transform; fs1 2.01M×192, fs5 410k; ~3.5 min on 4 GPUs).

Gotchas fixed en route: h5 pixels need `hdf5plugin` imported; the cube h5 pads every
episode-terminal action with NaN (10,000 rows, verified all-terminal) → `nanmean/nanstd`
for action z-scoring. tar/rsync to the network volume fail on chown (content unaffected).

## Arms (selection on s42, h25+h50)

| arm | critic init | teacher | h25 | h50 | sum | teacher CEM h25 / h50 |
|---|---|---|---|---|---|---|
| **warm** | sequential TD winner | freeze @80% | **88** | **74** | **162** | 68 / 62 |
| nofreeze | sequential TD winner | never frozen | 88 | 74 | 162 | 74 / 54 |
| expand03 | sequential TD winner | freeze @80%, value-expansion w=0.3 | 86 | 68 | 154 | **78** / 60 |
| fresh | scratch (2k pretrain) | freeze @80% | 84 | 68 | 152 | 64 / 60 |
| *(sequential references)* | | | *88* | *72 swept / 70 confirmed* | | *82 / 64* |

Winner: **warm** (tie with nofreeze, broken toward the simpler arm).

## Winner confirmed on all 6 cells

| cell | AC-warm | sequential champion | Δ |
|---|---|---|---|
| h25 s42 | 88 | 88 (LIP v1) | 0 |
| h25 s43 | 92 | 96 | −4 |
| h25 s44 | 76 | 78 | −2 |
| **h25 mean** | **85.3** | **87.3** | **−2.0** |
| h50 s42 | 74 | 72 (h50-swept lr2e4) | +2 |
| h50 s43 | 86 | 86 | 0 |
| h50 s44 | 62 | 64 | −2 |
| **h50 mean** | **74.0** | **74.0** | **0.0** |

(vs the h50 *confirmed* champion lipH50 70/78/62: +4/+8/0, mean **+4.0**. Reference points:
TD+CEM h25 79.3 mean; h50-native TD(τ0.1)+CEM 74.0 mean; latent+CEM 75.3 / 54.0;
floors 50.7 / 30.7. Cell resolution: n=50 → 1 episode = 2 pts.)

## Findings

1. **Tandem ≈ sequential on cube, at a fraction of the tuning.** The sequential pipeline
   behind the reference numbers: 25-config TD sweep + CEM selection + per-horizon LIP sweeps.
   AC-warm: one run, h25 recipe, no h50-specific tuning — and it ties the h50-swept champion's
   mean exactly. Parity, not a win, at h25 (−2.0 mean ≈ 1 episode/draw).
2. **The dissociation (main scientific result).** Every tandem teacher lost CEM quality at
   h25 (82 → 68 warm / 74 nofreeze / 78 expand / 64 fresh) while its student stayed at
   champion level. The value's ranking quality and its gradient-field teachability are
   different axes; co-training trades the former for the latter, and the student doesn't pay.
   Mirror-image confirmation of the PushT rugged-landscape story.
3. **Warm start matters.** The fresh-critic arm trails everywhere (84/68): 2k pretrain + 6.4k
   live TD steps don't recover what the sequential 6k-step teacher encodes.
4. **The freeze tail was unnecessary here** — warm ≡ nofreeze on both selection cells (and
   bitwise-identical training until step 6400, a nice implementation check). Cube's landscape
   is benign; keep the freeze flag for lr-sensitive settings.
5. **Value expansion (w=0.3) costs the student a little** (−2/−6 vs warm) **but preserves the
   teacher's CEM quality best** (78 at h25) — consistent with expansion anchoring the value on
   planner-visited states. Not worth it on cube; retest where WM error is the binding issue.
6. Training dynamics: all arms E_final 20.7 → 1.5–2.5, no divergence at lr 3e-4/K8.
   (`td_loss nan` in logs after step 6400 = critic frozen, not a numerical issue.)

## Interpretation for the tandem question

On cube the sequential TD→CEM-select→LIP pipeline was NOT teacher-bottlenecked — tandem
matches but doesn't beat it. The value of AC here is workflow (one run replaces two sweeps
+ selection) and the dissociation evidence. The place tandem should *win outright* is where
CEM-selection actively picks a bad-gradient teacher — i.e. PushT (`run_ac_pusht.sh`, ready).

## Artifacts

- Pod: 31.24.80.32:15419 — `/workspace/results/summary.csv`, actors+values in
  `/workspace/{actors,metrics}` (pod now idle).
- Local: `results_ogbench/` (summary, driver/pipeline/training logs, all 4 actor + 4 teacher
  checkpoints). Trainer: `stable-worldmodel/scripts/plan/train_lip_ac.py`. Drivers:
  `run_ac_ogbench.sh`, `pipeline_ogbench.sh`.

---

# Addendum: the >90 push (h25), round 1 — 2026-07-12 evening

Goal: h25 mean > 90 (champion 87.3, AC-warm 85.3). Diagnostics first: K/lr grid is
saturated (k12 ≤ 88 on s42); the ±2.5σ plan clamp truncates real expert actions
(4–9%/dim beyond 2.5σ, dim0 tail to 3.5σ); eval-time restarts untested.

**Round A — eval-time restarts (no retraining): negative, informative.**
restarts=8 (±robust_m=4) HURT both actors (warm s42+s44: 164→160/158; sequential:
166→160/158). Two mechanisms: argmin-V selection exploits value-ranking errors
(CEM's over-optimization pathology in miniature), and the actor is trained
exclusively from A0=0 — noisy inits are off-distribution for the learned update
rule. Effect symmetric across good-ranking (82) and drifted (68) teachers → it is
an actor-side property. Deploy-time compute does not stack onto LIP this way.

**Round B — during-run schedules + statics (all warm-started):**

| arm | s42 | s44 | sum | note |
|---|---|---|---|---|
| warm (round-1 baseline) | 88 | 76 | 164 | |
| sched (clr 1e-3→1e-4, τ 0.1→0.03, alr→3e-5) | 86 | 72 | 158 | alone: worse |
| amax35 (clamp 3.5σ) | 78 | 74 | 152 | alone: much worse |
| md5 (goal window = 25 primitives) | 86 | 76 | 162 | ~neutral |
| **schedamax (sched + amax 3.5)** | **88** | **80** | **168** | **composition wins** |

The interaction is the finding: each component alone regresses, together they give
the best selection sum recorded. Reading: the larger action range gives the actor
room to overdrive against a static teacher early in training; the annealed teacher
(smooth τ=0.1 early, converging lr) tames exactly that, then sharpens to τ=0.03.

**Confirm (3 draws): schedamax = 88 / 96 / 80, mean 88.0** — new best mean on the
project; ≥ the sequential champion on every draw (88=88, 96=96, 80>78; s44=80 is
the best number ever recorded on the hard draw); h50 s42 held at 74. Determinism:
selection cells reproduced exactly in confirm re-evals.

Honest read: +0.7 mean over champion is within n=50 noise; the defensible claims
are per-draw weak dominance and the s44 record. Target >90 needs +2 mean more.

**Round 2 (running):** seed replicas (seed1/seed2 — TwoRoom showed 2–4pt training
-seed spread), τ 0.2→0.03 (stronger early smoothing), slow12k (12k steps, schedules
stretched). Selection s42+s44 vs incumbent 168; confirm only if an arm beats it.
