# FAILURE AUTOPSY — why the LIP-AC line plateaus at 88 on cube (h25)

**Date:** 2026-07-14 · **Model under study:** min0 (v1-MLP LIP-AC, input `[A, ∇V, E]`, 3-seed 88.0)
· **Protocol:** h25, 50 tasks/draw, draws s42/s43/s44, success = block ≤ 4cm of target.
**Evidence base:** per-episode success arrays from **218 archived evals** (all S3–S7 arms, champion,
min0 seeds, AC-CEM planners), offline WM/value probes, and **video-recorded reruns** of min0_s0 on
all 3 draws (fail sets reproduced the original evals exactly).
**Gallery:** [failure_gallery/](failure_gallery/) — one mp4 per failure (3 panels: **agent | dataset
expert | goal**) + keyframe PNGs + [attribution.csv](failure_gallery/attribution.csv).

---

## 0. TL;DR

min0 misses ~12/100 episodes. **9.3 of those 12 are one deterministic failure mode:** grasp
acquisition with an airborne goal — the gripper approaches, scrapes or hovers, closes near-but-not-on
the cube, and the block never leaves the table. The same 14 tasks fail for **every** training seed,
every hyperparameter recipe (S7 swept 27 arms), and **both planner families** (gradient LIP and
sampling CEM fail the identical task sets). The remaining ~2.7/100 are **carry/place precision
misses** drawn stochastically from a small pool of transport/rotation tasks — this is the seed noise.

Mechanism (probe-verified): the WM's **predictor is optimistically wrong just off the expert action
manifold at the attachment boundary**. Fed expert actions it simulates grasp+lift almost perfectly;
fed optimizer-shaped plans it imagines the block attaching when it physically would not. Every
planner therefore converges to counterfeit plans that score *better than the expert demonstration*
(imagined steps-to-go 0.62 vs 1.36) and fail 100% in reality. Root cause: the training data is
expert-only — 45 genuine drops in 2M frames, zero attempted-and-missed grasps.

Fix in progress: targeted negative data (§8). Collected 2026-07-14: **1,050 verified grasp-and-miss
clips + 1,050 verified drop clips + 201 boundary positives** (~150k frames, physics-verified labels).

---

## 1. Failures are structural, not noise

Across all healthy recipes (success ≥ 80%) in the 218-eval archive:

| draw | evals | near-always-fail tasks (≥75% of recipes) | never-fail tasks | failure mass in core |
|---|---|---|---|---|
| s42 | 64 healthy | **11, 14, 17, 27, 38** (+6 at 62%) | 27/50 | 68% |
| s43 | 37 healthy | **8, 25** | 37/50 | 68% |
| s44 | 31 healthy | **12, 13, 23, 24, 31** (+29, 46 at 84%) | 26/50 | 73% |

The "core" = 14 tasks. The AC-CEM planners (different search paradigm, same WM+value) fail the
**identical** 5/5 core tasks on s42 → planner-independent. Physically, the core is one family
(features from the replicated eval sampler, verified exact — 1,760,000 valid rows):

| bucket | n | Δxy (m) | Δz goal (m) | lift in window | eff-block dist | start contact |
|---|---|---|---|---|---|---|
| core (≥75% fail) | 14 | **0.023** | **+0.136** | = Δz | **0.123** | 0.47 |
| intermittent | 4 | 0.108 | +0.080 | 0.164 | 0.019 | 0.79 |
| never-fail | 90 | 0.013 | −0.038 | 0.029 | 0.072 | 0.32 |

**Core = block ~static in xy, goal 7–25cm airborne, arm away, contact still to be acquired.**
Already-grasped transports: intermittent. On-table pushes: never fail.

## 2. The two failure modes (per 100 episodes, min0)

| share | mode | what happens |
|---|---|---|
| ~9.3 | **A — grasp-acquisition failure** | never attaches; block stays on table; deterministic on the 14 core tasks |
| ~2.7 | **B — carry/place precision miss** | attaches fine, ends off-target (sideways, overshoot, or just outside the 4cm threshold); seed-dependent pool |

Arithmetic: 9.3 + ~2.7 ≈ 12 ⇒ the 88.0. Mode A bounds the achievable card at 90/96/86 = **90.7**,
and S7's best seed (mw01te001_s0) produced exactly that card — everything actor-solvable is solved.

## 3. Gallery — every failure of min0_s0, attributed

Videos: 3 panels = agent | dataset expert | goal. Verdicts from panel-cropped red-cube tracking
(≈0.35 cm/px) + dataset goal geometry; keyframes `_kf{0,1,2}.png` alongside each clip.

| clip | mode | verdict | goal height | end vs goal |
|---|---|---|---|---|
| [s42_task11](failure_gallery/s42_task11.mp4) | A | never lifted, scrape | 10.8cm | on table, 28px below |
| [s42_task14](failure_gallery/s42_task14.mp4) | A | never lifted, pushed aside | 16.6cm | 53px below, 13px off |
| [s42_task17](failure_gallery/s42_task17.mp4) | A | never lifted (lift+reorient task) | 16.5cm | 39px below |
| [s42_task27](failure_gallery/s42_task27.mp4) | A | never lifted, nudged | 11.3cm | 33px below |
| [s42_task38](failure_gallery/s42_task38.mp4) | A | never lifted | 11.5cm | 28px below |
| [s42_task46](failure_gallery/s42_task46.mp4) | **B** | **carried off-target** (lifted ~10cm) | 24.6cm | 65px lateral miss |
| [s43_task08](failure_gallery/s43_task08.mp4) | A | never lifted | 9.4cm | 23px below |
| [s43_task14](failure_gallery/s43_task14.mp4) | **B** | set-down/push near-miss (~6cm > 4cm thr.) | ~table | 18px miss |
| [s43_task25](failure_gallery/s43_task25.mp4) | A | never lifted | 18.7cm | 52px below |
| [s44_task12](failure_gallery/s44_task12.mp4) | A | never lifted | 8.7cm | 21px below |
| [s44_task13](failure_gallery/s44_task13.mp4) | A | never lifted | 8.8cm | 22px below |
| [s44_task23](failure_gallery/s44_task23.mp4) | A | never lifted | 14.3cm | 38px below |
| [s44_task24](failure_gallery/s44_task24.mp4) | A | never lifted | 6.9cm | 20px below |
| [s44_task28](failure_gallery/s44_task28.mp4) | **B** | carried high, overshoot past goal | ~20cm | (−15, −30)px |
| [s44_task29](failure_gallery/s44_task29.mp4) | A | never lifted | 10.0cm | 25px below |
| [s44_task31](failure_gallery/s44_task31.mp4) | A | never lifted | 21.7cm | 53px below |
| [s44_task46](failure_gallery/s44_task46.mp4) | A | never lifted | 24.6cm | 64px below |

Count: **15× mode A, 2–3× mode B** (s43_task14 is a near-table precision case; graded B).
Note s44_task29 and s44_task46 are core tasks; s42_task46 / s44_task28 are pool tasks — matching
the cross-recipe statistics. An early tracker version mis-read 9 clips as "lifted-but-wrong-place";
that was contamination from the 3-panel layout + purple gripper. The corrected tracker (panel-cropped,
red-only) plus manual keyframe review produced the verdicts above.

## 4. Mode A mechanism — each suspect tested

1. **Harness** ✗ — champion control reproduces 96.0 (twice, on both pods).
2. **Planner/search** ✗ — LIP (1st-order) and CEM (0th-order) fail identical task sets.
3. **Actor hypers** ✗ — S7: 27 arms over mean-weight × expectile grid + amax/K/steps/n-step; every
   healthy recipe fails the same core; 3-seed confirms regress to 88.0.
4. **WM representation/dynamics on-manifold** ✗ — expert actions rolled through the frozen WM cover
   93% of the start→goal latent distance on core tasks (better than on easy tasks); replay fidelity
   of the physics stack is 0.3mm.
5. **Value** ✗ — reads true latents correctly (E(start) ≈ 23.6 vs true 25; E(goal,goal)=0; scores
   the expert's imagined terminal at 1.36).
6. **→ Predictor off-manifold** ✓ — the actor's own K=8 plans score **E = 0.62, imagined-better than
   the expert**, with grasp-like gripper statistics (mean grip 0.26 vs expert 0.24), and fail 100%
   really. The false basin is **broad**: under action-noise re-scoring (σ=0.1, m=16) counterfeit
   plans still out-score the expert (0.97 vs 1.36) ⇒ eval-time robustification (restarts, buffer
   select, solver `robust_m`) cannot fix it. Value-level fixes are also blocked in principle: V(z)
   scores the predictor's *output*, and the counterfeit enters upstream of V's input.

Root cause: **expert-only data.** Census of the 2M-frame corpus: 22,602 contact acquisitions, 89%
lift; 45 genuine mid-air drops (0.2%); zero attempted-and-missed grasps of the kind an optimizer
produces. Attachment is binary in reality, smooth in the model, and unconstrained by data exactly
where optimization pressure pushes.

## 5. Mode B — the precision pool

Tasks with 25–65% cross-recipe fail rates: s42_6 (lift-carry combo), s42_26, s44_14 (set-down +
rotation), s44_28 (rotation transport) + a rare tail. Grasping succeeds; the miss is in carry/
placement accuracy relative to the 4cm threshold. Which pool tasks fail varies by training seed —
this is the ±2–4 pt per-draw seed noise around the core-imposed ceiling.

## 6. What is settled (do not re-spend compute here)

- Hyper search on this WM is **exhausted** (S7: grid + scouts + 3-seed confirms; τ=0.5 controls
  collapse in triplicate 58.7/62.7/63.3 → the low-τ expectile asymmetry is load-bearing; 6k steps
  ties 8k at −25% cost; amax flat 2.5–4.5; K=8 right).
- Eval-time robustness: dead (broad false basin, §4).
- Value-only retraining on failure data: dead in principle (§4.6) — unless done as *hardening on
  imagined latents labeled by real outcomes*, which is untested (separability check pending).

## 7. The fix — dataset augmentation for the WM re-run

Collected 2026-07-14 (pod a, `/workspace/datasets/lewm_cube_negatives/`, 24 shards):

| class | count | recipe |
|---|---|---|
| grasp-and-miss | **1,050** | reset 10–18 steps before real grasp events; replay expert with xy-offset ring 0.15–0.45 (519), offset+timing combo (284), approach-too-high (238), close-timing jitter alone (9 — timing alone almost never breaks a grasp) |
| drops | **1,050** | replay real grasp to airborne (release height 4.5–15cm, half with arm still moving), force gripper open (dim 4 = −1); verified rise-then-fall |
| boundary positives | 201 | perturbed-but-still-successful grasps (contrastive) |

All labels physics-verified (fail: block never ≥3cm; drop: ≥4.5cm then <3cm). 65-step clips,
224×224 pixels + executed actions + qpos/qvel + block pos + contact + provenance (`src_ep`,
`src_t0`, `label`). The 149 episodes used by eval draws s42/s43/s44 are excluded from mining.
Replay fidelity of the collector: 0.3mm over 45 steps.

**Recommended before committing the re-run** (dose at 2×1k ≈ 6% of corpus — thin for from-scratch):
1. Scale to **4–5k/class** (collector is turnkey, ~35 min per 2k clips).
2. **On-policy batch (~1k):** record the current planner's real failed attempts — covers
   optimizer-characteristic geometry that expert-perturbations cannot.
3. **Transport/set-down misses (~500):** grasp faithfully, then bias the carry/set-down 2–20cm —
   targets mode B at the 4cm threshold.
4. Re-run trains encoder too (frees representation for falling/held-wrong states); afterwards
   rebuild caches → TD init → actors (all cheap: ~1h + minutes + 1.3h/actor at 6k steps).

## 8. Artifacts & repro

- Videos/keyframes/attribution: [failure_gallery/](failure_gallery/) (local copy; source of truth
  regenerable on pod a).
- Pod a (31.24.80.32:15419): `/workspace/s7_collect/` — `collect_negatives.py` (collector),
  `harvest_gallery.py` + `attribute_v2.py` (gallery/attribution), `smoke_collect.py`;
  eval outputs `gallery_s4{2,3,4}.txt` in `/workspace/ckpts/`; negatives in
  `/workspace/datasets/lewm_cube_negatives/`.
- Probes (offline, cache-only): expert-through-WM, actor-refinement replication, noise-fragility —
  commands in the session log; core task list: s42 {11,14,17,27,38}, s43 {8,25},
  s44 {12,13,23,24,29,31,46}.
- Related: [HANDOFF_min0.md](HANDOFF_min0.md) (actor line), [EXPERIMENTS.md](EXPERIMENTS.md)
  (campaign log incl. S7), memory `lip-ac-tandem`.
