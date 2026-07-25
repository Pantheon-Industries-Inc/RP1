# min0 — complete failure autopsy with renders (2026-07-14)

**Model:** min0 gated, training seed 0 (`/workspace/actors/lip_ac90s5_min0_s0.pt`, the 87.8 4-seed line).
**Protocol:** standard h25 eval (goal_offset 25, budget 50, 50 tasks/draw), draws s42/s43/s44, videos from the
harness's native recorder (`env_{task}.mp4`, 3-panel: **agent | dataset-expert | goal**).
**Verification:** rerun fail sets match the original S5/confirm evals exactly (s42 44/50, s43 47/50, s44 42/50),
so these are the genuine historical failures. Behavior metrics measured by red-cube tracking in the agent panel
vs the goal panel (scale ≈ 3–3.5 mm/px). Offline imagined-E numbers from the frozen-WM refinement probe
(actor's own K=8 plan vs the expert's actions, teacher = min0_s0's own value).

**Renders:** all clips + keyframes in [failure_gallery/](failure_gallery/).

---

## The two failure modes

**Mode A — grasp-acquisition failure (never-attach).** Task family: block ~static in xy, goal 7–25 cm airborne,
arm must approach and acquire the grasp. The WM imagines the *expert's* actions on these windows almost perfectly
(93% of latent distance), but the actor's own plans score **imagined-better than the expert** (mean E 0.62 vs 1.36)
and physically never attach — the predictor is optimistic about attachment just off the expert manifold, and the
optimizer lives exactly there. Closed loop, the gripper descends, scrapes or closes beside the cube for the whole
episode; the block never leaves the table. Planner-independent (CEM fails the identical tasks), hyper-independent
(64 recipes), seed-independent.

**Mode B — carry/place precision miss (attachment succeeds).** The block is grasped and moved but ends outside
the 4 cm success ball: carried off-target, overshot, or set down just wide. Small, partly stochastic across seeds.

**Note on drops:** there are NO drop failures in deployment — the planner never attaches on the core tasks, so it
never gets far enough to drop anything. The *drop clips* in the augmentation set are counterfactual training data,
not observed failures: they teach the WM the retention half of the attachment rule (not-properly-held at height ⇒
block separates and falls) and free-block gravity (~zero examples in the expert corpus), which is exactly the
physics the counterfeit-grasp imagination violates. Grasp-and-miss clips close the acquisition half; drops close
the retention half.

---

## Draw s42 — 44/50, six failures

| task | mode | video | key moment |
|---|---|---|---|
| 11 | A | [s42_task11.mp4](failure_gallery/s42_task11.mp4) | [kf](failure_gallery/s42_task11_kf1.png) |
| 14 | A | [s42_task14.mp4](failure_gallery/s42_task14.mp4) | [kf](failure_gallery/s42_task14_kf1.png) |
| 17 | A | [s42_task17.mp4](failure_gallery/s42_task17.mp4) | [kf](failure_gallery/s42_task17_kf1.png) |
| 27 | A | [s42_task27.mp4](failure_gallery/s42_task27.mp4) | [kf](failure_gallery/s42_task27_kf1.png) |
| 38 | A | [s42_task38.mp4](failure_gallery/s42_task38.mp4) | [kf](failure_gallery/s42_task38_kf1.png) |
| 46 | B | [s42_task46.mp4](failure_gallery/s42_task46.mp4) | [kf](failure_gallery/s42_task46_kf1.png) |

- **task 11** — src ep 2767 @ step 101. Goal: **+10.8 cm**, Δxy 0.7 cm (pure lift-in-place); arm starts 7.5 cm away.
  Cross-recipe fail rate **62/64**. Imagined E: actor plan **0.78** vs expert **1.85** (actor "beats" the expert in
  imagination). Video: gripper reaches the cube, closes beside it repeatedly; block never rises (≤1 px); goal cube
  hovers 28 px above. Textbook counterfeit grasp.
- **task 14** — src ep 4024 @ 15. Goal **+16.6 cm**, Δxy 5.3 cm. Fail rate **64/64** (nobody has ever passed it).
  E 0.38 vs 0.29. Video: approach + close attempts, zero lift, goal 53 px up.
- **task 17** — src ep 4434 @ 15. Goal **+16.5 cm with 0.79 rad yaw change** (lift *and* reorient). **64/64**.
  E 0.50 vs 2.75 — the actor's counterfeit is dramatic here (imagines 0.5 steps-to-go on a task the expert's own
  actions score 2.75). Video: no attach, goal 39 px up.
- **task 27** — src ep 6438 @ 98. Goal **+11.3 cm**, arm 17.5 cm away at start, low starting contact. **63/64**.
  E 0.38 vs 1.26. Video: scrape, no attach.
- **task 38** — src ep 7815 @ 98. Goal **+11.5 cm**, arm 17.6 cm away. **64/64**. E 0.97 vs 0.84. Video: no attach.
- **task 46** — src ep 8931 @ 34. Different animal: **already holding** at start (contact 1.0), transport 18.7 cm
  and place at just **+3 cm**; expert peaks 22 cm high mid-carry. Rare across recipes (<25%) but **fails all 4 min0
  seeds** — a min0-consistent mode-B miss. Video: lifts to ~10 cm, carries, ends **~23 cm sideways** of the goal
  (rise 27→15 px, dx +65 px). The lean input may be costing carry precision here.

## Draw s43 — 47/50, three failures

| task | mode | video | key moment |
|---|---|---|---|
| 8 | A | [s43_task08.mp4](failure_gallery/s43_task08.mp4) | [kf](failure_gallery/s43_task08_kf1.png) |
| 14 | B | [s43_task14.mp4](failure_gallery/s43_task14.mp4) | [kf](failure_gallery/s43_task14_kf1.png) |
| 25 | A | [s43_task25.mp4](failure_gallery/s43_task25.mp4) | [kf](failure_gallery/s43_task25_kf1.png) |

- **task 8** — src ep 2738 @ 12. Goal **+9.4 cm**, Δxy 1.4 cm. **37/37** (universal). E 0.38 vs 1.33.
  Video: no attach, goal 23 px up.
- **task 14** — src ep 3889 @ 129. **Set-down task**: goal **−17.9 cm** (from held-high to table), holding at start
  (contact 0.96). Rare across recipes; only seed 0 fails it on this draw. Video: block ends displaced, **~6 cm from
  the goal** — just outside the 4 cm ball. A near-miss place, the boundary case of mode B.
- **task 25** — src ep 5053 @ 16. Goal **+18.7 cm**. **37/37**. E 0.57 vs 1.98. Video: no attach, goal 52 px up.

## Draw s44 — 42/50, eight failures

| task | mode | video | key moment |
|---|---|---|---|
| 12 | A | [s44_task12.mp4](failure_gallery/s44_task12.mp4) | [kf](failure_gallery/s44_task12_kf1.png) |
| 13 | A | [s44_task13.mp4](failure_gallery/s44_task13.mp4) | [kf](failure_gallery/s44_task13_kf1.png) |
| 23 | A | [s44_task23.mp4](failure_gallery/s44_task23.mp4) | [kf](failure_gallery/s44_task23_kf1.png) |
| 24 | A | [s44_task24.mp4](failure_gallery/s44_task24.mp4) | [kf](failure_gallery/s44_task24_kf1.png) |
| 28 | B | [s44_task28.mp4](failure_gallery/s44_task28.mp4) | [kf](failure_gallery/s44_task28_kf1.png) |
| 29 | A | [s44_task29.mp4](failure_gallery/s44_task29.mp4) | [kf](failure_gallery/s44_task29_kf1.png) |
| 31 | A | [s44_task31.mp4](failure_gallery/s44_task31.mp4) | [kf](failure_gallery/s44_task31_kf1.png) |
| 46 | A | [s44_task46.mp4](failure_gallery/s44_task46.mp4) | [kf](failure_gallery/s44_task46_kf1.png) |

- **task 12** — ep 2581 @ 10. Goal **+8.7 cm**. **31/31**. E 0.38 vs 0.69. No attach (goal 21 px up).
- **task 13** — ep 3379 @ 97. Goal **+8.8 cm**. **31/31**. E 0.74 vs 1.19. No attach.
- **task 23** — ep 5435 @ 102. Goal **+14.3 cm**. **30/31**. E 0.93 vs 0.88. No attach, goal 38 px up.
- **task 24** — ep 5462 @ 9. Goal **+6.9 cm** — the *lowest* core goal, barely above the 4 cm ball, still 31/31.
  E 0.44 vs 0.95. No attach.
- **task 28** — ep 6529 @ 28. Transport-with-rotation: Δxy 15 cm, goal +19.7 cm, Δyaw 0.41, holding at start.
  Intermittent (**20/31**). Video: genuinely grasps, carries high (rise 78 px) but **overshoots past the goal**
  (miss −15, −30 px). Mode B.
- **task 29** — ep 6658 @ 102. Goal **+10.0 cm**. **26/31**. E 0.87 vs 2.90 (huge counterfeit gap). No attach.
- **task 31** — ep 6777 @ 17. Goal **+21.7 cm**. **30/31**. E 0.82 vs 0.54. No attach, goal 53 px up.
- **task 46** — ep 9691 @ 106. Goal **+24.6 cm** — the highest core goal. **26/31**. E 0.48 vs 1.54. No attach,
  goal 64 px up.

---

## Seed variation (same recipe, training seeds 0–3)

| seed | s42 fails | s43 fails | s44 fails |
|---|---|---|---|
| 0 (videos) | 11,14,17,27,38,**46** | 8,**14**,25 | 12,13,23,24,**28**,29,31,46 |
| 1 | 11,14,17,**26**,27,38,**46** | 8,25 | **1**,12,13,**21**,23,24,29,31,**40**,46,**49** |
| 2 | 11,14,17,27,**29**,38,**46** | 8,**11**,25 | 12,13,23,24,**28**,29,31,46 |
| 3 | 11,14,17,27,38,**44**,**46** | 8,**20**,25 | **11**,12,13,23,24,**28**,29,31 |

Bold = outside the universal core. Reading: the **core-14 is identical for every seed** (that's the 9.3/100).
s42_46 fails all four seeds (min0-consistent mode B); s44_28 fails 3 of 4. The remaining bolds are the stochastic
mode-B/rare tail (0–4 tasks per seed) — this churn is exactly the ±2–4 pt per-draw seed noise around the ceiling.

## Accounting per 100 episodes (≈12 misses at 88.0)

- **≈9.3 — mode A**, grasp acquisition, deterministic core-14. Fix: attachment-boundary data
  (grasp-and-miss + drop clips) → WM re-run.
- **≈1.3 — min0-consistent mode B** (s42_46 + usually s44_28): carry/place precision. Fix: perturbed-transport /
  set-down near-miss clips.
- **≈1.5 — stochastic tail** (seed-dependent pool + rare one-offs, incl. near-threshold misses like s43_14).
  Partly recoverable by seed selection; partly same fix as mode B.

*Related: HANDOFF_min0.md (model), EXPERIMENTS.md (campaign), S7 sweep + probes (session 2026-07-14). Collector
for augmentation clips: pod a `/workspace/s7_collect/collect_negatives.py`.*
