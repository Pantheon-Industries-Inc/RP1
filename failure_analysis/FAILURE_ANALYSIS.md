# LIP failure-case analysis — video + frame study

**Setup:** actor `lip4_r2w_s1` (a LIP-v4 tandem actor) on the Dyna round-2 world model `wm2_e2`, OGBench cube-single, expert-lance goals. Two regimes captured:
- **h25** (standard benchmark horizon): draw s44, 92% success → the residual failures.
- **h200** (whole-demo execution): draw s42, 0% success → total long-horizon collapse.

**Capture note:** the eval harness's built-in video panel renders the agent view as noise (its `infos['pixels']` buffer is not a clean render under `ob_type: states`). I instead captured clean 224px agent frames via the `SWM_RECORD_PATH` collector hook, which is verified-clean. Because `terminate_at_goal` ends successful episodes early and the collector drops <25-step clips, the surviving h25 records are exactly the long/failing episodes.

Videos in `failure_analysis/videos/`; sampled frames in `failure_analysis/h25/` and `failure_analysis/h200/`.

---

## h25 residual failures — grasp-contact failure (`h25_grasp_fail_1/2/3.mp4`)

All three residual failures show the **same** mode: the arm navigates to the vicinity of the red block but **never achieves a successful grasp-and-lift**, then the horizon expires with the block essentially where it started.

- **fail_1:** arm approaches the block (t≈29 gripper right beside it) then **retracts up-and-away** (t49) — block untouched.
- **fail_2:** arm ends **hovering directly next to the block** (t49), never closing/lifting.
- **fail_3:** arm ends **above the block**, no successful grasp.

This is precisely the campaign's known residual: the failure is the **contact moment** (closing on the block and lifting), not navigation. It matches the "attachment optimism" story — the WM imagines a successful grasp, the actor commits, and the real gripper fails to secure the block. These are the disjoint, per-actor ~5% failures that keep the single-actor number at ~95.

## h200 long-horizon failures — task-sequencing failure (`h200_longhorizon_fail_0/1/2.mp4`)

**Not** dramatic off-manifold flailing (my prior hypothesis) — the arm stays in plausible poses over the table the whole time. The real failure is that it **never sequences the coordinated grasp→transport→place** required to reach a goal 200 steps away:

- **fail_1 (clearest):** the red block sits at lower-left and **never moves** across all 400 steps (t0 = t228 = t399); the arm mills around the *center* of the workspace and **never even goes to the block** to attempt a grasp.
- **fail_0:** the arm does reach down toward the block and disturbs it, but never completes a place; by the end the block is displaced, not at the goal.

**Interpretation:** for a 200-step goal the planner gets no effective long-horizon direction from the start state — it isn't steered to "go grasp the block first," so it makes locally-plausible but globally-aimless motions. This is a *different* bottleneck than h25 (there the arm reaches the block but fumbles the grasp; here it often doesn't engage the task at all). It is consistent with, and corroborated by:
- **CEM-h200 = 0** (a value-free planner also fails → not the learned value alone).
- The **empirical value test**: the "better-calibrated" (expectile-0.5) value made *every* horizon worse and did **not** move h200 off 0.
- **h150→h200 fade** (49→26→1.3): a continuous capacity fade, not a discrete artifact.

**Conclusion:** h200 is a long-horizon *task-sequencing / planning-capacity* limit, orthogonal to the grasping fix (which Dyna delivered at h25–h100). The right lever is **subgoal decomposition** — break the 200-step goal into ~25–50-step on-manifold waypoints (HWM reach-select), so each leg is within the planner's effective range — **not** value-hyperparameter tuning.

---

### File index
| file | regime | shows |
|---|---|---|
| `videos/h25_grasp_fail_1.mp4` | h25 | approach then retract, no grasp |
| `videos/h25_grasp_fail_2.mp4` | h25 | stalls hovering beside block |
| `videos/h25_grasp_fail_3.mp4` | h25 | stalls above block |
| `videos/h200_longhorizon_fail_1.mp4` | h200 | block never moves; arm mills in center |
| `videos/h200_longhorizon_fail_0.mp4` | h200 | reaches block, disturbs it, no place |
| `videos/h200_longhorizon_fail_2.mp4` | h200 | long-horizon aimless motion |
| `h25/`, `h200/` | — | sampled PNG frames used above |
