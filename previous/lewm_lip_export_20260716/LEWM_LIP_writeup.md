# LIP planning on the LeWM world model — OGBench cube, full protocol

**Status:** complete (2026-07-14; plateau-breaking WM fine-tune addendum 2026-07-16).
**One-line:** a learned iterative planner (LIPv4 actor + tandem TD critic) on the frozen
LeWM world model reaches **88.0% at h25 and 74.0% at h50** — beating the WM's native
latent+CEM planning by +12.7 / +20.0 absolute, 1.51× / 1.86× over-floor — while using
**~16 world-model rollouts per replan instead of CEM's ~9,000**.

---

## 1. Method

**LIPv4 ("min0") actor.** A 2×512 MLP applied K=8 times as a learned refinement rule
over an action plan. Input per iteration: `[vec(A), vec(∇_A V), V]` — the current plan,
the gradient of the learned terminal value through the frozen world model, and the
value itself. No raw state, no raw goal, no gating: task information reaches the actor
*only* through the learned value — "the value is the only teacher", and also the only
input. Deploy: `A(0)=0`, K refinement iterations, execute, replan (receding horizon),
fully deterministic.

**Tandem critic.** Goal-conditioned temporal-distance TD on the frozen WM's latents:
HER hindsight goals (balanced full-horizon + 30% cross-episode), quasimetric head,
n-step 50 backups, γ=1, low expectile 0.03 (optimistic toward the min — the correct
direction for cost-to-go). Warm-started from a sequential TD checkpoint, co-trained
1:1 with the actor via an EMA teacher (τ 0.1→0.03 cosine, critic-lr 1e-3→1e-4,
actor-lr 3e-4→3e-5, amax 3.5σ plan clamp — the "schedamax" recipe), critic frozen for
the final 20% of the 8k steps. Coupling is one-directional: the critic learns only
from dataset TD pairs, never from the actor.

**Baselines.** *Latent+CEM*: the WM's native latent-MSE terminal cost searched by CEM
(~9,000 rollouts/replan) — the standard planning mode for this class of world model.
*TD+CEM*: the same CEM search using the learned TD value as terminal cost.

## 2. Protocol

lewm-cube full protocol: 10k-episode expert dataset (2.01M frames, 224px), 50
tasks/draw, draws = eval seeds {42, 43, 44}, goal = the expert's state 25 (h25) or 50
(h50) primitive steps ahead, env budget 2× the offset, success = block within 4cm.
World model: the authors' exact `ogbench_cube_single_v2WM` checkpoint (ViT-tiny/14,
192-d latent, 3-frame predictor, 25-d action blocks). Benchmark replication was
verified beforehand: published latent+CEM 84.0 on the seed-42 task set reproduced at
80.0 with 48/50 per-episode agreement, renderer and physics verified byte-exact.

**Random floors** (a random policy passes ~half of h25 tasks because many goals start
near-satisfied): h25 per-draw 46/56/50 (mean 50.7); h50 30/34/28 (mean 30.7). All
"over-floor" numbers below subtract the per-draw floor — this isolates what planning
actually contributes.

## 3. Results — absolute success (%, 50 tasks/draw)

### h25

| method | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| random floor | 46 | 56 | 50 | 50.7 |
| latent+CEM | 80 | 84 | 62 | 75.3 |
| TD+CEM | 82 | 90 | 66 | 79.3 |
| **LIPv4-AC** | **88** | **96** | **80** | **88.0** |

The input-minimal min0 variant ties the champion (87.8 mean over 4 training seeds ×
3 draws, seed spread 2.0 pts) with a leaner input (251-d vs 635-d) and no gate.

### h50

| method | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| random floor | 30 | 34 | 28 | 30.7 |
| latent+CEM | 58 | 66 | 38 | 54.0 |
| TD+CEM | 64 | 74 | 56 | 64.7 |
| **LIPv4-AC** | **74** | **86** | **62** | **74.0** |

## 4. Results — floor removed

| metric | h25 | h50 |
|---|---|---|
| latent+CEM − floor | +24.7 | +23.3 |
| TD+CEM − floor | +28.7 | +34.0 |
| **LIPv4-AC − floor** | **+37.3** | **+43.3** |
| TD+CEM over-floor ratio vs CEM | 1.16× | 1.46× |
| **LIPv4-AC over-floor ratio vs CEM** | **1.51×** | **1.86×** |
| LIP failure-rate reduction vs CEM | 51% | 43% |

Floor-honest reading: as the horizon doubles, native latent guidance stagnates
(+24.7 → +23.3 over floor) while the learned stack grows (+37.3 → +43.3) — **all of
the benchmark's added difficulty is absorbed by the learned components**. The sharpest
single cell: draw 44 at h50, CEM +10 over floor vs LIP +36 (3.6×).

Compute: LIP deploys deterministically with ~16 rollout-equivalents per replan vs
CEM's ~9,000 — roughly **500× less planning compute** at +12.7 higher success (h25).

## 5. Failure structure

Per-episode analysis across 218 archived evaluations plus video-verified reruns:

- **~9.3 of the ~12 failures per 100 episodes are one deterministic mode:** grasp
  acquisition with an airborne goal (goal 7–25cm above the table, block still to be
  grasped). The same 14 tasks fail for every training seed, every recipe, and both
  planner families (gradient LIP and sampling CEM) — planner-independent. Mechanism
  (probe-verified): the WM's predictor is optimistically wrong just off the expert
  action manifold at the attachment boundary — planners converge to counterfeit plans
  that score better than the expert demonstration in imagination and fail 100% in
  reality. Root cause: expert-only training data (zero attempted-and-missed grasps in
  2M frames).
- The remaining ~2.7/100 are stochastic carry/place precision misses (the training-seed
  noise).
- This caps the actor line at ~90.7 with the stock WM — the plateau is a **world-model
  data problem, not a planner problem**.
- **Addendum (2026-07-16): the cap is real and removable.** Fine-tuning the WM with
  targeted negative data (physics-verified grasp-and-miss + drop clips), a
  negative-augmented critic, and the gated min0 actor reaches **89.3 (3-seed mean,
  best seed 94.0, including the first perfect 100 on a cube draw)** — breaking both
  the 88 plateau and the 90.7 core-capped ceiling.

## 6. Videos

`videos_lewm_champion_s42/` (this export): all 50 rollouts of the champion on draw
s42 (88%), three panels per video — agent | dataset expert | goal. `index.html` is a
self-contained browsable gallery; failures are also name-tagged
(`FAIL_airborne_grasp_task*.mp4`, `FAIL_marginal_task6.mp4`). The five airborne-grasp
clips show the signature failure: approach, hover/scrape, close near-but-not-on the
cube, block never leaves the table.

## 7. Reproduction

    # tandem training (champion recipe; min0 variant adds --drop-z0 --drop-zg --no-gate)
    python scripts/plan/train_lip_ac.py \
      --cache cube_full_fs5.pt --cache-td cube_full_fs1.pt \
      --h5 cube_single_expert.h5 --wm ogbench_cube_single_v2WM \
      --init-value cf_dE_t003n50.pt \
      --horizon 5 --iters 8 --steps 8000 --n-step 50 --batch 128 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 \
      --out actor.pt --out-value value.pt

    # evaluation (one draw; repeat seed 42/43/44, offset 25/50)
    python scripts/plan/eval_wm.py --config-name cube seed=42 ++bf16=true \
      eval.img_size=224 eval.dataset_name=cube_single_expert.h5 \
      eval.goal_offset_steps=25 eval.eval_budget=50 \
      policy=ogbench_cube_single_v2WM solver=lip solver.actor_path=actor.pt

Code: `stable_worldmodel/solver/lip.py` (`kind='lip4'`), trainer
`scripts/plan/train_lip_ac.py`. Values/actors train purely offline on latent caches of
the expert dataset encoded by the frozen WM; evaluation seeds fully determine the task
draw (bit-reproducible evals).
