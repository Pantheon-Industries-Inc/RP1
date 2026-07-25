# HWM × LIPv4 on LeWM / OGBench-cube episodic eval — design spec (2026-07-18)

Goal: port the HWM hierarchy (arXiv 2604.03208, FAIR/NYU) onto the LeWM cube stack and
plan with LIPv4 at both levels. Fourth backbone for HWM (they did VJEPA2-AC, DINO-WM,
PLDM); first learned-iterative-planner instantiation (they use CEM/MPPI).

## Why we expect this to bite (two concrete targets)

1. **h25 core-grasp plateau.** Our 88-plateau autopsy: 14 core grasp-with-airborne-goal
   tasks carry ~70% of failures, planner-independent; WM imagines expert actions fine
   (93%). HWM Table 4 is the same phenomenon on Franka: flat 0% -> 80% with a *manual*
   grasp subgoal; their hierarchy recovers 70% automatically. A learned high level that
   proposes "grasp first" waypoints attacks exactly our non-greedy failure class.
2. **h200 universal cliff.** Flat latent/TD/LIP all 0.0 at h200 (TRM campaign). HWM's
   mechanism (fewer autoregressive steps at coarse scale + tiny macro-action search
   space) is aimed precisely at this regime (PushT d75 17->61, maze hard 44->83).

## Constraint that forces the frozen-base design

From-scratch LeWM retrains are broken-for-planning (latent+CEM 51.3 / LIPv4 56.9 vs
v2WM 75.3 / 87.6; ~46 floor). IDM-augmented retrain (finished 2026-07-18 22:38) did NOT
fix it: 54.2. So: **v2WM encoder E and predictor F1 stay frozen** = the low level.
This is HWM-faithful: their PLDM high level "reuses the backbone of a pretrained
low-level world model and keeps it frozen, training only the high-level world model";
DINO-WM/VJEPA2 encoders are frozen too. We train only: action encoder A_psi + high-level
predictor F2 (+ LIP heads).

## Architecture (paper-faithful, adapted to cached pooled latents)

Latent space: z = v2WM encode(...)["emb"][:,0] per frame — the SAME space the caches,
TD critics, and LIPv4 actors already use (cube_v2wm_fs1/fs5.pt).

- **A_psi (action encoder)**: transformer over the chunk of k_hl primitive actions
  between consecutive waypoints (paper A.2/A.3: transformer, CLS token -> MLP -> macro
  action). Macro dim: 8 primary (their maze), sweep {4, 8, 16} cheaply. Their Fig 7:
  too small = invalid plans, too big = unreachable subgoals; 4-8 is the sweet spot.
- **F2 (high-level WM)**: causal predictor over interleaved (z_{t_i}, l_{t_i}) at fixed
  stride k_hl env steps, predicts z_{t_{i+1}}. Teacher-forced l1 loss (their eq. 1
  analog; gamma_tf=1, gamma_roll=0 to start; optional rollout loss later — PLDM recipe
  used pure rollout, DINO-WM/VJEPA2 pure TF). Arch: transformer on the (l, z) token
  sequence (context N=6 waypoints), residual-MLP variant as ablation if tokens
  underperform.
- **Stride k_hl**: fixed (their maze design; D.1 ablation 10-12 native transitions
  optimal, degrades <10). For cube: 1 LL WM step = 5 env steps (fs5); LL LIPv4 plans
  horizon 5 WM steps = 25 env steps. Primary k_hl = 25 env steps (subgoal exactly one
  LL plan away; h200 = 8 macro steps). Sweep {15, 25, 50} if time.
- Waypoint training data: fs1 cache latents (10k eps x 201 steps, already encoded,
  frozen-e2e-consistent) + action chunks from the h5. Segments: per episode, all
  windows of N=6 waypoints at stride k_hl (paper maze: 60-step subsample, 6 waypoints,
  stride 10).

## Hierarchical LIPv4

- **Critic**: the TD quasimetric d(z, z_g) is stride-agnostic; reuse v2 teacher
  cf_dE_t003n50 for the HL actor's gradV. NOTE td-cache-stride lesson: do NOT train a
  TD on the fs25 cache (coarse cache hid grasp events and broke picks before) — if a
  fresh HL critic is needed, train on fs1 with longer n-step (100/200 env steps) for
  long-horizon calibration.
- **HL actor**: LIPv4 arch v4 ([A, gradV, E] inputs, gate-free min0, schedamax-6k
  recipe as base), action space = macro-actions z-scored per-dim against the empirical
  macro-action distribution (A_psi over all dataset chunks); amax clip ~3.5 in z-space
  mirroring the LL recipe. Imagination through frozen F2. Horizon H_hl: 2 (h50) / 3
  (h75) / 8 (h200), trained at H_hl=8 or matched-per-eval (decide after F2 quality
  probe).
- **Subgoal handoff**: z_sub = first predicted waypoint of the optimized HL plan
  (paper 2.1). LL = existing 87.6 actors (lipabl_ctrl_s{0,1,2}.pt) with goal := z_sub.
  LL executes its plan; replan both levels every k_ll env steps (harness cadence —
  confirm from eval_wm.py; HWM replans HL+LL every k env steps, k=4/5).
- **hcem baseline**: CEM over macro-actions at HL (theirs) + CEM at LL = HWM-faithful,
  isolates hierarchy-vs-LIP contributions. 2x2: {flat, hier} x {CEM, LIPv4}.

## Validation gates (before any eval burns GPU-days)

G1. Relay/bootstrap integrity: WM loads, h5 verified (2,010,000 frames), anchor evals
    reproduce v2WM refs bit-exact (champion s43=96.0, TD+CEM s42=82.0) on the new pod.
G2. Macro-action stats sane (z-scored dims, no collapse: per-dim std > 0.1 after
    z-scoring source chunks; cosine-sim structure across chunks).
G3. Fig-6 analog: waypoint-prediction l1 of F2 (1 step = k_hl env steps) vs LL F1
    autoregressive rollout (k_hl/5 steps) on held-out episodes (eps 9500-9999); F2 must
    win at >= 2*k_hl env-step horizons, roughly match at k_hl.
G4. Subgoal decodability probe: nearest-neighbor of predicted z_sub in the fs1 cache
    lands on plausible intermediate states (block-height/grasp readout via the R^2=0.99
    latent->state probe from augment_20260718).
G5. HL plan validity on the 14 core grasp tasks: does the first subgoal have the block
    grasped/lifted? (privileged state of NN + probe readout).

## Eval protocol

Canonical cube protocol: full 10k-ep h5, task draws = eval seeds 42/43/44 (seed drives
draw + solver rng), eval_budget=50, success = block within 4cm, ++bf16=true, img 224,
OMP_NUM_THREADS=16, MUJOCO_GL=osmesa.

Card (rows = methods, cols = horizons h25/h50/h75/h200; 3 draws each):
  flat latent+CEM (refs h25 75.3), flat LIPv4 (87.6), hcem, hlip.
Flat h50/h75/h100/h200 baselines re-run on this pod/harness for clean comparison.
Per-episode outcomes on the 14 core tasks (s42:{11,14,17,27,38}, s43:{8,25},
s44:{12,13,23,24,29,31,46}) vs v2 history.

## Assets / pod

New pod: root@31.24.80.32 -p 14476, 6x H100 80GB, /workspace 280G netvol.
From old pod (216.243.220.215:11270): code/stable-worldmodel, scripts, v2WM ckpt,
cube_v2wm_fs1/fs5.pt caches, actors (lipabl_ctrl_s* = 87.6 recipe 3 seeds,
lip_ac90_schedamax = champion 88, lipft_ft2nx10gate_s1 = 89.3 winner), metrics
(cf_dE_t003n50 v2 TD teacher + per-actor critics). h5 from HF (quentinll/lewm-cube).

## Open questions (blocking code, explorer agents out)

O1. Exact z shape ("emb"[:,0] = pooled vector D or patch tokens PxD?) and what space
    the lip solver matches at eval (goal encoding vs imagined rollout).
O2. LIPv4 v4 actor exact input layout + how E/gradV are computed in train_lip_ac.py;
    horizon/n-step unit semantics (env vs WM steps).
O3. eval_wm.py replan cadence + how goal_offset_steps=200 interacts with episode
    length 201 and the solver's horizon; where solvers register (to add hlip/hcem).
O4. HWM_PLDM exact class/loss shapes to mirror (hjepa.py) — waypoint batching, action
    encoder details, MPPI-over-macro bounds/clipping.
