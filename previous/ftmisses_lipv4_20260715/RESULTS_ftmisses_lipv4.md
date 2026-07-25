# Misses-finetuned LeWM × LIP-AC campaign — RESULTS (2026-07-15/16)

**Headline: the 88.0 plateau is broken and the core grasp family is solvable.**
`ft2nx10gate` (90/10 misses-finetuned WM × grasp-miss negatives ×10 in the critic ×
gated min0 actor) = **88.7 / 94.0 / 85.3 over seeds 0-2 → 3-seed mean 89.3**, vs the
3-seed v2 control 87.6 and the all-time single-seed plateau 88.0. Seed 1 (90/100/92 —
first perfect cube draw ever, s43) exceeds the failure-autopsy's **90.7 ceiling-with-
core-unfixed**, which is arithmetic proof the never-solved core tasks are now passing.

Per-episode core-task fixes (14 core grasp-with-airborne-goal tasks; **zero** fixes in
64+ recipes × all planner families on v2WM, per FAILURE_AUTOPSY.md):

| arm | seed | core fixed | which |
|---|---|---|---|
| ft2nx10gate | 0 | 2/14 | s44: 29, 31 |
| ft2nx10gate | 1 | **7/14** | s42: 11 · s43: 8, 25 · s44: 23, 29, 31, 46 |
| ft2nx10gate | 2 | 5/14 | s42: 11 · s43: 25 · s44: 29, 31, 46 |
| v2nx10gate (control) | 0 | 0/14 | — |
| v2nx10gate (control) | 1 | 1/14 | s44: 29 |

## The full factorial (LIPv4/min0 tandem, schedamax 6k, h25, 3-draw s42/43/44)

| arm | WM | negs | actor | seeds → means |
|---|---|---|---|---|
| v2neg0 | v2 | – | v4 gate-free | 86.7 / 88.0 / 88.0 → **87.6** |
| v2negx10 | v2 | ×10 | v4 | 82.7 / 81.3 / 84.0 → 82.7 |
| v2negx30 | v2 | ×30 | v4 | 84.0 / 76.7 / 83.3 → 81.3 |
| v2nx10gate | v2 | ×10 | gated | 86.0 / 84.7 → 85.4 |
| ftneg0/negx10/negx30 (ft1) | ft1 | 0-30 | v4 | 66.7 / 68.7 / 66.0 (s0) |
| ftnx10gate (ft1) | ft1 | ×10 | gated | 74.7 (s0) |
| ft2neg0 | ft2 | – | v4 | 74.0 (s0) |
| ft2negx10 | ft2 | ×10 | v4 | 70.0 (s0) |
| **ft2nx10gate** | **ft2** | **×10** | **gated** | **88.7 / 94.0 / 85.3 → 89.3** |

Reading: every ingredient is necessary, none sufficient.
- ft2 WM alone (v4 actor): 74.0 — gate-free v4 cannot exploit it (rougher gradients).
- Negatives alone (v2): −5/−6 at 3 seeds, both doses — actively costly on a WM that
  can't imagine misses (the pessimism bleeds into healthy grasp states).
- Gate alone on v2 (+negs): 85.4 — nothing.
- ft2 WM × negatives × gate: +1.7 mean over control, +2-7 core fixes/seed.

## WM diagnostic ladder (planner dependence on WM differential structure)

| probe | v2 | ft1 (miss-only) | ft2 (90/10, ep10) |
|---|---|---|---|
| latent+CEM (encoder geometry) | ~75.3 | 75.3 | 76.7 |
| TD+CEM (rollout rankings) | 82/90/66 = 79.3 | 74/88/66 = 76.0 | 82/86/80 = **82.7** |
| LIP v4 tandem (gradient field) | 86.7 | 66.7 | 74.0 |
| LIP gated tandem | (87.8 banked) | 74.7 | **89.3** |

- ft1 (miss-only diet): encoder intact, rankings −3, **gradient field destroyed (−20)**;
  broke 8-12 previously-solved tasks/draw planner-independently; fixed 0/14 core.
  Diagnosis: blanket attachment pessimism.
- ft2 (90/10 diet): rankings now BEAT v2 (+3.4, driven by s44 66→80 where core lives,
  fixing core 13 under CEM); gradient field still rougher than v2 for the pure
  residual v4 actor, but the gate's per-input damping absorbs exactly that.

## Protocol notes
- New pod bootstrapped from scratch (91.199.227.82:11514, 4×H100/208c, empty volume);
  eval harness anchored bit-exact before any conclusion (champion s43 = 96.0 EXACT,
  TD+CEM s42 = 82.0 EXACT; TD recipe reconstruction behav_corr 0.979, evals identical).
- Negatives: 111 grasp-miss episodes (4541 frames, state-only npz) re-rendered via
  qpos/qvel replay in mujoco (`env.unwrapped` + dataset-matched render flags — first
  render had an info-overlay/target-marker bug, caught by comparing against h5 frames;
  physics verified: median obs err 0.107 on known-convention dims, block lift ≤ 8mm),
  encoded per-WM-family, merged into the fs1 critic cache by episode replication
  (ep-base 100000, NStepGoalSampler is episode-uniform → ×K replicas = ×K batch share).
- Actor data untouched by negatives (miss eps too short for the context sampler —
  by design the critic is the entry point).
- All trainings train_lip_ac.py tandem warm, schedamax schedules, 6k steps (S7: 6k=8k),
  3-draw eval s42/43/44, eval_budget 50, select by 3-draw mean only.

## Assets
- Winning actors + teachers: `actors/lipft_ft2nx10gate_s{0,1,2}.pt` (+ `_value.pt`),
  ft2 TD init `actors/ft2_dE_t003n50.pt` (local copies).
- Pod: /workspace/actors, /workspace/metrics, caches `cube_ft2_*`, ckpt
  `/workspace/ckpts/ogbench_cube_single_ftmisses2` (remapped 90/10 WM).
- Local WM ckpts: `stable-worldmodel/checkpoints/ogbench_cube_single_ftmisses{,2}`
  (HF-ViT→repo key remap; remap script pattern in this campaign's session).
- Full scores: `summary_ftm.csv`; event log: `driver_ftm.log`.

## Open threads
1. ft2nx10gate seed spread is wide (85.3-94.0, n=3). More seeds would firm the mean;
   the core-fix claim is already solid (5-7/14 at 2 of 3 seeds vs 0 historical).
2. s42 core {14, 17, 27, 38} still unsolved by the actor (11 now falls) — residual
   family for the next WM-data iteration (more diverse miss coverage? drops?).
3. Best-seed deploy: s1 (94.0) is the natural deployment actor.
4. Negatives dose beyond ×10 untested on ft2 (only ×10 ran with the gate).
5. h50 unprobed for the winner.
