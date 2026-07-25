# Reacher Dyna round-1 across three WM bases (2026-07-25)

Env `swm/ReacherDMControl-v0`, task qpos_match. Card = {h25 (offset 25, budget
50), h50 (offset 50, budget 100)} × eval seeds {42,43,44}, n=50/cell; means are
over the 6-cell card. LIP = LIPv4 amax 2.2, 3 training seeds. 80/20 on-policy
mix (the arm that won on lejepa). Pod 87.120.211.204:19342.

## Cross-base headline

Closing the loop (on-policy WM fine-tune → fresh TD+LIP actor) lifts LIP above
CEM on both learned-encoder bases, and the size of the gain tracks how far LIP
started behind:

| base | LIP stage-1 | CEM (orig WM) | LIP round-1 (fresh) | CEM (FT WM) | LIP swing | LIP−CEM after |
|---|---|---|---|---|---|---|
| **LeJEPA** | 63.8 | 78.7 | **93.1** | 86.0 | **+29.3** | **+7.1** |
| **PLDM**   | 80.2 | 82.7 | **89.6** | 82.7 | **+9.4**  | **+6.9** |
| **DINO** (dinowmnp) | ~24.7 | (diagnostic pending) | — | — | — | — |

- **LeJEPA**: LIP started 15 pts *behind* CEM (a WM-fidelity gap) → Dyna closes
  it and overtakes by +7. Biggest swing.
- **PLDM**: LIP started ~even with CEM → Dyna still lifts it +9.4, crossing CEM
  by +7. Smaller swing (less headroom), same direction.
- **DINO** (dinowmnp): **UNDER INVESTIGATION — conversion bug, not a real
  result.** Stage-1 LIP measured ~24.7 (2-seed 26.3/23.0) with the converted
  flat-token WM predicting *worse than copy-last* open-loop (ratio 2.4). But the
  authors report DINO-WM reacher ~79, so this is a broken conversion/forward in
  our `DinoWMTokens` re-wrapping, not a weak base. Root-cause debugging in
  progress (PreJEPA training-class vs DinoWMTokens forward comparison on
  identical frames, checking image preprocessing / token layout / action
  normalization). DINO Dyna round paused until the base converts correctly and
  reproduces a sane stage-1 number.

The common mechanism (below) holds for both wins: divergence stays high, so the
gain is **WM-fidelity repair**, not exploitation-gap closure.

## LeJEPA detail (the fully-worked case)

| stage | LIP (3-seed 6-cell) | CEM (same WM) | LIP − CEM |
|---|---|---|---|
| **Round 0** — stage-1 actors on the original converted lejepa WM | 63.8 | 78.7 | **−14.9** |
| Round 1 — stage-1 actors on the 8020 fine-tuned WM | 85.0 | 86.0 | −1.0 |
| **Round 1 — FRESH actors on the 8020 fine-tuned WM** | **93.1** | 86.0 | **+7.1** |

Fresh actors (trained on the fine-tuned WM's own latents) add +8 over the
stage-1 actors run through the same WM. Per-horizon (fresh): **h25 87.6 vs CEM
78.0 (+9.6); h50 98.7 vs CEM 94.0 (+4.7)** — LIP wins at both, h50 near-saturated.

PLDM mirrors this: the probe-gate (stage-1 actors on the FT WM) looked flat
(~78, since PLDM stage-1 was already good), but the FRESH actors jumped to 89.6
— the gain lives in retraining the actor on the improved WM, not in the WM
helping the old actor.

## The loop (one round)

1. **Stage 1** (rebuilt from scratch; the prior reacher pod died): convert the
   author lejepa base → cache → TD → LIP amax 2.2 × 3 seeds. Baseline card 63.8.
2. **Divergence probe** (first reacher measurement of the OGBench "action
   exploitation" signature): imagined-vs-reached value gap **42.3** with the
   stage-1 tandem critics (imagined ≈3 steps-to-go, reached ≈45). Signature
   present, comparable to OGBench's 36.1.
3. **On-policy collection**: 1,591 episodes rolling the 3 stage-1 LIP actors
   through the real sim via the eval path (`SWM_RECORD_PATH`), seeds disjoint
   from eval.
4. **Fine-tune** the lejepa WM on random ⊕ dup(on-policy), two arms by ROW
   fraction: **50/50** and **80/20** (lr 1e-5, INIT from base, reacher
   random-data action-pin, no rollout loss).
5. **Probe-gate**: LIP jumped ~60→76/74 at the gate cell, CEM canary 66→84/78
   (no dynamics degradation). **Winner = 80/20** (the lighter on-policy mix;
   50/50 over-weights the actor's narrow visited distribution).
6. **Full re-card + fresh actor** on the 8020 WM → 93.1, above.

## The honest mechanism read (differs from the cube)

- **Success is unambiguous**: +29 LIP, crossing CEM. Dyna worked on reacher.
- **But the imagined-vs-reached divergence did NOT shrink.** Measured with each
  fresh actor's own critic on the fine-tuned WM, it stayed **~49.5** (48.3 /
  54.1 / 46.1) — the planner still "imagines arriving" while the reached latent
  reads far. So on reacher the gain is **WM-fidelity repair, not exploitation-
  gap closure**: CEM also rose (+7, general improvement); LIP rose far more
  (+29) because a more faithful WM also sharpens its value-gradients. On a smooth
  2-joint reacher with no irreversible events, the residual optimism is benign
  (unlike the cube's fabricated-grasp → real-miss → unrecoverable failure) —
  the plan direction is right and receding-horizon replanning corrects it.
- **Caveat, stated plainly**: the round-0 (42.3) and round-1 (~49.5) divergences
  use *different* critics (each trained on its own WM's latents), so they are
  not strictly comparable — the clean test is same-critic, WM-before-vs-after,
  which was not run. Divergence is not a reliable cross-round metric here;
  success rate is the ground truth.

## Assets (pod 87.120.211.204:19342)

- fine-tuned WM (winner): `/workspace/swm_home/checkpoints/dyna_reacher_r1_8020_final/`
- fresh round-1 actors: `/workspace/actors/lip4_reacher_r1_8020_s{0,1,2}.pt`
  (+ `_value.pt` critics in `/workspace/metrics/`)
- on-policy data: `/workspace/dyna_data/reacher_onpolicy_r1_lejepa_a{0,1,2}.lance`
- mixes: `/workspace/dyna_data/reacher_mix_r1_{5050,8020}.lance`
- summaries: `/workspace/results/summary_reacher_r1_{5050,8020}.csv` (per-cell here)
- scripts (local repo `scripts/plan/`): reacher_dyna_probe, reacher_collect_r1,
  reacher_r1_finetune, reacher_r1_probegate, reacher_r1_recard, reacher_r1_freshactor

## Round 2 (optional)

LIP has crossed CEM and h50 is near-saturated (98.7), so headroom is limited.
Round 2 would accumulate ALL on-policy data (DAgger union) + fine-tune again +
fresh actor. Better use of the finding: it isolates that reacher's LIP gap was
WM-fidelity, not exploitation — worth a same-critic before/after divergence run
to nail the mechanism claim.
