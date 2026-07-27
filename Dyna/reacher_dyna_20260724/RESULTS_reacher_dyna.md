# Reacher Dyna round-1 across three WM bases (2026-07-25)

> **CORRECTIONS 2026-07-27 — read before any number below.**
>
> 1. **Every divergence figure in this document is an artifact and is RETRACTED**
>    (the 42.3 "baseline", the ~49.5 post-Dyna values, and the mechanism claim
>    built on them). `dyna_harness/ab_divergence.py` pairs consecutive probe
>    rounds *by position*, but round t+1 contains only the envs still alive
>    (50 -> 36 -> 30 -> ...), so nearly every row was mismatched. Joining on the
>    goal latent instead gives **+1.27 (LeWM)** and **+3.01 (PLDM)**. The claim
>    "divergence stays high, so the gain is WM-fidelity repair rather than
>    exploitation-closure" rested on the broken numbers and does not stand.
>    The same tool produced the OGBench campaign's 20.9 / 36.1 figures.
>
> 2. **The Dyna loop trained on rollouts from its own eval set.**
>    `reacher_collect_r1.sh` never overrode `eval.dataset_name`, so on-policy
>    collection drew tasks from the same 1,024-episode file the evaluation drew
>    from (~2,100 draws over 1,024 episodes = essentially full coverage).
>    Fixed by an `eval.ep_range` filter: collection now uses episodes 0:8000 and
>    evaluation 8000:10000 over the same canonical file.
>
> 3. **Headline numbers superseded.** On the authors' canonical data with
>    held-out draws: LeWM LIP 76.9 -> 78.2 (+1.3, NOT +29.3); PLDM LIP
>    78.8 -> 89.2 (+10.4, which *does* reproduce). Full matrix in
>    `MATRIX_reacher.md`. The 80/20 arm was used throughout, not the 50/50
>    specified; the 50/50 WM exists but never received a fresh actor.

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
| **DINO** (dinowmnp) | 24.7 | **72.0** (h25 s42) | on hold — see below | — | — | — |

- **LeJEPA**: LIP started 15 pts *behind* CEM (a WM-fidelity gap) → Dyna closes
  it and overtakes by +7. Biggest swing.
- **PLDM**: LIP started ~even with CEM → Dyna still lifts it +9.4, crossing CEM
  by +7. Smaller swing (less headroom), same direction.
- **DINO** (dinowmnp): **the WM is FINE; our LIP stack on it is not.** With
  `get_cost` implemented for the flat-token class (it was missing, which is why
  DINO had no sampling baselines at all), **CEM = 72.0** on the base exactly as
  shipped — the same neighbourhood as lejepa 78.7 / pldm 82.7 / the authors'
  ~79. So the LIP 24.7 is *not* a broken world model. DINO Dyna is on hold:
  Dyna fine-tunes the WM, and this WM is already good — the defect is in the
  LIP/value stack on it. Full investigation below.

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

## DINO investigation (2026-07-25) — the WM was never the problem

The reacher DINO base (`dinowmnp`, frozen DINOv2-small, flat tokens 196×384 =
75264-d, pixels-only) gave LIP ~24.7, and its converted WM appeared to predict
*worse than copy-last* open-loop (1-step pred/copy ratio 2.44). That looked like
a broken conversion. It was not.

**Six hypotheses tested and eliminated** (all forward-pass diagnostics):

| hypothesis | verdict |
|---|---|
| our `DinoWMTokens` re-wrapping is unfaithful | **No** — forward is byte-identical (max diff 0.0) to the PreJEPA *training* class on pixel tokens, predictions and action embeddings |
| load bug / dropped buffer | **No** — shipped `_object.ckpt` vs `_weights.ckpt`: 300 keys each, 0 only-in-either, 0 nonzero value diffs; object config matches our reconstruction (history 3, 196 patches, action-only, interpolate=True) |
| predictor is residual (missing skip) | **No** — `pred + z_last` is 22× *worse*; offset is near-orthogonal to the feature mean (cos 0.13) |
| DINOv2 pos-embedding interpolation convention (old `scale_factor`+0.1 vs new `size=`) | **No** — 2.64 vs 2.70 |
| image normalization | **No** — ImageNet best of four (2.64 vs raw[0,1] 6.60, [-1,1] 4.23, ×255 28.1) |
| resize / action scale / frameskip | **No** — swept, already optimal (224→196, scale 1.0, fs5) |

**What is real:** a fixed additive offset `c` in feature space, ‖c‖ = 213.6,
data-independent (split-half cosine 0.9985). Calibrating it out on one half and
testing on the **held-out** half takes 1-step ratio 2.37 → **0.42** and 4-step
→ **0.32**; retrieval@1 of the true next frame is 0.943 (copy-last 0.391). So
the predictor's *dynamics* are good and only a DC term is off.

**The decisive test inverted the fix.** CEM (h25, s42, n=50) on the two bases:

| base | CEM |
|---|---|
| `dinowmnp_reacher` (as shipped, offset present) | **72.0** |
| `dinowmnp_cal_reacher` (offset calibrated out) | 46.0 |

Removing the offset improves 1-step latent error 5.6× and **costs 26 points of
planning**. A rank diagnostic (true 20-step plan vs 63 random plans) shows the
two costs rank on-manifold plans indistinguishably (mean rank 3.20 vs 3.45), so
the gap appears only under CEM's 30 iterations of optimization pressure. The
offset's cross term `2c·(f − z_g)` in a fixed 213-norm direction acts as an
accidental **off-manifold penalty**: it suppresses exactly the model-exploitation
that unconstrained optimization would otherwise find. Calibrating it away is a
better *predictor* and a worse *planner* — the same optimizer's-curse theme this
whole campaign is about, arriving from the opposite direction.

**Conclusion for DINO:** the base plans fine as shipped (CEM 72.0 ≈ authors'
79). The LIP 24.7 is a defect in our LIP/value stack on this WM, and the prime
suspects are the compute concessions forced by the 75264-d latent: the latent
cache was capped at 200k rows (10% of the data), and LIP trained at K=4 /
batch 16 (vs K=8 / batch 128 elsewhere — 8× fewer samples at equal steps), plus
a TD quasimetric asked to work on a 75264-d input. Running Dyna on DINO is
premature until LIP works on it, since Dyna's job is to fix the *WM*.

Code produced: `DinoWMTokens.get_cost` (+ public `rollout_traj`, chunked to
avoid OOM at B·S=3000 candidates × 588 tokens) — this unlocks CEM/MPPI/GD/TD+CEM
on every flat-token WM; and an optional `pred_bias` buffer with a
load-state-dict hook so existing checkpoints are unaffected.

### DINO root cause CONFIRMED: a train/eval distribution mismatch in the value

Not a broken WM, not a data-starved value, not the flat-token dimensionality.
The TD value is **trained** on pairs of ENCODER latents (both arguments come
from the cache), but at planning time the cost hook calls
`cost(predicted_emb, goal_emb)` — first argument from the **predictor**, second
from the **encoder**. This DINO predictor's output sits a fixed ‖c‖ = 213.6
(31% of ‖enc‖) away from encoder space, so the learned value is evaluated far
outside its training distribution. lejepa/pldm have essentially no such offset
(1.2–14%), which is why only DINO collapses.

**One-variable proof** — same value, same solver, same seed, only the offset
removed from the predictor:

| config (h25, s42, n=50) | success |
|---|---|
| plain CEM, raw WM | 72.0 |
| plain CEM, calibrated WM | 46.0 |
| TD+CEM, raw WM (the collapse) | **22.0** |
| **TD+CEM, calibrated WM** | **60.0** |

The value nearly triples (22.0 → 60.0) and beats the raw-MSE cost on the same
WM (+14). The two effects are **separable and opposed**: calibration costs the
raw-MSE cost 26 points and gains the learned value 38.

**Mechanism is a near-field crush, not a rank collapse.** Held-out audits
(episodes 9500–9999) show all four values are equally healthy on their
*training* distribution — AUDIT-1 enc→enc Spearman is +0.768–0.771 for DINO,
lejepa and pldm alike. Under the deployment pairing (pred→enc) DINO's rank
correlation only slips 0.768→0.620, but `V(k=1)` reads **9.91 instead of 2.85**,
so k=1 and k=5 become indistinguishable: broad ordering survives, the fine
discrimination CEM needs does not.

| value / WM | ‖E[pred−enc]‖ | A1 enc→enc ρ | A1 pred→enc ρ | V(k=1) | A2 top-1 agree | TD+CEM |
|---|---|---|---|---|---|---|
| DINO, raw WM | 213.5 (31%) | +0.768 | +0.620 | 2.85→9.91 | **0.125** | 22.0 |
| DINO, calibrated WM | 9.5 (1.4%) | +0.768 | +0.722 | 2.85→6.13 | **0.438** | 60.0 |
| lejepa (known-good) | 2.02 (14%) | +0.771 | +0.697 | 2.47→5.61 | **0.458** | 55.7 |
| pldm (known-good) | 0.19 (1.2%) | +0.770 | +0.769 | 2.47→2.64 | **0.896** | 86.7 |

**A 90-second screener that replaces a 62-minute eval.** AUDIT-2 (rank agreement
with the raw-latent-MSE cost on a shared candidate set — a reacher analogue of
TwoRoom's SCSA) predicts planning almost exactly: top-1 agreement 0.125→22.0,
0.438→60.0, 0.458→55.7, 0.896→86.7, i.e. **TD+CEM ≈ 100 × top-1**, Pearson
r ≈ 0.98. Note that AUDIT-1 alone would NOT have discriminated these cases: the
load-bearing parts are the pred→enc condition and the top-1 metric. Tooling:
`/workspace/audit_value.py`. This is the methodology TwoRoom had (it cached with
`--state-key state` and audited the metric directly) and reacher was missing.

**Correction to the record:** two earlier explanations in this campaign were
wrong and are retracted — (1) "DINOv2 features are uninformative on reacher"
(retrieval@1 of the true next frame is 0.943, and CEM scores 72/94), and (2)
"the value was data-starved at 200k rows × 75264-d" (the TwoRoom DINO value
trained on ~103k rows at the same width with identical TD settings, and all
values here audit identically well on their training distribution).

**Next:** train the value on the deployment pairing (first argument from a
predictor rollout, second from the encoder) so we keep the raw WM's better
dynamics *and* a distribution-matched value, instead of paying the calibration
tax. Screened by AUDIT-2 top-1; 62-min eval spent only on the winner.
