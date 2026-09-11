# Unified-config handoff (2026-09-03)

Two candidate single configs for plain LIPv4 across environments. A is the
locked FINAL from RESULTS.md (best over TwoRoom/Cube/Reacher). B is the
shallow n-step-1 variant that also covers PushT. Medians over seeds; each
seed = mean over eval draws 42-44. Held-out protocol throughout.

## Definitions (the three the appendix leaves thin)

- **max-delta** (chunks): cap on actor hindsight-goal separation,
  `z_g = z_{t+D}`, `D ~ U{1..max_delta}` on the frameskip-5 cache
  (fs x max_delta primitive steps). Actor-side; the critic cap
  `td_max_delta` is unset (= full episode).
- **replay_prob**: per-sample probability of replacing the dataset start
  with the previous actor step's imagined final window (`tr[:, -3:]`, same
  goal, zeroed action history). WM/critic untouched.
- **expand_weight** (value expansion): extra critic term on the refiner's
  own rollout, `d(z0,zg) <- plan_cost + gamma^(H*fs) * d_bar(z_H, zg)`,
  same low expectile, added to the TD loss. Optimistic bound absorber on
  planner-visited states; can co-exploit WM error.
- amax: action clip `[-amax, amax]` on refined plans (checkpoint field,
  deploy-rewritable). acr (`--ac-weight` / `planner.ac_weight`):
  anti-constancy `lambda * ||E_b[sum_t A]||^2 / E_b||sum_t A||^2`.
  ES: snapshots every 2000 steps, argmax on val draws 48-51 (PushT 50-51).

## Config A -- FINAL (best over Reacher / OGBench / TwoRoom)

gamma 0.98 | n-step 50 | expectile 0.03 | amax 2.5 | max-delta 20 | K 8 |
actor depth 3 + critic depth 3, 2x budgets | acr 0.2 | fine ES |
ema 0.005 | window w1 (tw/cu), w2 (reacher) | replay 0 tw / 0.5 else |
expand 1.0 cube / 0 else.

Budgets: tw STEPS 16000 / TD 12000; cube STEPS 12000 / TD 24000;
reacher STEPS 16000; pusht STEPS 12000 / value 24000.
Caches: tw-unig-dcm20b-v1, tw-unig-p-dc345-v1, cu-unig-dc345-v1,
cu-unig-p-dc345-v1. Tags rlp-*-fin-l02-20260827, rlp-*-swp3-l02-20260825.

| cell | h25 / tau.1 | h100 / tau.05 | n |
|---|---|---|---|
| TwoRoom-LeWM | 100.0 | 96.8 | 6 |
| TwoRoom-PLDM | 95.9 | 94.0 | 6 |
| Cube-LeWM | 89.3 | 83.3 | 6 |
| Cube-PLDM | 84.0 | 81.2 | 6 |
| Reacher-LeWM (w2) | 99.7 | 93.7 | 6 |
| Reacher-PLDM (w2) | 99.7 | 91.0 | 6 |
| PushT (w2) | 31.3 | | 3 |

At-or-above paper on 9 of its 10 numbers (deficit TwoRoom-PLDM h100 -2.0).
Fails PushT: the n-step-50 teacher costs -9.3 alone and deep capacity -26
(isolation matrix, tags pusht-iso-*-20260901).

## Config B -- v2-lambda0.5 (best over all four incl. PushT)

As A except: **depth 2 / 1x budgets**, **n-step 1**, **acr 0.5**;
window w4 on PushT. Replay/expand unchanged (tw 0 / others 0.5;
cube+pusht 1.0 / tw+re 0).

Budgets: tw STEPS 8000 / TD 6000; cube STEPS 6000 / TD 12000; reacher
STEPS 6000; pusht STEPS 6000 (recipe defaults).
Caches: tw-v2-n1s2-v1, tw-v2-p-n1s2-v1, cu-v2-n1s2-v1, cu-v2-p-n1s2-v1;
reacher window key includes `_n1`. Tags rlp-*-v2l05[-p][-s345]-20260902,
pusht-v2l05-s*-20260902.

| cell | h25 / tau.1 | h100 / tau.05 | n | delta vs A (hard col) |
|---|---|---|---|---|
| TwoRoom-LeWM | 100.0 | 94.7 | 6 | -2.1 |
| TwoRoom-PLDM | 96.7 | 82.7 | 3 | **-11.3** |
| Cube-LeWM | 90.0 | 86.3 | 6 | +3.0 |
| Cube-PLDM | 87.3 | 85.3 | 3 | +4.1 |
| Reacher-LeWM (w2) | 100.0 | 96.3 | 6 | +2.6 |
| Reacher-PLDM (w2) | 99.3 | 90.7 | 3 | -0.3 |
| PushT (w4) | 64.7 | | 3 | **+33.4** |

PushT reference: latent+CEM 79.3 on the same draws (78/84/76).

Why B works where A does not: n-step 50 is PushT's only corner
incompatibility (gamma and expectile are benign); TwoRoom h100 can be
rescued by any one of {n50, depth, acr 0.5} (substitutes), and acr 0.5 is the
only one PushT tolerates. B's open wound is TwoRoom-PLDM h100 (lottery seed
74.7 returns); rescue probes (lambda 1.0, 2x budget) were cancelled by
directive.

## Per-env knobs that are NOT unified in either config

| knob | tw | cube | reacher | pusht | status |
|---|---|---|---|---|---|
| window frames | 1 | 1 | 2 | 4 | declared env property |
| step budgets | per env | per env | per env | per env | declared |
| replay_prob | 0 | 0.5 | 0.5 | 0.5 | inherited from replication; unify or declare |
| expand_weight | 0 | 1.0 | 0 | 1.0 | inherited; unify or declare |

## PushT ledger under B (medians, n=3 unless noted)

Helped: w4 (+8.7 over w1), acr 0.5 (needed for tw, neutral here).
Null: acr 0.2 vs 0, ES, mean_weight {0, 0.3}, deploy-amax {1.6, 2.0},
robust_m {4, 8}, buffer select (64.0).
Harmful: deep capacity (-26 at base), n-step 50 (-9.3), 2x budget (52.7,
-12), max-delta 8/12 (58.7/60.7).
Promising, open: max-delta 30 = 68.7 at n=1 (n=3 sweep md30/40/50 tags
pusht-v2md{30,40,50}f-s*-20260903, results pending); deploy restarts R=32 =
68.7 (rejected by directive).
Failure anatomy (450 aligned episodes vs CEM): 44 mode-A under-optimization
(self-flagged at t=0), 15 mode-B WM holes (cross-seed universal, imagined
approach ~2.3 vs reality 10-21); union RLP+CEM = 86.2.

## Dyna assets (cube only)

`CU_DYNA=1` swaps in `assets/core/world_model/{lewm,pldm}_cube_dyna`: one
Dyna iteration, **h25-only collection**, deepcap-era planner (amax 1.6),
50/50 expert:on-policy mix by rows, WM fine-tuned lr 1e-5, epoch 1, no
anchor. Under B: LeWM 94.0/82.0 vs 90.0/86.3, PLDM 92.0/85.3 vs 87.3/85.3
(h25 +4, h100 -4). The h100 drop is the h25-only collection forgetting
long-horizon dynamics, not a property of Dyna. A v2 re-run needs: collect at
both offsets, with the v2 planner, anchored fine-tune. Harness lives on
branch `eval-sweep` (`Dyna/dyna_harness/`), cube/tworoom only; TwoRoom Dyna
WMs were lost with the July pods. PushT Dyna = harness port (gated).

## Ops

- Launch from a slim repo copy (exclude .git .pixi assets wandb logs docs):
  the 4.3 GB workdir upload kills TLS (`SSLV3_ALERT_BAD_RECORD_MAC`).
- `sky jobs logs --tail 0 <id>`; completed-job default tail truncates.
- Per-eval results: unigamma `[summary]` via collect_unig.yaml
  (armin-rlp-checkpoints); PushT `results_rlp_s*.txt` on the OLD
  `checkpoints` volume. Reacher latched cells in job logs.
- Every gamma/n-step/depth change needs a fresh CACHE_VERSION.
- Put the seed in EXPERIMENT_TAG; three seeds on one tag race the
  skip-guard and share one planner.
