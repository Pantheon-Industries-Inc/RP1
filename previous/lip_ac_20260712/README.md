# LIP-AC: TD critic + LIP actor trained in tandem (2026-07-12)

## Why

The sequential protocol (train TD → select by CEM → train LIP against the frozen
winner) selects the value by a zeroth-order criterion: CEM only needs correct
*ranking* of sampled plans. LIP consumes the value's *gradient field* through the
WM, so the CEM-best TD is not necessarily the best LIP teacher. PushT is the
existing evidence: the CEM-picked teacher `td_e0.03_n5` scores **76/54**
(h25/h50, s42) under CEM but its LIP student only **66/36**.

LIP-AC trains both jointly, DDPG/TD3-style, with the planner as a K-step
learned-optimizer actor.

## The trainer — `stable-worldmodel/scripts/plan/train_lip_ac.py`

Design (see the module docstring for details):

- **Critic** `d_phi`: n-step expectile TD on the fs1 (dense) cache — identical
  loss/sampler to `train_metric.py --learner td`. The TD loss is the *only*
  gradient that reaches phi.
- **Teacher** `EMA(d_phi)` (Polyak, `--ema-tau 0.005`): provides TD bootstrap
  targets AND everything the actor sees (gradient features, refinement loss).
  Actor gradients flow *through* it, never *into* it — the critic cannot learn
  to "make plans look good" (the classic actor-critic collapse).
- **Actor**: exactly `train_lip.py`'s objective (K refinement iterations,
  loss = E_final + 0.1·mean E_k, backprop through the frozen WM), against the
  teacher.
- **Schedule**: `--pretrain` critic-only warmup (auto: 2000 fresh / 0 when
  warm-started via `--init-value`), then `--critic-ratio` critic steps per actor
  step; critic+teacher frozen after `--freeze-critic-frac` (default 0.8) of
  actor steps so the lr-sensitive planner settles on a stationary value.
- **Optional feedback** `--expand-weight w`: the actor's imagined terminal
  latents become extra TD backups `d(z0,zg) ← H·fs + d_bar(zT,zg)` — value
  expansion on planner-visited states. With low expectile this is a one-sided
  bound absorber (good plans tighten d, bad plans barely raise it). Off by
  default; it lets the pair co-exploit WM errors, so always compare against w=0.
- **Outputs**: actor checkpoint in `train_lip.py` format + the teacher via
  `save_metric`. The deployed value is the EMA teacher (what the actor was
  optimized against), so eval (`solver=lip`, `+metric=`) works unchanged.

Smoke-tested end-to-end on synthetic data (fresh / warm-start / expand paths,
plus checkpoint round-trip through the `LIPSolver` and `load_metric` load
paths).

## The experiment — `run_ac_pusht.sh`

Four arms, all reusing the sequential winners' recipes (actor K8/lr3e-4/8k,
critic e0.03/n5):

| arm | critic init | teacher schedule | expansion |
|---|---|---|---|
| `ac_fresh` | scratch (+2k pretrain) | freeze @ 80% | — |
| `ac_warm` | sequential TD winner | freeze @ 80% | — |
| `ac_nofreeze` | sequential TD winner | never frozen | — |
| `ac_expand03` | sequential TD winner | freeze @ 80% | w=0.3 |

Phases: train all 4 in parallel → LIP-eval each on s42 h25+h50 (selection) →
CEM-eval each arm's *teacher* (diagnostic: did tandem training move the value's
CEM quality?) → confirm the winner on all 6 cells (h25/h50 × s42/43/44).

Baselines to beat (already in the pod's `results/summary.csv`):
`lipwin_k8_lr3e-4_st8000_*` (66/68/66 h25, 40/26/… h50) and
`tdwin_td_e0.03_n5_*` (76/80/74 h25, 54/42/42 h50).

## Launching (PushT pod from the 2026-07-11 run)

```bash
scp -P PORT -i ~/.ssh/id_ed25519 \
  stable-worldmodel/scripts/plan/train_lip_ac.py \
  root@HOST:/workspace/code/stable-worldmodel/scripts/plan/
scp -P PORT -i ~/.ssh/id_ed25519 lip_ac_20260712/run_ac_pusht.sh root@HOST:/workspace/
ssh -p PORT -i ~/.ssh/id_ed25519 root@HOST \
  'cd /workspace && nohup bash run_ac_pusht.sh > logs/run_ac_nohup.log 2>&1 & echo started'
```

Reruns are idempotent (cached rows in `summary.csv` / existing checkpoints are
skipped; delete to force).

## Readouts

1. **Main**: `lipac_*_s42` sum vs sequential LIP (102). Any arm ≫ 102 says the
   sequential teacher was the bottleneck; parity says the PushT gap is not a
   teacher-selection problem (points back at the rugged-landscape/actor side).
2. **`accem_*` diagnostic**: teacher CEM score vs 76/54. Dropped → tandem hurt
   the value's ranking quality (trade-off); held → tandem is a free lunch.
3. **`fresh` vs `warm`**: does AC even need the sequential TD stage?
4. **`nofreeze` vs `warm`**: is the settle-phase heuristic load-bearing?
5. **`expand03` vs `warm`**: does shaping d on planner-visited states help, or
   does WM-error exploitation bite?

TwoRoom is saturated (LIP ≈ CEM ≈ 100), so it's only useful as a
does-not-break check — adapt the driver by swapping WM/caches/config-name and
critic hypers to e0.1/n50 if wanted.
