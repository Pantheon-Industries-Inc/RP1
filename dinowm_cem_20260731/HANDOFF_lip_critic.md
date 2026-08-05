# HANDOFF — DINO-WM TD/LIP campaign (2026-08-01 → 08-03)

**Pod `157.66.255.84`, port `19403`** (the port changes on every pod restart —
it has been 11890, then 19403). 4×H200. `/workspace` is an MFS network volume and
has survived two pod swaps intact; the container's pip env and `/root` do **not**.

Read alongside [`RESULTS_lip_critic_20260802.md`](RESULTS_lip_critic_20260802.md)
(fuller detail on bugs and falsified hypotheses) and
[`HANDOFF.md`](HANDOFF.md) (the original checkpoint identification).

---

## 1. READ THIS FIRST: THE FLOOR IS HIGH — every magnitude here is compressed

**MEASURED 2026-08-03 (complete): `nomove` = 54.7, `random` = 55.3.**
Random actions score the same as doing nothing, so the floor is solidly ~55.

| | d44 | d42 | d43 | mean |
|---|---|---|---|---|
| nomove | 54.0 | 58.0 | 52.0 | **54.7** |
| random | 54.0 | 60.0 | 52.0 | **55.3** |

So the eval's usable range is only ~25 points wide. Floor-corrected skill:

| planner | raw | above floor (54.7) | share of CEM |
|---|---|---|---|
| **plain latent+CEM** | 80.0 | **+25.3** | 100 % |
| TD+CEM (`std96k` critic, n=2) | 76.0 | +21.3 | ~84 % |
| TD+CEM (`raw24k` critic) | 76.0 | +21.3 | ~84 % |
| LIPv4 `fx96_exp0.5` | 67.3 | +12.6 | ~50 % |
| LIPv4 `fx96_ctrl` | 66.0 | +11.3 | **~45 %** |
| LIPv4 `fx96_rep0.25` | 64.0 | +9.3 | ~37 % |
| random actions | 55.3 | +0.6 | ~2 % |

**What this does and does not overturn:**

* The **ordering survives** — CEM > TD+CEM > LIP, all three genuinely above
  floor. LIP is not broken; it captures under half the available skill.
* **Magnitudes were roughly doubled** by the floor. "66 vs 80" is really
  "+11 vs +25 on a 25-point scale".
* **The LIP variants are UNRESOLVABLE.** Their 62–72 spread sits on a per-draw
  floor of 52–60 with SE ≈ 6, so amax, actor-lr, steps, expand, replay and the
  critic upgrade were all compared *inside the noise band*. That explains why
  ~20 configurations returned the same answer and why three differently-trained
  actors all hit exactly 62.0 on draw 43.
* The claim that a 17 %-better critic "moved planning by exactly 0.0" is
  literally true but was never resolvable at this n against this floor. Do not
  cite it as evidence that critic quality is irrelevant.

**⇒ The top priority is a HARDER EVAL, not more method work.** See §9.1.

The floor run is COMPLETE. Open items elsewhere are listed in §6.

**Lesson worth carrying:** this control costs 2 minutes per cell and calibrates
every number in the campaign. It should have been the first thing run, not the
last.

## 2. Numbers measured (all held-out episodes 8000–9999, 196 px, fp32, n=50/draw)

| planner | draws 44/42/43 | mean |
|---|---|---|
| **plain latent+CEM** | 78 / 82 / 80 | **80.0** |
| TD+CEM, old critic (`raw24k`) | 70 / 78 / 80 | 76.0 |
| TD+CEM, **new critic** (`std96k`) | 76 / 76 / *never ran* | **76.0** (n=2) |
| LIP `fx96_ctrl` (new critic) | 70 / 66 / 62 | 66.0 |
| LIP `fx96_exp0.5` | 70 / 70 / 62 | 67.3 |
| LIP `fx96_rep0.25` | 64 / 66 / 62 | 64.0 |
| LIP post-action-fix, old critic | 68 / 70 / 70 | 69.3 |
| **floor — nomove** | 54 / 58 / 52 | **54.7** |
| **floor — random** | 54 / 60 / 52 | **55.3** |

SE is ~6 points per draw, ~3.5 for a 3-draw mean. Differences under ~7 points
between learned planners are not resolvable at this n.

Critic quality (probe on held-out episodes, spearman / pair_acc / monotone):

| critic | samples | scores |
|---|---|---|
| stock recipe equivalent | 6.1M | — (never run at this volume alone) |
| `raw24k` | 24.6M | 0.4743 / 0.6870 / 0.6236 |
| **`std96k`** (best) | 98.3M | **0.5532 / 0.7252 / 0.6534** |
| *LeWM teacher (reference)* | 6.1M | *0.484 / 0.691 / 0.923* |

Two levers work and are additive: **per-dim standardization** (+0.03 spearman at
every volume) and **sample volume** (+0.05 from 24.6M→98.3M, unsaturated). The
best DINO critic now BEATS the LeWM teacher on both unconfounded measures — and
it changed TD+CEM 76.0 → 76.0. But see §1: that comparison sits inside the noise
band of this eval, so it is NOT evidence that critic quality does not matter.

`monotone` is CONFOUNDED — do not use it. TD on *ground-truth cube xyz* scores
only 0.4907 on it, because the cube is static during reach and a flat distance
counts as non-decreasing. Use spearman / pair_acc.

## 3. Conclusions (floor-corrected)

**Holds:** on frozen-DINOv2 token dynamics, plain latent+CEM is the strongest
planner (+25.3 over floor). TD+CEM recovers ~84 % of that, LIPv4 only ~37-50 %. So
adding a learned value metric does not beat planning directly in token-grid MSE,
and the more aggressively a planner optimizes the learned head the worse it does
(none -> 30 CEM iterations -> explicit gradient descent). Supporting mechanism:
LIP drives its predicted cost **5.6x down (27.7 -> 4.97) with no success gain**.
Latent MSE is anchored at the goal embedding and cannot be gamed; a learned
scalar has minima that optimizers find.

**Does NOT hold / unresolved at this n:** any comparison *between* LIP variants,
and the claim that critic quality is irrelevant to planning. The critic upgrade
(spearman 0.474 -> 0.553, beating the LeWM teacher) moved TD+CEM 76.0 -> 76.0,
but with SE ~6 per draw on a 25-point usable range that measurement cannot
distinguish "no effect" from "small effect". Re-test on the harder eval (§9.1)
before drawing any conclusion about the critic.

**Solid regardless of the floor:** plain latent+CEM = 80.0 on this upstream
checkpoint versus 38-52 for our own from-scratch retrain of the same
architecture. That settles the older open question: the retrain ceiling was a
**training** failure, not a harness or convention problem. (Note the floor
applies to that comparison too -- 38-52 is at or BELOW the 54.7 no-op floor,
which makes the retrain failure even starker: it planned worse than doing
nothing.)

## 4. Three real bugs found (all in the DINO LIP path, all mine)

1. **Action convention.** The predictor was trained on **z-scored** action
   blocks; `rollout_terminal_dino` / `rollout_traj_dino` were fed RAW actions
   (the original `train_lip_dino.py` docstring asserts raw, and my ports
   inherited it). Measured rollout error vs the true next latent:

   | h | raw | normalized | zero-action |
   |---|---|---|---|
   | 1 | 0.1117 | **0.0604** | 0.1718 |
   | 5 | 0.1859 | **0.0678** | 0.2720 |

   At raw scale the WM was barely better than being fed ZERO actions. Fixed by
   `fix_action_convention.py`. **Invalidated every pre-fix LIP number**; those
   artefacts are archived under `/workspace/actors/invalid_pre_actionfix/`.
2. **Double-scaled returns.** `_proposal_lip_dino` returned raw actions, but
   `policy.py:303` applies `process['action'].inverse_transform` to the solver
   output. LeWM's known-good `_proposal_lip` returns `A` un-scaled. Fixed.
3. **GPU-0 contention.** Running training on GPU 0 while the EGL eval daemon used
   it core-dumped three evals (`timeout: the monitored command dumped core`).
   **GPU 0 must be eval-only; EGL evals must never overlap each other.**

## 5. Hypotheses falsified by measurement — do not re-run these

1. reduction is second-order → WRONG (full tokens beat pooling, monotone +0.13)
2. frozen DINOv2 geometry cannot support cost-to-go → unsupported (no
   train/held-out gap: 0.7993 vs 0.8014)
3. head capacity is the bottleneck → WRONG (7-cell null: width 256→2048,
   embed 128→512, depth 2→3, all ±0.01)
4. the held-out split explains the gap to the published 86 → WRONG (all-10k
   scored 78.0, *below* held-out 82.0)
5. the TD bootstrap signal is the bottleneck → WRONG (fully supervised temporal
   regression was no better: 0.616 vs 0.624 monotone, worse on ranking)
6. low effective rank means the extra dims are noise → WRONG, and cleanly:
   PCA-whitening is monotonically worse the more you compress
   (k=16 0.379, k=24 0.399, k=32 0.417, k=64 0.437, raw-384 **0.474**).
   **Every principal component carries signal** — which is why the large sample
   requirement is real rather than wasteful.
7. the rollout carries no signal by h=5 → WRONG, that was bug #1 compounding

Also swept flat, do not revisit: expectile (0.01–0.5), lr, tau, n_step
(25/50/100), p_cross, weight_decay, huber_beta, critic-ratio. `head=mlp`
actively **inverts** the metric (spearman −0.29), so the quasimetric prior is
load-bearing.

## 6. Open items — NOTHING IS RUNNING as of 19:20 UTC 2026-08-03

| item | cost |
|---|---|
| `tdcem96_e43` never ran (I killed the queue so the floor could jump ahead) | ~36 min |
| 30-step-actor control skipped — `/tmp/fix_smoke.pt` was lost in a container recycle. Re-train 30 steps then eval ×3; if it matches the 3000-step actors, actor training is irrelevant at this floor | ~15 min |
| `fx_{ctrl,a12,a20}_s3000` (pre-critic-upgrade LIP) still unscored | ~25 min |
| `/workspace/gpu0_all.sh` is the idempotent driver for all of the above | re-run it |

## 7. Code built

**Every script listed here is committed to
[`scripts_lip_critic/`](scripts_lip_critic/) in this folder** (37 files) — the
session scratchpad they were written in is gone. On the pod the generated ones
live under `/workspace/code/stable-worldmodel/scripts/plan/`; the `make_*.py`
patchers regenerate them from the shared originals, which were never edited.

| file | purpose |
|---|---|
| `scripts/plan/eval_wm_dino.py` | `eval_wm.py` + two fixes: honours `interpolate_pos_encoding=False` (HF Dinov2 rejects the kwarg), and a metric hook that handles patch-token latents, auto-detecting flatten-vs-pool from the head width |
| `scripts/plan/train_lip_ac_dino.py` | the real LIPv4 recipe (PlannerNetV4, co-trained critic, canonical schedules) on token rollouts; lance-backed pixels; `--replay-prob` ported to token space |
| `scripts/plan/train_lip_dino_lance.py` | v1 dino LIP, lance-backed (superseded) |
| `solver/lip.py` | added `rollout_traj_dino` (+`return_toks`), `kind="lip4_dino"`, action-convention fix |
| `critic_sweep.py`, `critic_whiten.py` | §2 and falsification 6, with affine folding |
| `exp_action_convention.py` | found bug #1 |
| `exp_conditioning.py`, `exp_ceiling.py`, `exp_offmanifold.py`, `probe_fit_vs_gen.py`, `td_vs_regression.py` | diagnostics behind §5 |
| `verify_env.py` | version pins + a strict-load check of the DINO-WM and both critics (§8) |
| `measure_floor.sh` | **the floor control in §1** (complete; its 30-step-actor arm still skipped) |
| `bootstrap2.sh` | container rebuild, ~4 min (see §8) |

**Affine folding trick** (reused for standardization and whitening): train the
head on transformed latents, then fold the transform into its single entry
`Linear` — `W' = W·T`, `b' = b − W'·μ`. The saved metric then accepts RAW latents,
so the CEM hook, LIP and every probe work unchanged. Verified numerically per
cell (max|diff| ≈ 1e-5). `QuasimetricHead` makes this clean: both `z_i` and `z_j`
pass through the same `self.enc`.

## 8. Ops gotchas that cost real time

**Environment verified clean 2026-08-03** (`verify_env.py`): torch 2.4.1+cu124,
torchvision 0.19.1+cu124, transformers 4.49.0, numpy 1.26.4, **cuDNN 90100**,
lance 8.0.0, mujoco 3.10.0, spt 0.1.7, no stray `nvidia-*-cu13` wheels. The
DINO-WM loads with **0 missing / 0 unexpected / 0 shape mismatches** and
`_vit_layout_compat` performs **no** remapping — so the reconstructed
`config.json` is exactly right and nothing depends on the transformers version
for this checkpoint. The environment is not a confounder for any number here.

* **Container recycles** wipe the pip env and `/root` (including the 20 GB
  dataset) while `/workspace` survives. Re-run `/root/bootstrap2.sh` (~4 min +
  3 min dataset). Install the swm `env` extras BEFORE re-pinning
  `"numpy<2" torch==2.4.1 torchvision==0.19.1 --index-url .../cu124`, then purge
  leftover `nvidia-*-cu13` wheels — otherwise cuDNN is shadowed and every conv
  dies. `torch.backends.cudnn.version()` must read **90100**.
* **`pkill -f <pattern>` matches the invoking ssh shell** if the pattern appears
  in its cmdline. This killed my session three times and produced one false
  "already alive". Use PID files.
* **Never inline a heredoc containing prose over ssh** — a single apostrophe
  (e.g. "LeWM's") terminates the quoted argument. Write scripts locally, `scp`,
  then run. This bit me five times.
* **Always queue the work behind a bootstrap.** I once repaired the env and left
  nothing running: **9.5 h of four idle H200s.**
* Validate config keys against the dataclass BEFORE a long cache load:
  `TDConfig` has 19 fields and does **not** include `rank_weight`,
  `eikonal_weight`, `num_components` or `td_weight` (those are other learners'
  CLI args). Two separate crashes came from assuming otherwise.
* LIP training is ~6 s/step at batch 32 / iters 8 (3000 steps ≈ 5 h). Each
  trainer holds ~127 GB of a 143 GB card → **one trainer per GPU**. LIP evals are
  ~2.5 min; CEM evals ~36 min.

## 9. Next steps

1. **Build a harder eval FIRST — this gates everything else.** With 54.7 % free
   from `nomove`, the h25 protocol often starts the cube near its goal. Filter
   tasks by required cube displacement: `task_stats.py` (this folder) already
   computes `disp = ||cube(goal) - cube(start)||` and `lift` from
   `privileged_block_0_pos`; on h25 draws `disp` spans 0.08-0.32 m. Keep only
   tasks above ~0.15 m (and/or `lift > 0.1` for genuine pick-and-place), then
   re-measure `nomove` to confirm the floor dropped. Target: floor 10-20, which
   widens the resolvable range 2-3x. THEN re-run plain CEM / TD+CEM / LIP and the
   critic comparison, which are currently all inside the noise band.
2. Only after (1): re-test the `std96k` critic and the expand/replay cells. They
   may well show real effects once the range is wide enough to see them.
3. The remaining untested lever is a **learned projection trained with a
   temporal / predictive objective** on the cached frozen features -- giving
   DINOv2 the property LeWM and PLDM get by construction (both end in
   `MLP(192->2048->192, norm_fn=BatchNorm1d)`).
4. Dyna stage 1 is **collected and verified** but stage 2 was never built:
   `/workspace/dyna_dino/onpolicy_lipdino.lance`, 600 episodes, 30 000 rows,
   exactly 50 rows/episode, 40.5 % success rows, split filter confirmed per call.
   Note `success_rate: 0.0` in those collection logs is an artifact of
   `terminate_at_goal=False` (the eval derives success from `world.terminateds`,
   which never fires) -- the recorder `success` column is the valid signal. Also
   note DINOv2 is frozen, so a Dyna fine-tune can only move the predictor and
   action embedder (~10 % of parameters). Collected with a LIP actor that is now
   known to be only ~45 % as good as CEM, so consider re-collecting with CEM if
   collection cost allows (CEM is ~36 min per 50 episodes vs LIP ~2.5 min).
