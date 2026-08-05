# Patched stable-worldmodel files (snapshot)

**Why this exists:** `/workspace/code/stable-worldmodel` on the pods is **not a git
repo**, so every change below lived only as loose files on machines that have already
died twice. This is a verbatim snapshot, taken 2026-07-26 from pod
`157.66.254.11:15788`, of every file the Dyna/RLP campaign modified.

Two ways to restore onto a fresh pod:
1. **Replay the patch scripts** (preferred — they are anchored and idempotent, and they
   fail loudly if upstream moved): run the `patch_*.py` files in `../dyna_harness/` in
   the order listed below.
2. **Copy these files over** the stock tree (faster, but blind to upstream drift).

## What changed, and why

| file | change | added by |
|---|---|---|
| `stable_worldmodel/world/world.py` | `SWM_RECORD_PATH` env-gated recorder in `_evaluate_from_dataset` — dumps (pixels, action, qpos, qvel) per step to a lance dataset. Needed because the Dyna collector must roll out the LIP actor through the **eval** path (which sets goals in `infos`); `World.collect` resets without goals, so the actor cannot plan. Also the only clean source of agent video (the harness video panel renders `infos['pixels']` as noise under `ob_type: states`). | `patch_world_record.py` |
| `stable_worldmodel/trm/head.py` | `IQEHead` — Interval Quasimetric Embedding (Wang & Isola). Interval-union measure via sort + exclusive cummax, `maxmean` aggregation with learnable α. Unit-tested: matches brute-force union to 2.4e-7, `d(x,x)=0`, 0% triangle violations. | `patch_iqe_eikonal.py` |
| `stable_worldmodel/trm/io.py` | `build_metric` handles `head='iqe'` and `learner='qrl'` so `load_metric` can rebuild them. | `patch_iqe_eikonal.py`, `patch_qrl_wiring.py` |
| `stable_worldmodel/trm/learners/td.py` | (a) Eikonal unit-gradient penalty `(‖∇_z V‖·ds − 1)²`, `ds` = measured mean per-step latent displacement so the target means "V changes ~1 per **env step**"; (b) planner-relevant pairwise ranking hinge (one goal, two states at different true distances); (c) `td_weight` so the distance-regression term can be switched off. | `patch_iqe_eikonal.py`, `patch_ranking_loss.py`, `patch_td_weight.py` |
| `stable_worldmodel/trm/learners/qrl.py` | **new** — QRL (Wang et al., ICML 2023): maximise a bounded spreading transform of `d(s,g)` on random pairs subject to `E[relu(d(s,s′)−c)²] ≤ ε²` on observed transitions, via a learned Lagrange multiplier. Never regresses a far-pair label; long distances are derived by the triangle inequality. | `patch_qrl_wiring.py` |
| `stable_worldmodel/trm/learners/__init__.py` | register `qrl`. | `patch_qrl_wiring.py` |
| `scripts/plan/train_metric.py` | CLI: `--head iqe`, `--eikonal-weight`, `--num-components`, `--rank-weight`, `--rank-margin`, `--td-weight`, `--learner qrl`, `--qrl-*`. | all of the above |
| `scripts/plan/train_lip_ac.py` | (a) **`blocks()` episode-end clamp** — without it, action blocks bleed across episode boundaries and crash (ragged stack) on the final episode; (b) `--critic-rank-weight` / `--critic-td-weight` so the tandem critic can carry the ranking objective instead of TD (otherwise tandem TD-training erodes any ranking-trained init). | `patch_blocks_clamp.py`, `patch_lipac_rank.py` |
| `scripts/train/lewm.py` | contains the multi-step rollout loss. **Inert and deliberately unused** — the rollout loss was removed by directive and is empirically justified (its WM measured latent+CEM 44 vs a 68.7 reference). Do not pass `wm.rollout_len` / `wm.rollout_weight`. | (pre-existing) |

Also relevant, not in this directory: `patch_init_weights.py` adds `INIT_WEIGHTS` warm-start
to the generated `scripts/train/lewm_expert.py` (used for the Dyna WM fine-tune); it is
applied by `patch_lewm_expert.py` on the pod at launch time.

## Recommended replay order

```
patch_blocks_clamp.py      # correctness fix — apply first
patch_world_record.py      # Dyna collector
patch_iqe_eikonal.py       # IQE head + eikonal
patch_ranking_loss.py      # ranking hinge
patch_td_weight.py         # --td-weight  (then patch_td_weight2.py for train_metric wiring)
patch_qrl_wiring.py        # installs qrl_learner.py as trm/learners/qrl.py
patch_lipac_rank.py        # tandem-critic ranking
patch_init_weights.py      # WM fine-tune warm-start
```

Every patch asserts on its anchors, so a silent mismatch after an upstream change is not
possible — it will raise instead.
