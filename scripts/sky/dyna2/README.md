# Dyna v2 (cube) -- 2026-09-03

Why: `docs/campaigns/2026-09-03/DYNA_GAP_DIAGNOSIS.md`. The shipped Dyna WMs
were one iteration on the 2026-08-05 open-loop planner's h25 rollouts, no
anchor, LeWM objective for both bases; they cure 6/9 h25 core tasks and 0/17
at h100 (net negative there).

Pipeline per base (`run_dyna2.sh <base> [iters]`):
1. `dyna2_collect_ft.yaml` -- collect with the deployed config-B actors
   (`rlp-cu-v2l05*-20260902` for iteration 1, the previous iteration's actors
   after) at offsets 25 AND 100 on episodes 0:8000 (`terminate_at_goal=False`,
   outcome-labelled), 50/50 mix vs the expert 0:8000 slice, anchored
   fine-tune (`overlays/wm_finetune.py`: latent anchor to the init weights,
   ANCHOR_WEIGHT=1.0; native PLDM objective for PLDM; 1 epoch, lr 1e-5,
   expert action pin). Export: `/checkpoints/<user>/<tag>/wm/{weights.pt,config.json}`.
2. `tworoom_g98_rescue.yaml` with `CU_DYNA_WM=<that dir>` -- fresh caches,
   TD teacher, three config-B actors with fine ES, held-out eval 8000:10000 at
   h25+h100 with per-task arrays (`EVAL_RAWDIR=volume`).
Directive 2026-09-04: NO deploy-time restarts -- the planner runs its K=8
refinement once, always. `select_seed.yaml` is diagnostic only (measures the
seed-lottery cost); reported numbers stay median-of-seeds.

Tags: `rlp-cu-dyna2-<base>-it<k>-<date>` (collect+ft), `rlp-cu-dyna2t-...`
(train+eval). Smoke: `SMOKE=1` on the
collect job (1 actor, 1 call, 400-episode lance, 1 epoch).
