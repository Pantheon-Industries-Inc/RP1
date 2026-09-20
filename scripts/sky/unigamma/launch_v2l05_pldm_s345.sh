#!/usr/bin/env bash
# Config B (v2-lambda0.5) PLDM cells, training seeds 3-5 -- brings the three
# PLDM cells from n=3 to n=6 (LeWM cells already have s345 from 2026-09-02).
#
# Recipe reconstructed 2026-09-20 from the W&B configs of the 2026-09-02 jobs
# (rlp-tw-v2l05-p 17876, rlp-cu-v2l05-p 17874, rlp-re-v2l05-p 17875; the
# controller logs were cleaned 2026-09-10) and the config-B Cube launcher in
# scripts/sky/dyna2/run_dyna2.sh. Shared: depth 2 / 1x budgets, n-step 1,
# ac_weight 0.5, gamma 0.98, amax 2.5, max_delta 20, K 8, within-run early
# stopping over {step2000, step4000, final} on selection draws 48-51, report
# draws 42/43/44, held-out eval 8000:10000. Per-env anchors as banked:
#   tworoom-pldm  steps 8000 / batch 128 / TD 6000 / mw 0.0 lr 1e-3 / replay 0 / expand 0 / freeze 0.8
#   cube-pldm     steps 6000 / batch 256 / TD 12000 / mw 0.1 lr 3e-4 / replay 0.5 / expand 1 / freeze 0.5 / bf16 eval
#   reacher-pldm  steps 6000 / batch 256 / w=2 window teacher 12k / mw 0.5 lr 3e-4 / replay 0.5 / expand 0 / freeze 0.5 / latched tau .1/.05
#
#   bash scripts/sky/unigamma/launch_v2l05_pldm_s345.sh
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=${DATE:-20260920}
SEEDS=${SEEDS:-"3 4 5"}
PRIO=${PRIO:-p1}
USERV=armin@pantheon.inc
COMMON=(--env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV
        --env TRAIN_SEEDS="$SEEDS" --env MAXPAR=4)

echo "==> tworoom / pldm"
sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "rlp-tw-v2l05-p-s345" --priority $PRIO -y --async \
  --env ENVNAME=tworoom --env BASE=pldm --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
  --env SPLIT=1 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
  --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
  --env TR_GAMMA=0.98 --env TR_NSTEP=1 --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 \
  --env ACR_LAM=0.5 --env UNIG_SCHED=tw --env TW_REPLAY=0 \
  --env STEPS=8000 --env BATCH=128 --env TD_STEPS=6000 --env RH=5 \
  --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
  --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
  --env CACHE_VERSION="tw-v2-p-n1s2-v1" --env EXPERIMENT_TAG="rlp-tw-v2l05-p-s345-${DATE}" \
  "${COMMON[@]}" 2>&1 | tail -1

echo "==> cube / pldm"
sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "rlp-cu-v2l05-p-s345" --priority $PRIO -y --async \
  --env ENVNAME=cube --env BASE=pldm --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
  --env SPLIT=0 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
  --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
  --env CU_DYNA=0 --env CU_DYNA_WM="" --env CU_EXPAND=1.0 --env CU_GAMMA=0.98 --env TR_NSTEP=1 \
  --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 --env ACR_LAM=0.5 \
  --env STEPS=6000 --env BATCH=256 --env TD_STEPS=12000 --env RH=5 \
  --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
  --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
  --env CACHE_VERSION="cu-v2-p-n1s2-v1" --env EXPERIMENT_TAG="rlp-cu-v2l05-p-s345-${DATE}" \
  "${COMMON[@]}" 2>&1 | tail -1

echo "==> reacher / pldm"
sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml -n "rlp-re-v2l05-p-s345" --priority $PRIO -y --async \
  --env BASE=pldm --env GRID=cross --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
  --env SMOKE=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 --env ACTOR_IMPORT_TAG="" \
  --env RS_GAMMA=0.98 --env RS_NSTEP=1 --env CRIT_DEPTH=2 --env RS_LAYERS=2 --env RS_MD=20 --env RS_K=8 \
  --env ACR_LAM=0.5 --env VALUE_FRAMES=2 --env VALUE_EXPECTILE=0.03 \
  --env STEPS=6000 --env BATCH=256 --env CKPT_EVERY=2000 --env RS_CKPT_SELECT=1 --env RS_CKPT_VAL_SEEDS="48 49 50 51" \
  --env RS_LATCHED=1 --env FINAL_ONLY=1 --env HELD_AT_END=0 --env REPORT_DRAWS="42 43 44" \
  --env SUCCESS_THRESHOLDS="0.1 0.05" --env PLANNER_RH=5 \
  --env EXPERIMENT_TAG="rlp-re-v2l05-p-s345-${DATE}" \
  "${COMMON[@]}" 2>&1 | tail -1

echo "3 PLDM s345 jobs submitted (3 x H200:4). Collect with:"
echo "  sky jobs launch scripts/sky/unigamma/collect_unig.yaml -n unig-collect-v2l05p -y --async --env TAG_GLOB='rlp-*-v2l05-p*'"
