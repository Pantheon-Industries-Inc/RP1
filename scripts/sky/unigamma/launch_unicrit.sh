#!/usr/bin/env bash
# UNIFIED CRITIC on top of config B -- one critic-training recipe for all
# seven cells (TwoRoom/Cube/Reacher x LeWM/PLDM, PushT/LeWM). Actor side is
# config B unchanged (amax 2.5, md 20 / pc 0.3 except PushT's md 6 / pc 0.1,
# ac 0.5, K 8, ES on 48-51, per-env budgets/anchors/replay).
#
# The critic recipe (= PushT's W1NEAR_FRZ teacher, made the rule everywhere):
#   * offline TD teacher only, deployed verbatim: freeze_critic_frac 0, so the
#     co-critic never runs and every planner.* TD knob is inert (see
#     BEST_CONFIGS_20260914.md section 7a)
#   * teacher steps = ACTOR steps (ratio 1.0; PushT's ladder 3k -7.4 / 6k best /
#     12k / 24k -7.4 is the only measured curve): tw 8000, cube 6000,
#     reacher 6000, pusht 6000. Batch 1024 everywhere.
#   * n-step 1, gamma 0.98, expectile 0.03 (tworoom was 0.1), depth 2,
#     p_cross 0.3, max_delta = full episode, near-goal oversampling 0.3 of
#     in-episode goals at 1..3 primitive steps (fs1 cache everywhere),
#     one teacher per training seed (cube/reacher shared seed 0 before).
#   * window frames stay the environment property (1 / 1 / 2 / 1).
#
# Prerequisite: GIT_TOKEN secret refreshed (tworoom/reacher yamls clone
# Value_Metric_LeWM@eval-sweep; every such job failed on it 2026-09-09 and
# 2026-09-20). Run `sky workspace use armin` first.
#
#   bash scripts/sky/unigamma/launch_unicrit.sh            # all 7 cells, seeds 0-2
#   CELLS="cube-pldm pusht" SEEDS="0 1 2 3 4 5" bash ...   # subset
#   DRY=1 bash ...                                          # print only
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.sky/bin:$PATH"
DATE=${DATE:-20260920}
SEEDS=${SEEDS:-"0 1 2"}
PRIO=${PRIO:-p1}
USERV=armin@pantheon.inc
CELLS=${CELLS:-"tworoom-lejepa tworoom-pldm cube-lewm cube-pldm reacher-lejepa reacher-pldm pusht"}
COMMON=(--env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV --env MAXPAR=4)
run() { if [ "${DRY:-0}" = 1 ]; then printf '%q ' "$@"; echo; else "$@" 2>&1 | grep -Ev "^warning: the SkyPilot|curl -fsSL" | tail -1; fi; }

for CELL in $CELLS; do
  ENV=${CELL%%-*}; BASE=${CELL#*-}; B1=$(echo $BASE | cut -c1)
  case $ENV in
  tworoom)
    NAME="rlp-tw-unicrit-${B1}-${DATE}"
    run sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO -y --async \
      --env ENVNAME=tworoom --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
      --env SPLIT=1 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
      --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
      --env TR_GAMMA=0.98 --env TR_NSTEP=1 --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 \
      --env ACR_LAM=0.5 --env TW_REPLAY=0 \
      --env UNIG_SCHED=frz --env TD_STEPS=8000 --env TW_TD_EXPECTILE=0.03 --env UNIG_PC=0.3 \
      --env NEAR_FRAC=0.3 --env NEAR_MAX=3 --env TD_PER_SEED=1 \
      --env STEPS=8000 --env BATCH=128 --env RH=5 \
      --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
      --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
      --env TRAIN_SEEDS="$SEEDS" --env CACHE_VERSION="tw-unicrit-${B1}-v1" --env EXPERIMENT_TAG="$NAME" "${COMMON[@]}"
    ;;
  cube)
    NAME="rlp-cu-unicrit-${B1}-${DATE}"
    run sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO -y --async \
      --env ENVNAME=cube --env BASE=$BASE --env GRID=unig --env ONLYCFG=unig_ctrl_a2.5 \
      --env SPLIT=0 --env SMOKE=0 --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
      --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
      --env CU_DYNA=0 --env CU_DYNA_WM="" --env CU_EXPAND=1.0 --env CU_REPLAY=0.5 --env CU_GAMMA=0.98 --env TR_NSTEP=1 \
      --env CRIT_DEPTH=2 --env UNIG_LAYERS=2 --env UNIG_MD=20 --env UNIG_K=8 --env ACR_LAM=0.5 \
      --env UNIG_SCHED=frz --env TD_STEPS=6000 --env CU_TD_EXPECTILE=0.03 --env UNIG_PC=0.3 \
      --env NEAR_FRAC=0.3 --env NEAR_MAX=3 --env TD_PER_SEED=1 \
      --env STEPS=6000 --env BATCH=256 --env RH=5 \
      --env CKPT_SELECT=1 --env CKPT_EVERY=2000 --env CKPT_VAL_SEEDS="48 49 50 51" \
      --env EVAL_SEEDS="42 43 44" --env UNIG_OFF="25,100" --env EVAL_RAWDIR=volume \
      --env TRAIN_SEEDS="$SEEDS" --env CACHE_VERSION="cu-unicrit-${B1}-v1" --env EXPERIMENT_TAG="$NAME" "${COMMON[@]}"
    ;;
  reacher)
    NAME="rlp-re-unicrit-${B1}-${DATE}"
    run sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml -n "$NAME" --priority $PRIO -y --async \
      --env BASE=$BASE --env GRID=cross --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
      --env SMOKE=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 --env ACTOR_IMPORT_TAG="" \
      --env RS_GAMMA=0.98 --env RS_NSTEP=1 --env CRIT_DEPTH=2 --env RS_LAYERS=2 --env RS_MD=20 --env RS_K=8 \
      --env ACR_LAM=0.5 --env VALUE_FRAMES=2 --env VALUE_EXPECTILE=0.03 \
      --env RS_FREEZE=0 --env RS_TD_STEPS=6000 --env RS_TD_PC=0.3 --env RS_TD_NEAR_FRAC=0.3 --env RS_TD_NEAR_MAX=3 --env RS_TD_PER_SEED=1 \
      --env STEPS=6000 --env BATCH=256 --env CKPT_EVERY=2000 --env RS_CKPT_SELECT=1 --env RS_CKPT_VAL_SEEDS="48 49 50 51" \
      --env RS_LATCHED=1 --env FINAL_ONLY=1 --env HELD_AT_END=0 --env REPORT_DRAWS="42 43 44" \
      --env SUCCESS_THRESHOLDS="0.1 0.05" --env PLANNER_RH=5 \
      --env TRAIN_SEEDS="$SEEDS" --env EXPERIMENT_TAG="$NAME" "${COMMON[@]}"
    ;;
  pusht)
    # PushT already IS this critic recipe (W1NEAR_FRZ + value.steps=6000 = actor
    # steps); the actor side is the 78.7 es1k recipe. Re-run only for a same-date tag.
    for S in $SEEDS; do
      DRY=${DRY:-0} DATE=$DATE REPLAY=0 MAX_DELTA_OVERRIDE=6 \
        EXTRA_OVR="planner.p_cross=0.1 value.steps=6000 planner.ckpt_every=1000" TAGSUF=-unicrit \
        scripts/sky/launch_pusht_critic_arms.sh W1NEAR_FRZ $S
    done
    ;;
  *) echo "unknown cell $CELL" >&2; exit 1;;
  esac
  echo "-> $CELL"
done
echo "Collect: python3 scripts/wandb_collect_rlp.py 'unicrit' --reacher"
