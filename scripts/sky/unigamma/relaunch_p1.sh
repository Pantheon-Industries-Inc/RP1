#!/usr/bin/env bash
# One-off: relaunch the 13 jobs cancelled from the p2 queue at p1.
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=20260818
EVAL_SEEDS="42 43 44"
launch_rlp() {
  local NAME=$1 BASE=$2 GAMMA=$3 NSTEP=$4 SEEDS=$5 ONLY=$6
  echo "==> $NAME"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE="$BASE" --env GRID=g98matrix \
    --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="$ONLY" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="$GAMMA" --env TR_NSTEP="$NSTEP" \
    --env CACHE_VERSION="tw-${NAME}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="$EVAL_SEEDS" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
}
launch_base() {
  local NAME=$1 BASE=$2 GAMMA=$3 NSTEP=$4
  echo "==> $NAME"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE="$BASE" --env GRID=measurements \
    --env PLANNER_COSTS=value --env PLANNERS="cem adam mppi" \
    --env PRE_MEASUREMENT_SMOKE=1 \
    --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="$GAMMA" --env TR_NSTEP="$NSTEP" \
    --env CACHE_VERSION="tw-${NAME}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="0" --env EVAL_SEEDS="$EVAL_SEEDS" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
}
launch_rlp "rlp-tw-g98m-p-n200-s1" pldm 0.98  200 "1" ""
launch_rlp "rlp-tw-g98m-p-n200-s2" pldm 0.98  200 "2" ""
for S in 0 1 2; do launch_rlp "rlp-tw-g98m-p-g996-s${S}" pldm 0.996 50 "$S" ""; done
launch_rlp "rlp-tw-g98m-p-g1ref" pldm 1.0 50 "0 1 2" ""
launch_rlp "rlp-tw-g98m-l-g996-h25" lejepa 0.996 50 "0 1 2" "g98m_h25"
launch_base "rlp-tw-g98m-b-l-n100" lejepa 0.98  100
launch_base "rlp-tw-g98m-b-l-n200" lejepa 0.98  200
launch_base "rlp-tw-g98m-b-l-g996" lejepa 0.996 50
launch_base "rlp-tw-g98m-b-p-n100" pldm   0.98  100
launch_base "rlp-tw-g98m-b-p-n200" pldm   0.98  200
launch_base "rlp-tw-g98m-b-p-g996" pldm   0.996 50
echo "13 jobs relaunched at p1."
