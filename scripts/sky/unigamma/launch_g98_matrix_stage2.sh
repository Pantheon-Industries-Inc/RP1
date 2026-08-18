#!/usr/bin/env bash
# Stage 2 of the g98-matrix sweep: the two LeJEPA h25 jobs held back to stay
# inside the 80-GPU budget while jobs 8022-8024 run. Launch when those free.
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=20260818
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
    --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="42 43 44" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc \
    2>&1 | tail -1
}
launch_rlp "rlp-tw-g98m-l-n100-h25" lejepa 0.98 100 "0 1 2" "g98m_h25"
launch_rlp "rlp-tw-g98m-l-n200-h25" lejepa 0.98 200 "0 1 2" "g98m_h25"
