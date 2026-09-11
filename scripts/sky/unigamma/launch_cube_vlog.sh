#!/usr/bin/env bash
# Cube h100 vlog probe on the banked recipe -- one job per (base x seed).
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=20260818
launch() {   # launch <base> <seed>
  local BASE=$1 SEED=$2
  local NAME="rlp-cu-vlog-${BASE}-s${SEED}"
  echo "==> $NAME"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=cube --env BASE="$BASE" --env GRID=cube_vlog \
    --env SPLIT=0 --env SMOKE=0 --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA=1.0 --env TR_NSTEP=50 \
    --env CACHE_VERSION="cu-vlog-${BASE}-s${SEED}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="$SEED" --env EVAL_SEEDS="42 43 44" \
    --env STEPS=6000 --env BATCH=256 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=10 --env OFFSETS="100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
}
for S in 0 1 2; do launch lewm "$S"; done
for S in 0 1 2; do launch pldm "$S"; done
echo "6 cube-vlog jobs submitted (24 GPUs)."
