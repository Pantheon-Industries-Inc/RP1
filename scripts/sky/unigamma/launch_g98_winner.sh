#!/usr/bin/env bash
# Winner-hypers h100 arms under the rescue critics + the g996xN100 consistency
# cell. Stage A (now, 24 GPUs): g996-winner sharded per (base x seed).
# Stage B (chained on capacity): n100-winner packed + g996n100-winner packed.
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=20260818
launch() {   # launch <name> <base> <gamma> <nstep> <seeds>
  local NAME=$1 BASE=$2 GAMMA=$3 NSTEP=$4 SEEDS=$5
  echo "==> $NAME"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE="$BASE" --env GRID=g98winner \
    --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="$GAMMA" --env TR_NSTEP="$NSTEP" \
    --env CACHE_VERSION="tw-${NAME}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="42 43 44" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
}
STAGE=${STAGE:-A}
if [ "$STAGE" = A ]; then
  # per-block discount at winner hypers, sharded (the front-runner arm)
  for S in 0 1 2; do launch "rlp-tw-g98w-l-g996-s${S}" lejepa 0.996 50 "$S"; done
  for S in 0 1 2; do launch "rlp-tw-g98w-p-g996-s${S}" pldm   0.996 50 "$S"; done
else
  # per-step 0.98 with horizon-covering backup, packed; and the untested
  # consistency cell gamma-per-block x n-step 100, packed
  launch "rlp-tw-g98w-l-n100"     lejepa 0.98  100 "0 1 2"
  launch "rlp-tw-g98w-p-n100"     pldm   0.98  100 "0 1 2"
  launch "rlp-tw-g98w-l-g996n100" lejepa 0.996 100 "0 1 2"
  launch "rlp-tw-g98w-p-g996n100" pldm   0.996 100 "0 1 2"
fi
