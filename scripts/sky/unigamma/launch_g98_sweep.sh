#!/usr/bin/env bash
# RLP planner-hyper sweep at h100 under the per-block critic.
# SELECTION stage: EVAL_SEEDS="50 51", sharded per (base x seed).
# REPORT stage:    EVAL_SEEDS="42 43 44", ONLYCFG=<winner>, packed 3 seeds.
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=20260818
STAGE=${STAGE:-select}
launch() {   # launch <name> <base> <seeds> <evalseeds> <onlycfg>
  local NAME=$1 BASE=$2 SEEDS=$3 EVS=$4 ONLY=$5
  echo "==> $NAME"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE="$BASE" --env GRID=g98sweep \
    --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="$ONLY" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="${SWEEP_GAMMA:-0.996}" --env TR_NSTEP="${SWEEP_NSTEP:-50}" --env SWEEP_VN="${SWEEP_VN:-}" \
    --env CACHE_VERSION="tw-${NAME}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="$EVS" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
  sleep 45   # stagger clones: five simultaneous LFS pulls rate-limited earlier
}
if [ "$STAGE" = select ]; then
  for S in 0 1 2; do launch "rlp-tw-${SWEEP_TAG:-g98s}-l-s${S}" lejepa "$S" "50 51" ""; done
  for S in 0 1 2; do launch "rlp-tw-${SWEEP_TAG:-g98s}-p-s${S}" pldm   "$S" "50 51" ""; done
else
  # STAGE=report WINNER_L=<cfg> WINNER_P=<cfg>
  launch "rlp-tw-${SWEEP_TAG:-g98s}-l-report" lejepa "0 1 2" "42 43 44" "${WINNER_L:?}"
  launch "rlp-tw-${SWEEP_TAG:-g98s}-p-report" pldm   "0 1 2" "42 43 44" "${WINNER_P:?}"
fi
