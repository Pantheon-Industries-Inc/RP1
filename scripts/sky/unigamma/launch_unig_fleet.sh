#!/usr/bin/env bash
# Unified-gamma cross-environment fleet — 2026-08-18.
#
# The experiment: gamma dose-response {0.98, 0.99, 1.0} x vnorm {none, log}
# per environment, n-step 50, every cell trained ONCE at the unified clip
# amax=2.5 / max_delta=12 (user decision: "train everything at amax=2.5",
# adapt amax only at deployment). One actor per (gamma x vnorm x seed),
# evaluated at h25+h100 (TwoRoom, Cube) or tau 0.1/0.05 latched (Reacher).
# gamma=0.99 is the revised unified proposal (gamma ~ 1 - 1.5/d_max, ceiling
# 100 steps); 0.98 is the cube/reacher bespoke + original unified proposal
# (falsified on TwoRoom h100 at a1.8); 1.0 is the TwoRoom bespoke.
#
# Footprint: fleet = 9 tworoom + 3 cube + 6 reacher jobs x H200:4 = 72 GPUs.
#
#   STAGE=smoke bash scripts/sky/unigamma/launch_unig_fleet.sh   # 3 canaries
#   STAGE=fleet bash scripts/sky/unigamma/launch_unig_fleet.sh   # 18 jobs
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
DATE=20260818
STAGE=${STAGE:-smoke}

launch_tw() {  # launch_tw <gtag> <gamma> <seeds> <smoke> [evalseeds]
  local GT=$1 GAMMA=$2 SEEDS=$3 SMK=$4 EVS=${5:-"42 43 44"}
  local NAME="rlp-tw-unig-${GT}-s${SEEDS// /}"
  [ "$SMK" = 1 ] && NAME="rlp-tw-unig-smoke"
  echo "==> $NAME (tworoom lejepa gamma=$GAMMA seeds='$SEEDS' smoke=$SMK)"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE=lejepa --env GRID=unig \
    --env SPLIT=1 --env SMOKE="$SMK" --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="$GAMMA" --env TR_NSTEP=50 \
    --env CACHE_VERSION="tw-unig-${GT}-s${SEEDS// /}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="$EVS" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
  sleep 20
}

launch_cu() {  # launch_cu <gtag> <gamma> <smoke> [evalseeds]
  local GT=$1 GAMMA=$2 SMK=$3 EVS=${4:-"42 43 44"}
  local NAME="rlp-cu-unig-${GT}"
  [ "$SMK" = 1 ] && NAME="rlp-cu-unig-smoke"
  echo "==> $NAME (cube lewm gamma=$GAMMA smoke=$SMK)"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=cube --env BASE=lewm --env GRID=unig \
    --env SPLIT=0 --env SMOKE="$SMK" --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA=1.0 --env CU_GAMMA="$GAMMA" --env TR_NSTEP=50 \
    --env CACHE_VERSION="cu-unig-${GT}-v1" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env TRAIN_SEEDS="0 1 2" --env EVAL_SEEDS="$EVS" \
    --env STEPS=6000 --env BATCH=256 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
  sleep 20
}

launch_re() {  # launch_re <gtag> <gamma> <vnorm> <smoke>
  local GT=$1 GAMMA=$2 VN=$3 SMK=$4
  local NAME="rlp-re-unig-${GT}-${VN}"
  [ "$SMK" = 1 ] && NAME="rlp-re-unig-smoke"
  echo "==> $NAME (reacher lejepa gamma=$GAMMA vnorm=$VN smoke=$SMK)"
  local RONLY=1; [ "$SMK" = 1 ] && RONLY=0   # SMOKE pins its own draw
  sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env BASE=lejepa --env GRID=cross \
    --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
    --env RS_GAMMA="$GAMMA" --env RS_VNORM="$VN" \
    --env SMOKE="$SMK" --env REPORT_ONLY="$RONLY" \
    --env TRAIN_SEEDS="0 1 2" \
    --env EXPERIMENT_TAG="${NAME}-${DATE}" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
  sleep 20
}

if [ "$STAGE" = smoke ]; then
  launch_tw g99 0.99 "0" 1 "42"
  launch_cu g99 0.99 1 "42"
  launch_re g99 0.99 log 1
  echo "3 smoke canaries submitted (12 GPUs). Gate the fleet on all three SUCCEEDED."
else
  for GT_G in "g98 0.98" "g99 0.99" "g100 1.0"; do
    set -- $GT_G
    for S in 0 1 2; do launch_tw "$1" "$2" "$S" 0; done
  done
  for GT_G in "g98 0.98" "g99 0.99" "g100 1.0"; do
    set -- $GT_G
    launch_cu "$1" "$2" 0
  done
  for GT_G in "g98 0.98" "g99 0.99" "g100 1.0"; do
    set -- $GT_G
    for VN in none log; do launch_re "$1" "$2" "$VN" 0; done
  done
  echo "18 fleet jobs submitted (72 GPUs)."
fi
