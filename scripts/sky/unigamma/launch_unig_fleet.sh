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

launch_tw_bnd() {  # launch_tw_bnd <btag> <boundary> <seed> [gtag] [gamma]
  local BT=$1 BND=$2 S=$3 GT=${4:-g98} GAMMA=${5:-0.98}
  local NAME="rlp-tw-unig-${GT}${BT}-s${S}"
  echo "==> $NAME (tworoom lejepa gamma=$GAMMA boundary=$BND seed=$S)"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE=lejepa --env GRID=unig \
    --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="unig_ctrl" \
    --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
    --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="$GAMMA" --env TR_NSTEP=50 --env BOUNDARY="$BND" \
    --env CACHE_VERSION="tw-unig-${GT}${BT}-s${S}-v1" \
    --env EXPERIMENT_TAG="${NAME}-20260819" \
    --env TRAIN_SEEDS="$S" --env EVAL_SEEDS="42 43 44" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
  sleep 20
}

launch_re_bnd() {  # launch_re_bnd <btag> <boundary> [gtag] [gamma]
  local BT=$1 BND=$2 GT=${3:-g98} GAMMA=${4:-0.98}
  local NAME="rlp-re-unig-${GT}${BT}"
  echo "==> $NAME (reacher lejepa gamma=$GAMMA boundary=$BND)"
  sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env BASE=lejepa --env GRID=cross \
    --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
    --env RS_GAMMA="$GAMMA" --env RS_VNORM=none --env RS_BOUNDARY="$BND" \
    --env SMOKE=0 --env REPORT_ONLY=1 \
    --env TRAIN_SEEDS="0 1 2" \
    --env EXPERIMENT_TAG="${NAME}-20260819" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
  sleep 20
}

# STAGE selects a slice so each environment's fleet can gate on its own smoke:
#   smoke | tw | cu | re | fleet (= tw + cu + re) | bnd (TD-seam variants)
run_tw() { for GT_G in "g98 0.98" "g99 0.99" "g100 1.0"; do set -- $GT_G
  for S in 0 1 2; do launch_tw "$1" "$2" "$S" 0; done; done; }
run_cu() { for GT_G in "g98 0.98" "g99 0.99" "g100 1.0"; do set -- $GT_G
  launch_cu "$1" "$2" 0; done; }
run_re() { for GT_G in "g98 0.98" "g99 0.99" "g100 1.0"; do set -- $GT_G
  for VN in none log; do launch_re "$1" "$2" "$VN" 0; done; done; }

case "$STAGE" in
  smoke)
    launch_tw g99 0.99 "0" 1 "42"
    launch_cu g99 0.99 1 "42"
    launch_re g99 0.99 log 1
    echo "3 smoke canaries submitted (12 GPUs). Gate each slice on its smoke." ;;
  tw) run_tw; echo "9 tworoom jobs submitted (36 GPUs)." ;;
  cu) run_cu; echo "3 cube jobs submitted (12 GPUs)." ;;
  re) run_re; echo "6 reacher jobs submitted (24 GPUs)." ;;
  fleet) run_tw; run_cu; run_re; echo "18 fleet jobs submitted (72 GPUs)." ;;
  bnd)
    for BT_B in "sm smooth" "dc disc"; do set -- $BT_B
      for S in 0 1 2; do launch_tw_bnd "$1" "$2" "$S"; done
      launch_re_bnd "$1" "$2"
    done
    echo "8 boundary jobs submitted (32 GPUs)." ;;
  bnd99)   # boundary-fixed gamma=0.99 arms (gamma=1.0 needs no rerun: both
           # fixes reduce to the identical target at g=1, so the fleet g100
           # arms already ARE the boundary-fixed gamma=1 points)
    for BT_B in "sm smooth" "dc disc"; do set -- $BT_B
      for S in 0 1 2; do launch_tw_bnd "$1" "$2" "$S" g99 0.99; done
      launch_re_bnd "$1" "$2" g99 0.99
    done
    echo "8 gamma=0.99 boundary jobs submitted (32 GPUs)." ;;
  md20)  # the unified horizon-covering band: gamma=0.98, md=20, amax=2.5.
         # Hypothesis (Section 6a interaction): the 0.98 discount plays the
         # E-compressor role vlog played at a1.8, making the wide band usable
         # with raw E. Smooth-boundary probe carried because md20's band
         # (100 steps) straddles the n=50 seam.
    for S in 0 1 2; do
      NAME="rlp-tw-unig-g98md20-s${S}"
      echo "==> $NAME"
      sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
        -n "$NAME" --priority p1 -y --async \
        --env ENVNAME=tworoom --env BASE=lejepa --env GRID=unig \
        --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="" \
        --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
        --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
        --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
        --env TR_GAMMA=0.98 --env TR_NSTEP=50 --env UNIG_MD=20 \
        --env CACHE_VERSION="tw-unig-g98-s${S}-v1" \
        --env EXPERIMENT_TAG="${NAME}-20260819" \
        --env TRAIN_SEEDS="$S" --env EVAL_SEEDS="42 43 44" \
        --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
        --env MAX_DELTA=12 --env OFFSETS="25 100" \
        --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
        --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
      sleep 20
      NAME="rlp-tw-unig-g98md20sm-s${S}"
      echo "==> $NAME"
      sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
        -n "$NAME" --priority p1 -y --async \
        --env ENVNAME=tworoom --env BASE=lejepa --env GRID=unig \
        --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="unig_ctrl" \
        --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
        --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
        --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
        --env TR_GAMMA=0.98 --env TR_NSTEP=50 --env UNIG_MD=20 --env BOUNDARY=smooth \
        --env CACHE_VERSION="tw-unig-g98sm-s${S}-v1" \
        --env EXPERIMENT_TAG="${NAME}-20260819" \
        --env TRAIN_SEEDS="$S" --env EVAL_SEEDS="42 43 44" \
        --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
        --env MAX_DELTA=12 --env OFFSETS="25 100" \
        --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
        --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
      sleep 20
    done
    NAME="rlp-cu-unig-g98md20"
    echo "==> $NAME"
    sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
      -n "$NAME" --priority p1 -y --async \
      --env ENVNAME=cube --env BASE=lewm --env GRID=unig \
      --env SPLIT=0 --env SMOKE=0 --env ONLYCFG="" \
      --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 \
      --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
      --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
      --env TR_GAMMA=1.0 --env CU_GAMMA=0.98 --env TR_NSTEP=50 --env UNIG_MD=20 \
      --env CACHE_VERSION="cu-unig-g98-v1" \
      --env EXPERIMENT_TAG="${NAME}-20260819" \
      --env TRAIN_SEEDS="0 1 2" --env EVAL_SEEDS="42 43 44" \
      --env STEPS=6000 --env BATCH=256 --env MAXPAR=4 --env RH=5 \
      --env MAX_DELTA=12 --env OFFSETS="25 100" \
      --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
      --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
    sleep 20
    for VN in none log; do
      NAME="rlp-re-unig-g98md20-${VN}"
      echo "==> $NAME"
      sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml \
        -n "$NAME" --priority p1 -y --async \
        --env BASE=lejepa --env GRID=cross \
        --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
        --env RS_GAMMA=0.98 --env RS_VNORM="$VN" --env RS_MD=20 \
        --env SMOKE=0 --env REPORT_ONLY=1 \
        --env TRAIN_SEEDS="0 1 2" \
        --env EXPERIMENT_TAG="${NAME}-20260819" \
        --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
        --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
      sleep 20
    done
    echo "9 md20 jobs submitted (36 GPUs)." ;;
  damax)  # deploy-clip sweep on the frozen gamma=0.98 fleet actors (eval-only):
          # imported checkpoints get top-level amax rewritten; training amax
          # stays 2.5 in train_args. Values: per-env recipe clip + 3.0 probe.
    for AMX in 1.8 3.0; do
      AT=${AMX/./p}
      for S in 0 1 2; do
        NAME="rlp-tw-unig-g98da${AT}-s${S}"
        echo "==> $NAME"
        sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
          -n "$NAME" --priority p1 -y --async \
          --env ENVNAME=tworoom --env BASE=lejepa --env GRID=unig \
          --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="unig_ctrl" \
          --env INCLUDE_WINNERS=0 --env REUSE_ONLY=1 --env ACTOR_ONLY=0 \
          --env STAGED=0 --env ACTOR_IMPORT_TAG="rlp-tw-unig-g98-s${S}-20260818" \
          --env DEPLOY_AMAX="$AMX" --env FULLCACHE=0 \
          --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
          --env TR_GAMMA=0.98 --env TR_NSTEP=50 \
          --env CACHE_VERSION="tw-unig-g98-s${S}-v1" \
          --env EXPERIMENT_TAG="${NAME}-20260819" \
          --env TRAIN_SEEDS="$S" --env EVAL_SEEDS="42 43 44" \
          --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
          --env MAX_DELTA=12 --env OFFSETS="25 100" \
          --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
          --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
        sleep 20
      done
    done
    for AMX in 1.6 3.0; do
      AT=${AMX/./p}
      NAME="rlp-cu-unig-g98da${AT}"
      echo "==> $NAME"
      sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
        -n "$NAME" --priority p1 -y --async \
        --env ENVNAME=cube --env BASE=lewm --env GRID=unig \
        --env SPLIT=0 --env SMOKE=0 --env ONLYCFG="unig_ctrl" \
        --env INCLUDE_WINNERS=0 --env REUSE_ONLY=1 --env ACTOR_ONLY=0 \
        --env STAGED=0 --env ACTOR_IMPORT_TAG="rlp-cu-unig-g98-20260818" \
        --env DEPLOY_AMAX="$AMX" --env FULLCACHE=0 \
        --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
        --env TR_GAMMA=1.0 --env CU_GAMMA=0.98 --env TR_NSTEP=50 \
        --env CACHE_VERSION="cu-unig-g98-v1" \
        --env EXPERIMENT_TAG="${NAME}-20260819" \
        --env TRAIN_SEEDS="0 1 2" --env EVAL_SEEDS="42 43 44" \
        --env STEPS=6000 --env BATCH=256 --env MAXPAR=4 --env RH=5 \
        --env MAX_DELTA=12 --env OFFSETS="25 100" \
        --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
        --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
      sleep 20
    done
    for AMX in 2.2 3.0; do
      AT=${AMX/./p}
      NAME="rlp-re-unig-g98da${AT}"
      echo "==> $NAME"
      sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml \
        -n "$NAME" --priority p1 -y --async \
        --env BASE=lejepa --env GRID=cross \
        --env AMFIX=2.5 --env CROSS_EXPANDS="0" --env CROSS_REPLAYS="0.5" \
        --env RS_GAMMA=0.98 --env RS_VNORM=none \
        --env REUSE_ONLY=1 --env ACTOR_IMPORT_TAG="rlp-re-unig-g98-none-20260818" \
        --env DEPLOY_AMAX="$AMX" \
        --env SMOKE=0 --env REPORT_ONLY=1 \
        --env TRAIN_SEEDS="0 1 2" \
        --env EXPERIMENT_TAG="${NAME}-20260819" \
        --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
        --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1
      sleep 20
    done
    echo "10 deploy-amax jobs submitted (40 GPUs, eval-only)." ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
