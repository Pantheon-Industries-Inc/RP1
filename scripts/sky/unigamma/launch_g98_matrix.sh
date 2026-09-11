#!/usr/bin/env bash
# g98-matrix expansion -- 2026-08-18. Full TwoRoom gamma-unification sweep.
#
# RLP h25 + h100 on lejepa + pldm, each cell trained individually, for the
# three rescue arms (0.98/n100, 0.98/n200, 0.996/n50 = "0.98 per block"),
# plus a gamma=1.0/n50 PLDM reference at the SAME anchor rows, plus the
# TD-critic baseline triad (value + CEM / Adam / MPPI) per (base x arm).
#
# LeJEPA h100 RLP cells are already in flight (jobs 8022-8024, GRID=g98rescue)
# and are NOT relaunched here (ONLYCFG=g98m_h25 on the lejepa RLP jobs).
#
# Footprint: 17 jobs x H200:4 = 68 GPUs + 12 in flight = 80 (the budget).
# The two remaining lejepa-h25 jobs (n100, n200) launch when 8022-8024 free
# their nodes -- see launch_g98_matrix_stage2.sh.
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"

DATE=20260818
EVAL_SEEDS="42 43 44"

# launch_rlp <name> <base> <gamma> <nstep> <train_seeds> <onlycfg>
launch_rlp() {
  local NAME=$1 BASE=$2 GAMMA=$3 NSTEP=$4 SEEDS=$5 ONLY=$6
  echo "==> $NAME (RLP $BASE gamma=$GAMMA n=$NSTEP seeds='$SEEDS' only='$ONLY')"
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
    --env PANTHEON_USER=armin@pantheon.inc \
    2>&1 | tail -1
}

# launch_base <name> <base> <gamma> <nstep>   (TD+CEM/Adam/MPPI, value cost)
launch_base() {
  local NAME=$1 BASE=$2 GAMMA=$3 NSTEP=$4
  echo "==> $NAME (baselines $BASE gamma=$GAMMA n=$NSTEP)"
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
    --env PANTHEON_USER=armin@pantheon.inc \
    2>&1 | tail -1
}

# ---- PLDM RLP: 3 rescue arms x 3 seeds, sharded per seed (fast wall-clock)
for S in 0 1 2; do launch_rlp "rlp-tw-g98m-p-n100-s${S}" pldm 0.98  100 "$S" ""; done
for S in 0 1 2; do launch_rlp "rlp-tw-g98m-p-n200-s${S}" pldm 0.98  200 "$S" ""; done
for S in 0 1 2; do launch_rlp "rlp-tw-g98m-p-g996-s${S}" pldm 0.996 50  "$S" ""; done

# ---- PLDM gamma=1.0 reference at the same anchor rows (packed, 3 seeds)
launch_rlp "rlp-tw-g98m-p-g1ref" pldm 1.0 50 "0 1 2" ""

# ---- LeJEPA h25 (h100 in flight): g996 arm now; n100/n200 in stage 2
launch_rlp "rlp-tw-g98m-l-g996-h25" lejepa 0.996 50 "0 1 2" "g98m_h25"

# ---- Baselines: TD+CEM / TD+Adam / TD+MPPI per (base x arm)
launch_base "rlp-tw-g98m-b-l-n100" lejepa 0.98  100
launch_base "rlp-tw-g98m-b-l-n200" lejepa 0.98  200
launch_base "rlp-tw-g98m-b-l-g996" lejepa 0.996 50
launch_base "rlp-tw-g98m-b-p-n100" pldm   0.98  100
launch_base "rlp-tw-g98m-b-p-n200" pldm   0.98  200
launch_base "rlp-tw-g98m-b-p-g996" pldm   0.996 50

echo "17 jobs submitted (68 GPUs; 80 with the 3 in-flight rescue jobs)."
