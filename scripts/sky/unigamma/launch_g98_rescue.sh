#!/usr/bin/env bash
# g98-rescue campaign -- 2026-08-18. Can TwoRoom h100 work at gamma=0.98?
#
# Three arms, one job per arm (TD teacher is per-job, so gamma x n-step is
# job identity; seeds 0/1/2 train inside each job), LeJEPA h100 only -- the
# cell where gamma=0.98 collapses (Section 10 of the 2026-08-17 campaign).
# Each job carries a raw-E control row and a vnorm=log row at md20.
#
# Banked references (same protocol, jobs 7917-7919 / escale):
#   gamma=0.98 n50: ctrl 66.22 +- 21.7, vlog 81.78 +- 3.7
#   gamma=1.00 n50: ctrl 88.67 (seed 3), vlog 97.33 (seed 3; protocol-seed
#                   rerun 7914-7916 in flight)
#
#     bash scripts/sky/unigamma/launch_g98_rescue.sh
set -euo pipefail

DATE=20260818
EVAL_SEEDS=${EVAL_SEEDS:-"42 43 44"}

launch() {           # launch <tag> <gamma> <nstep>
  local TAG=$1 GAMMA=$2 NSTEP=$3
  local NAME="rlp-tw-${TAG}"
  echo "==> $NAME (TR_GAMMA=$GAMMA TR_NSTEP=$NSTEP)"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml \
    -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom \
    --env BASE=lejepa \
    --env GRID=g98rescue \
    --env SPLIT=1 \
    --env SMOKE=0 \
    --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 \
    --env REUSE_ONLY=0 \
    --env ACTOR_ONLY=0 \
    --env STAGED=0 \
    --env ACTOR_IMPORT_TAG="" \
    --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA="$GAMMA" \
    --env TR_NSTEP="$NSTEP" \
    --env CACHE_VERSION="tw-${TAG}-v1" \
    --env EXPERIMENT_TAG="rlp-tw-${TAG}-${DATE}" \
    --env TRAIN_SEEDS="0 1 2" \
    --env EVAL_SEEDS="$EVAL_SEEDS" \
    --env STEPS=8000 \
    --env BATCH=128 \
    --env MAXPAR=4 \
    --env RH=5 \
    --env MAX_DELTA=12 \
    --env OFFSETS="25 100" \
    --env WANDB_PROJECT=RLP \
    --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc \
    2>&1 | tail -2
}

launch g98n100 0.98  100
launch g98n200 0.98  200
launch g996n50 0.996 50

cat <<'EOF'

Submitted. Gate on, per job:
  * '[grid] GRID=g98rescue tworoom/lejepa -> 2 configs'
  * TD logs show '--gamma <arm> --n-step <arm>'
  * '[vnorm] log' from the trainer on the vlog rows
  * '[args-ok] ... vnorm=...' before each eval
  * '[summary]' with non-null h100 scores
EOF
