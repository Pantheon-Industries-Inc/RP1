#!/usr/bin/env bash
# TwoRoom RLP improvement campaign -- 2026-08-17.
#
# Targets the two cells where RLP trails value+CEM under the held-out protocol:
#   TwoRoom LeJEPA h100   RLP 94.2  vs  CEM 96.0  (and DMPO 99.1)
#   TwoRoom PLDM   h25    RLP 98.2  vs  CEM 100.0
#
# h25 and h100 actors are trained SEPARATELY: every arm carries its own
# max_delta (goal band) and is scored at its own single horizon, so an h100
# number is never read off an h25-tuned refiner. max_delta 12 covers the h25
# goal (5 blocks); max_delta 20 reaches the 100 primitive steps an h100 goal
# sits at. Arms named es25_/pl25_ are h25-trained, es100_/pl100_ are
# h100-trained; the *_md12 h100 arms deliberately keep the narrow band so the
# banked 94.2 / 96.0 conditions stay anchored inside the factorial.
#
#   GRID=escale          lejepa, 10 arms:  4 h25 (ctrl/log/loggn/log@a1.4)
#                        + 6 h100 (2x2 on md{12,20} x vnorm{none,log},
#                        plus loggn@md20 and log@md20@a2.6)
#   GRID=pldm_h25_probe  pldm,   12 arms:  9 h25 (amax 1.0-2.2 ladder at K=8,
#                        ctrl = 1.8, plus log and loggn at the ctrl clip)
#                        + 3 h100 (md12 anchor, md20, md20+log)
#
# Each grid is sharded ONE JOB PER TRAIN SEED. Sharding by seed (not by arm)
# puts the control in every job by construction and keeps each job to a single
# TD teacher (TDSEEDS=$TRAIN_SEEDS); replicating the control across seeds also
# measures job-to-job drift, the confound that comparing against a banked
# number cannot.
#
# 6 seeds x 2 grids = 12 jobs x H200:4 = 48 GPUs. Draws 50/51 (selection) and
# 42/43/44 (reporting), 50 episodes each -> 1,500 reported episodes per arm.
#
# Credentials are resolved by the API server from the platform secrets manager
# (the `secrets:NAME` reference form in tworoom_split_rerun.yaml), so nothing
# needs to be exported and no secret value appears here or on a command line.
#
#     bash scripts/sky/launch_tworoom_20260817.sh
#
# SMOKE=1 runs the tiny end-to-end path check instead (one config, one seed,
# 60 steps, one eval cell) under its own tag, so a 60-step actor can never be
# reused by the real job:
#
#     SMOKE=1 bash scripts/sky/launch_tworoom_20260817.sh
set -euo pipefail

SMOKE=${SMOKE:-0}
DATE=20260817
SUFFIX=""; [ "$SMOKE" = 1 ] && SUFFIX="-smoke"

# Selection draws first, then the reporting draws. Winners are picked on
# 50/51; 42/43/44 are the numbers that get quoted, and are never selected on.
EVAL_SEEDS="50 51 42 43 44"

launch() {           # launch <base> <grid> <seed> <cachekey>
  local BASE=$1 GRID=$2 SEED=$3 CKEY=$4
  local NAME="rlp-tw-${CKEY}-s${SEED}${SUFFIX}"
  echo "==> $NAME"
  sky jobs launch scripts/sky/tworoom_split_rerun.yaml \
    -n "$NAME" --priority p2 -y --async \
    --env ENVNAME=tworoom \
    --env BASE="$BASE" \
    --env GRID="$GRID" \
    --env SPLIT=1 \
    --env SMOKE="$SMOKE" \
    --env ONLYCFG="" \
    --env INCLUDE_WINNERS=0 \
    --env REUSE_ONLY=0 \
    --env ACTOR_ONLY=0 \
    --env STAGED=0 \
    --env ACTOR_IMPORT_TAG="" \
    --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env CACHE_VERSION="tw-${CKEY}-s${SEED}-v1${SUFFIX}" \
    --env EXPERIMENT_TAG="rlp-tw-${CKEY}-s${SEED}-${DATE}${SUFFIX}" \
    --env TRAIN_SEEDS="$SEED" \
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

if [ "$SMOKE" = 1 ]; then
  # One job per grid is enough to exercise both new code paths end to end:
  # escale covers --vnorm (train -> checkpoint -> solver reload), pldm_h25_probe
  # covers the K/offsets positional fields and the train_args provenance guard.
  launch lejepa escale           0 escale
  launch pldm   pldm_h25_probe   0 pldmamax
  echo
  echo "Smokes submitted. Gate before the real launch:"
  echo "  * '[grid] GRID=escale tworoom/lejepa -> 7 configs'  (NOT 0 -- ONLYCFG fails open)"
  echo "  * '[vnorm] log' from the trainer and '[lip] actor vnorm=log' from the solver"
  echo "  * '[args-ok] ... iters=8 amax=... md=... vnorm=...' before each eval"
  echo "  * '[summary]' present with non-null scores at both offsets"
  exit 0
fi

SEEDS=${SEEDS:-"0 1 2 3 4 5"}
for SEED in $SEEDS; do launch lejepa escale         "$SEED" escale;   done
for SEED in $SEEDS; do launch pldm   pldm_h25_probe "$SEED" pldmamax; done

cat <<'EOF'

12 jobs submitted (48 H200s). Watch with:
  sky jobs queue | grep rlp-tw-

Per-job sanity, in the logs:
  [grid] GRID=... -> 7 configs, 4 in flight      # 0 configs = ONLYCFG fell open
  [args-ok] ... iters=8 ...                      # every actor matches its row
  grep -c 'REUSED persistent actor' == 0         # nothing inherited from an old tag

Results land in /checkpoints/armin@pantheon.inc/rlp-tw-*-20260817/results/.
EOF
