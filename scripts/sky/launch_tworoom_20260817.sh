#!/usr/bin/env bash
# TwoRoom RLP improvement campaign -- 2026-08-17.
#
# Targets the two cells where RLP trails value+CEM under the held-out protocol:
#   TwoRoom LeJEPA h100   RLP 94.2  vs  CEM 96.0  (and DMPO 99.1)
#   TwoRoom PLDM   h25    RLP 98.2  vs  CEM 100.0
#
# Two grids, each sharded ONE JOB PER TRAIN SEED. Sharding by seed (not by
# arm) means every job contains the control by construction, and each job
# trains only its own TD teacher (TDSEEDS=$TRAIN_SEEDS), so setup is cheap.
# Replicating the control across three jobs additionally measures job-to-job
# drift -- the exact confound that comparing against a banked number cannot.
#
#   GRID=escale          lejepa, 7 arms: ctrl / log / loggn / log@a2.6 /
#                        log@a1.4 / max_delta=20 / log+max_delta=20
#   GRID=pldm_h25_probe  pldm,   7 arms: amax 1.0-2.2 at K=8, ctrl = 1.8
#
# 6 jobs x H200:4 = 24 GPUs. Every arm is evaluated at BOTH h25 and h100 on
# draws 50/51 (selection) and 42/43/44 (reporting), 50 episodes each.
#
# Requires a GitHub PAT in the environment -- the harness clones
# Value_Metric_LeWM for stable-worldmodel. `--secret NAME` reads it from the
# environment, so no secret value ever appears in this file or in the command
# line. Run as:
#
#     export GIT_TOKEN=<pat>
#     bash scripts/sky/launch_tworoom_20260817.sh
#
# SMOKE=1 runs the tiny end-to-end path check instead (one config, one seed,
# 60 steps, one eval cell) under its own tag, so a 60-step actor can never be
# reused by the real job:
#
#     SMOKE=1 bash scripts/sky/launch_tworoom_20260817.sh
set -euo pipefail

: "${GIT_TOKEN:?export GIT_TOKEN=<github PAT> before launching}"
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
    -n "$NAME" --priority p1 -y --async \
    --secret GIT_TOKEN --secret WANDB_API_KEY \
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

for SEED in 0 1 2; do launch lejepa escale         "$SEED" escale;   done
for SEED in 0 1 2; do launch pldm   pldm_h25_probe "$SEED" pldmamax; done

cat <<'EOF'

6 jobs submitted (24 H200s). Watch with:
  sky jobs queue | grep rlp-tw-

Per-job sanity, in the logs:
  [grid] GRID=... -> 7 configs, 4 in flight      # 0 configs = ONLYCFG fell open
  [args-ok] ... iters=8 ...                      # every actor matches its row
  grep -c 'REUSED persistent actor' == 0         # nothing inherited from an old tag

Results land in /checkpoints/armin@pantheon.inc/rlp-tw-*-20260817/results/.
EOF
