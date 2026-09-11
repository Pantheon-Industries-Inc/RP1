#!/usr/bin/env bash
# g1actor: actor-hyper-only sweep at h100, Table-1 recipe otherwise.
# Finer sharding: one job per (base x seed x vnorm x mw) = 4-6 configs = one
# training wave. Canary-gated on the W&B checkpoint restore.
set -euo pipefail
export PATH="$HOME/.sky/bin:$PATH"
G() { local NAME=$1 BASE=$2 SEEDS=$3 EVS=$4 ONLY=$5
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority p1 -y --async \
    --env ENVNAME=tworoom --env BASE="$BASE" --env GRID=g1actor --env VNHALF=all \
    --env SPLIT=1 --env SMOKE=0 --env ONLYCFG="$ONLY" --env INCLUDE_WINNERS=0 --env REUSE_ONLY=0 \
    --env ACTOR_ONLY=0 --env STAGED=0 --env ACTOR_IMPORT_TAG="" --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 \
    --env TR_GAMMA=1.0 --env TR_NSTEP=50 \
    --env CACHE_VERSION="tw-${NAME}-v1" --env EXPERIMENT_TAG="${NAME}-20260818" \
    --env TRAIN_SEEDS="$SEEDS" --env EVAL_SEEDS="$EVS" \
    --env STEPS=8000 --env BATCH=128 --env MAXPAR=4 --env RH=5 \
    --env MAX_DELTA=12 --env OFFSETS="100" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer \
    --env PANTHEON_USER=armin@pantheon.inc 2>&1 | tail -1; echo "-> $NAME"; sleep 4; }

G "rlp-tw-g1a-canary" lejepa "0" "50 51" "g1a_a1.8_m0.1_log"
CJ=""; until [ -n "$CJ" ]; do CJ=$(sky jobs queue 2>/dev/null | grep "rlp-tw-g1a-canary" | head -1 | awk '{print $1}'); sleep 20; done
echo "canary job $CJ"
while true; do
  L=$(sky jobs logs $CJ --no-follow 2>/dev/null | grep -E "wandb-restore|cannot overwrite|smudge|weights.pt only" | head -2)
  [ -n "$L" ] && { echo "$L"; break; }
  ST=$(sky jobs queue 2>/dev/null | awk -v j=$CJ '$1==j {print $(NF-2)}')
  [ "$ST" = FAILED ] && { echo "CANARY FAILED"; sky jobs logs $CJ --no-follow 2>&1 | tail -6; exit 1; }
  sleep 75
done
echo "$L" | grep -q "restored" || { echo "canary restore not confirmed: $L"; exit 1; }
echo "=== canary OK -> fleet ==="
for BASE in lejepa pldm; do
  AMS="1.8 2.2 2.6 3.0"; [ "$BASE" = pldm ] && AMS="1.8 2.4 2.8 3.2"
  B1=$(echo $BASE | cut -c1)
  for S in 0 1 2; do for V in none log; do for M in 0.1 0.3; do
    ONLY=""; for A in $AMS; do ONLY="$ONLY g1a_a${A}_m${M}_${V}"; done
    G "rlp-tw-g1a-${B1}-s${S}-${V}-m${M/0./}" "$BASE" "$S" "50 51" "${ONLY# }"
  done; done; done
done
G "rlp-tw-g1a-hedge-l" lejepa "0 1 2" "42 43 44" "g1a_a2.6_m0.3_none g1a_a2.6_m0.3_log g1a_a1.8_m0.1_log"
G "rlp-tw-g1a-hedge-p" pldm "0 1 2" "42 43 44" "g1a_a2.8_m0.3_none g1a_a2.8_m0.3_log g1a_a1.8_m0.1_log"
echo G1ACTOR_FLEET_LAUNCHED
