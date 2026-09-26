#!/usr/bin/env bash
# MPPI baseline under the uniJ protocol, BOTH costs: latent (parameter-free single-frame latent distance, one arm) and
# value (the cell's locked TD teacher SNAPSHOT, one arm per training seed = n=6). Same world model,
# same held-out split and report draws (42-44), one arm per training seed (n=6). No actor training -- eval only.
#   scripts/sky/unigamma/launch_mppi.sh [GPUS=1] [DATE=20260923]
# Cells and their locked teacher budgets (from docs/campaigns/2026-09-22/RESULTS_uniJ_shared_stop.md):
#   tworoom/lejepa h25 15000 h100 6000 | tworoom/pldm h25 3000 h100 3000
#   cube/lewm     h25 3000  h100 9000 | cube/pldm    h25 15000 h100 12000
set -uo pipefail
cd "$(dirname "$0")/../../.."; export PATH="$HOME/.sky/bin:$PATH"
eval "$(python3 scripts/sky/unigamma/recipes/recipe_env.py scripts/sky/unigamma/recipes/uniJ.yaml)"
GPUS=${GPUS:-1}; DATE=${DATE:-20260923}; USERV=armin@pantheon.inc; PRIO=${PRIO:-p1}
CELLS=${CELLS:-"tworoom:lejepa:25:15000 tworoom:lejepa:100:6000 tworoom:pldm:25:3000 tworoom:pldm:100:3000 cube:lewm:25:3000 cube:lewm:100:9000 cube:pldm:25:15000 cube:pldm:100:12000"}
for c in $CELLS; do
  IFS=: read -r ENVN BASE HZ TSTEP <<< "$c"
  P=""; [ "$BASE" = pldm ] && P="-p"
  if [ "$HZ" = 25 ]; then GAMMA=$UNI_GAMMA_H25; else GAMMA=$UNI_GAMMA_H100; fi
  CVSFX="-uniJ-g${GAMMA#0.}-e${UNI_TD_EXPECTILE#0.}-t$((UNI_TD_STEPS/1000))k"
  PFX=tw; [ "$ENVN" = cube ] && PFX=cu
  CV="${PFX}-v2${P}-n1s2-v1${CVSFX}"                       # same cache key as the locked cell => teachers reused on a warm node
  NAME=rlp-${PFX}-mppi-h${HZ}${P}; TAG=${NAME}-${DATE}
  echo "==> $ENVN/$BASE h$HZ teacher step $TSTEP gamma $GAMMA ($NAME)"
  sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO --gpus H200:$GPUS -y --async \
    --env ENVNAME=$ENVN --env BASE=$BASE --env GRID=planners --env PLANNERS=mppi --env PLANNER_COSTS="${PLANNER_COSTS:-latent value}" \
    --env PLANNER_CRIT_SEEDS="$UNI_SEEDS" --env PLANNER_CRIT_STEP=$TSTEP \
    --env EVAL_SEEDS="$UNI_REPORT_DRAWS" --env UNIG_OFF=$HZ --env OFFSETS=$HZ --env RH=$UNI_RH \
    --env SPLIT=1 --env SMOKE=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 --env STAGED=0 --env FULLCACHE=0 \
    --env SHARD_INDEX=0 --env SHARD_COUNT=1 --env TD_PER_SEED=$UNI_TD_PER_SEED --env TRAIN_SEEDS="$UNI_SEEDS" \
    --env TR_GAMMA=$GAMMA --env CU_GAMMA=$GAMMA --env TW_TD_EXPECTILE=$UNI_TD_EXPECTILE --env CU_TD_EXPECTILE=$UNI_TD_EXPECTILE \
    --env TD_STEPS=$UNI_TD_STEPS --env TD_SAVE_EVERY=$UNI_TD_SNAP --env TD_SNAPS="3000 6000 9000 12000 15000" \
    --env TR_NSTEP=$UNI_NSTEP --env CRIT_DEPTH=$UNI_CRIT_DEPTH --env TDB=$UNI_TD_BATCH \
    --env CACHE_VERSION="$CV" --env EXPERIMENT_TAG="$TAG" --env MAXPAR=$GPUS --env CUBE_LOCK=${CUBE_LOCK:-gpu} \
    --env JANITOR_PRUNE_PREFIXES="rlp-re-uniJ rlp-re-unicrit" \
    --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV 2>&1 | { grep -E "Submitted|sky jobs logs [0-9]+|rror" || true; } | tail -1
done
