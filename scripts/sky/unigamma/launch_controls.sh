#!/usr/bin/env bash
# Planner controls under the uniJ protocol: CEM / Adam / MPPI with the VALUE cost on every cell, plus the LATENT cost
# for MPPI only (user decision 2026-09-24: latent arms for CEM/Adam are not needed), same world model, same
# held-out split, same report draws, and for the value cost the cell's LOCKED teacher snapshot (one arm per training
# seed => n=6). Eval only, no actor training. L2O / DMPO are a separate wave (they train an inner-loop policy).
#   scripts/sky/unigamma/launch_controls.sh reacher|tworoom|cube [GPUS=1]
set -uo pipefail
cd "$(dirname "$0")/../../.."; export PATH="$HOME/.sky/bin:$PATH"
eval "$(python3 scripts/sky/unigamma/recipes/recipe_env.py scripts/sky/unigamma/recipes/uniJ.yaml)"
ENV=${1:?reacher|tworoom|cube}; GPUS=${GPUS:-1}; DATE=${DATE:-20260923}; USERV=armin@pantheon.inc; PRIO=${PRIO:-p1}
PLANNERS=${PLANNERS:-"cem adam mppi"}; PLANNERS_LATENT=${PLANNERS_LATENT:-mppi}
case $ENV in
  reacher) CELLS=${CELLS:-"reacher:lejepa:25:18000 reacher:pldm:25:18000"} ;;
  tworoom) CELLS=${CELLS:-"tworoom:lejepa:25:15000 tworoom:lejepa:100:6000 tworoom:pldm:25:3000 tworoom:pldm:100:3000"} ;;
  cube)    CELLS=${CELLS:-"cube:lewm:25:3000 cube:lewm:100:9000 cube:pldm:25:15000 cube:pldm:100:12000"} ;;
  *) echo "unknown env $ENV" >&2; exit 2 ;;
esac
for c in $CELLS; do
  IFS=: read -r ENVN BASE HZ TSTEP <<< "$c"
  P=""; [ "$BASE" = pldm ] && P="-p"
  if [ "$HZ" = 25 ]; then GAMMA=$UNI_GAMMA_H25; else GAMMA=$UNI_GAMMA_H100; fi
  # the locked teacher of reacher's cells is the FINAL 18k teacher -> no _step suffix
  CSTEP=$TSTEP; [ "$TSTEP" = "$UNI_TD_STEPS" ] && CSTEP=""
  if [ "$ENVN" = reacher ]; then
    NAME=rlp-re-ctrl-h25${P}; TAG=${NAME}-${DATE}
    echo "==> $ENVN/$BASE h$HZ teacher ${CSTEP:-final} ($NAME)"
    sky jobs launch scripts/sky/unigamma/reacher_gamma.yaml -n "$NAME" --priority $PRIO --gpus H200:$GPUS -y --async \
      --env BASE=$BASE --env GRID=latched_measurements --env PLANNERS="$PLANNERS" --env PLANNERS_LATENT="$PLANNERS_LATENT" --env MEASUREMENT_COSTS="${COSTS:-latent value}" \
      --env PLANNER_CRIT_SEEDS="$UNI_SEEDS" --env PLANNER_CRIT_STEP="$CSTEP" --env PLANNER_RH=$UNI_RH \
      --env REPORT_DRAWS="$UNI_REPORT_DRAWS" --env SUCCESS_THRESHOLDS="0.1 0.05" --env VALUE_FRAMES=$UNI_WINDOW_REACHER \
      --env RS_GAMMA=$UNI_GAMMA_H25 --env RS_NSTEP=$UNI_NSTEP --env CRIT_DEPTH=$UNI_CRIT_DEPTH --env VALUE_EXPECTILE=$UNI_TD_EXPECTILE \
      --env RS_TD_STEPS=$UNI_TD_STEPS --env RS_TD_SAVE_EVERY=$UNI_TD_SNAP --env RS_TD_PER_SEED=$UNI_TD_PER_SEED \
      --env TRAIN_SEEDS="$UNI_SEEDS" --env SMOKE=0 --env EXPERIMENT_TAG="$TAG" --env MAXPAR=$GPUS \
      --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV 2>&1 | { grep -E "Submitted|sky jobs logs [0-9]+|rror" || true; } | tail -1
  else
    PFX=tw; [ "$ENVN" = cube ] && PFX=cu
    CVSFX="-uniJ-g${GAMMA#0.}-e${UNI_TD_EXPECTILE#0.}-t$((UNI_TD_STEPS/1000))k"
    NAME=rlp-${PFX}-ctrl-h${HZ}${P}; TAG=${NAME}-${DATE}
    echo "==> $ENVN/$BASE h$HZ teacher ${CSTEP:-final} ($NAME)"
    sky jobs launch scripts/sky/unigamma/tworoom_g98_rescue.yaml -n "$NAME" --priority $PRIO --gpus H200:$GPUS -y --async \
      --env ENVNAME=$ENVN --env BASE=$BASE --env GRID=planners --env PLANNERS="$PLANNERS" --env PLANNERS_LATENT="$PLANNERS_LATENT" --env PLANNER_COSTS="${COSTS:-latent value}" \
      --env PLANNER_CRIT_SEEDS="$UNI_SEEDS" --env PLANNER_CRIT_STEP="$CSTEP" \
      --env EVAL_SEEDS="$UNI_REPORT_DRAWS" --env UNIG_OFF=$HZ --env OFFSETS=$HZ --env RH=$UNI_RH \
      --env SPLIT=1 --env SMOKE=0 --env REUSE_ONLY=0 --env ACTOR_ONLY=0 --env STAGED=0 --env FULLCACHE=0 \
      --env SHARD_INDEX=0 --env SHARD_COUNT=1 --env TD_PER_SEED=$UNI_TD_PER_SEED --env TRAIN_SEEDS="$UNI_SEEDS" \
      --env TR_GAMMA=$GAMMA --env CU_GAMMA=$GAMMA --env TW_TD_EXPECTILE=$UNI_TD_EXPECTILE --env CU_TD_EXPECTILE=$UNI_TD_EXPECTILE \
      --env TD_STEPS=$UNI_TD_STEPS --env TD_SAVE_EVERY=$UNI_TD_SNAP --env TD_SNAPS="3000 6000 9000 12000 15000" \
      --env TR_NSTEP=$UNI_NSTEP --env CRIT_DEPTH=$UNI_CRIT_DEPTH --env TDB=$UNI_TD_BATCH \
      --env CACHE_VERSION="${PFX}-v2${P}-n1s2-v1${CVSFX}" --env EXPERIMENT_TAG="$TAG" --env MAXPAR=$GPUS --env CUBE_LOCK=${CUBE_LOCK:-gpu} \
      --env JANITOR_PRUNE_PREFIXES="rlp-re-uniJ rlp-re-unicrit" \
      --env WANDB_PROJECT=RLP --env WANDB_ENTITY=armin-sommer --env PANTHEON_USER=$USERV 2>&1 | { grep -E "Submitted|sky jobs logs [0-9]+|rror" || true; } | tail -1
  fi
done
