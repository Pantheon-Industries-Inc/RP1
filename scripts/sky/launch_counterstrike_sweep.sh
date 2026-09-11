#!/usr/bin/env bash
# Counterstrike (PushT RLP) hyperparameter sweep — 2026-08-15.
#
# Grid (24 cells): AMAX {1.0 1.6 2.2 3.0} x MEAN_WEIGHT {0.1 0.3 0.5}
#                  x ACTOR_LR {1e-4 3e-4}, planner co-critic TD = cube defaults
# Probe (4 cells): TD_MODE=pusht (co-critic gamma 1.0 / n1 / exp .03->.01) at
#                  AMAX {1.6 2.2} x MEAN_WEIGHT {0.1 0.3}, lr 3e-4 — does the
#                  velocity-aliasing short-backup corner help the co-trained
#                  critic too, or only the offline init value?
#
# Protocol: SELECTION on eval seeds 50/51 only (never report those); the
# winner then reruns on report seeds 42/43/44 with three actor seeds.
# All cells share the seed-0 full run's caches (CACHE_TAG) and wait up to
# 4 h for its encode stages to land. Priority p2 per sweep convention.
#
# Usage: scripts/sky/launch_counterstrike_sweep.sh [--dry-run]

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=${1:-}
CACHE_TAG=counterstrike-20260815
SWEEP=cs-swp-20260815

launch () {  # launch <name> <env overrides...>
  local name=$1; shift
  local cmd=(sky jobs launch scripts/sky/counterstrike_pusht.yaml
    -n "$name" --priority p2 --async -y
    --env CACHE_TAG=$CACHE_TAG --env WAIT_CACHE_MIN=240
    --env EVAL_SEEDS="50 51" --env EXPERIMENT_TAG="$SWEEP-$name")
  for kv in "$@"; do cmd+=(--env "$kv"); done
  if [ "$DRY" = --dry-run ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}

slug () { echo "$1" | tr -d '.' | tr -d '-'; }

for AMAX in 1.0 1.6 2.2 3.0; do
  for MW in 0.1 0.3 0.5; do
    for LR in 1e-4 3e-4; do
      NAME="cs-a$(slug $AMAX)-m$(slug $MW)-l$(slug $LR)"
      CONDS="rlp"
      # one planner-independent CEM anchor on the selection seeds
      [ "$AMAX" = 1.6 ] && [ "$MW" = 0.1 ] && [ "$LR" = 3e-4 ] && CONDS="rlp cem_latent"
      launch "$NAME" "AMAX=$AMAX" "MEAN_WEIGHT=$MW" "ACTOR_LR=$LR" \
        "EVAL_CONDS=$CONDS"
    done
  done
done

for AMAX in 1.6 2.2; do
  for MW in 0.1 0.3; do
    NAME="cs-tdp-a$(slug $AMAX)-m$(slug $MW)"
    launch "$NAME" "AMAX=$AMAX" "MEAN_WEIGHT=$MW" "ACTOR_LR=3e-4" \
      "TD_MODE=pusht" "EVAL_CONDS=rlp"
  done
done

echo "launched 28 cells (24 grid + 4 TD-mode probe) at p2"
