#!/usr/bin/env bash
# Counterstrike expand ablation (second wave, 2026-08-16): does value
# expansion through WM rollouts (planner.expand_weight) help or hurt on
# PushT? All first-wave cells ran the cube recipe expand=1.0; Reacher's
# winner was expand=0, and expansion is a value-error amplifier when the
# WM's h-step reads are noisy. 4 cells at expand=0 mirror the first wave's
# central cells (lr 3e-4). Selection seeds 50/51, p2, shared caches.
#
# Usage: scripts/sky/launch_counterstrike_expand.sh [--dry-run]

set -euo pipefail
cd "$(dirname "$0")/../.."

DRY=${1:-}
CACHE_TAG=counterstrike-20260815
SWEEP=cs-swp-20260815

launch () {
  local name=$1; shift
  local cmd=(sky jobs launch scripts/sky/counterstrike_pusht.yaml
    -n "$name" --priority p2 --async -y
    --env CACHE_TAG=$CACHE_TAG --env WAIT_CACHE_MIN=240
    --env EVAL_SEEDS="50 51" --env EVAL_CONDS=rlp
    --env EXPERIMENT_TAG="$SWEEP-$name")
  for kv in "$@"; do cmd+=(--env "$kv"); done
  if [ "$DRY" = --dry-run ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}

for AMAX in 1.6 2.2; do
  for MW in 0.1 0.3; do
    NAME="cs-e0-a$(echo $AMAX | tr -d .)-m$(echo $MW | tr -d .)"
    launch "$NAME" "AMAX=$AMAX" "MEAN_WEIGHT=$MW" "ACTOR_LR=3e-4" "EXPAND=0"
  done
done

echo "launched 4 expand=0 cells at p2"
