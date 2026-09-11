#!/usr/bin/env bash
# Counterstrike wave 3 (2026-08-16): pure-LIPv4 resolution of the inference
# optimization limit — user-directed: NO restarts / argmin-V; instead give
# the one-shot refiner MORE, FINER steps (iterations up, amax down).
#
# Rationale: K iterations x amax clip bounds the total residual budget; the
# restart dose-response showed the refiner lands in the wrong contact mode
# with K=8 coarse steps. K {16, 24} x amax {0.8 1.2 1.6 2.2} tests whether a
# longer, finer refinement path converges across modes on its own. Fixed at
# the wave-1 winners: expand=0, mean_weight 0.1, actor_lr 3e-4. Two extra
# K=8 cells complete the low-amax column of wave 1.
#
# Usage: scripts/sky/launch_counterstrike_iters.sh [--dry-run]

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
    --env MEAN_WEIGHT=0.1 --env ACTOR_LR=3e-4 --env EXPAND=0
    --env EXPERIMENT_TAG="$SWEEP-$name")
  for kv in "$@"; do cmd+=(--env "$kv"); done
  if [ "$DRY" = --dry-run ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}

for K in 16 24; do
  for AMAX in 0.8 1.2 1.6 2.2; do
    launch "cs-k${K}-a$(echo $AMAX | tr -d .)" "ITERS=$K" "AMAX=$AMAX"
  done
done
# complete the K=8 (LIPv4 default) low-amax column at expand=0
for AMAX in 0.8 1.2; do
  launch "cs-k8-a$(echo $AMAX | tr -d .)" "AMAX=$AMAX"
done

echo "launched 10 iters-x-amax cells at p2"
