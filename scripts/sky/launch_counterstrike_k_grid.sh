#!/usr/bin/env bash
# Counterstrike wave 5 (2026-08-16, user-directed): granular K x amax x
# mean-weight grid at the high-iteration regime. K=24/a2.2/m0.1 hit 78.0
# (vs 63.0 at K=8, TD+CEM 80.0, latent+CEM 83.0), amax trending UP at high
# K — this grid resolves the peak. ~30 GPUs.
#
#   K 24: amax {2.0 2.2 2.4 2.6 2.8 3.0} x mw {0.05 0.1 0.2}  (17 new;
#         a2.2/m0.1 = the wave-3 winner, already measured)
#   K 32: amax {2.0 2.2 2.4 2.6 3.0}     x mw {0.05 0.1 0.2}  (13 new;
#         a2.2/m0.1 + a3.0/m0.1 already launched as cs-k32-*)
# Fixed: expand=0, lr 3e-4, selection seeds 50/51, shared caches, p2.
#
# Usage: scripts/sky/launch_counterstrike_k_grid.sh [--dry-run]

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
    --env ACTOR_LR=3e-4 --env EXPAND=0
    --env EXPERIMENT_TAG="$SWEEP-$name")
  for kv in "$@"; do cmd+=(--env "$kv"); done
  if [ "$DRY" = --dry-run ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
}

slug () { echo "$1" | tr -d '.' ; }

for K in 24 32; do
  if [ "$K" = 24 ]; then AMAXES="2.0 2.2 2.4 2.6 2.8 3.0"; else AMAXES="2.0 2.2 2.4 2.6 3.0"; fi
  for AMAX in $AMAXES; do
    for MW in 0.05 0.1 0.2; do
      # already measured / already launched
      [ "$K" = 24 ] && [ "$AMAX" = 2.2 ] && [ "$MW" = 0.1 ] && continue
      [ "$K" = 32 ] && [ "$AMAX" = 2.2 ] && [ "$MW" = 0.1 ] && continue
      [ "$K" = 32 ] && [ "$AMAX" = 3.0 ] && [ "$MW" = 0.1 ] && continue
      launch "cs-k${K}-a$(slug $AMAX)-m$(slug $MW)" \
        "ITERS=$K" "AMAX=$AMAX" "MEAN_WEIGHT=$MW"
    done
  done
done

echo "launched 30 granular K-grid cells at p2"
