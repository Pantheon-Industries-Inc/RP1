#!/usr/bin/env bash
# Aggregate counterstrike sweep results into one table.
# Pulls each cell's RESULT blocks from its managed-job log and prints
#   <cell> <label> <seed> <success_rate>
# Usage: scripts/sky/collect_counterstrike_sweep.sh [name-prefix]  (default cs-)

set -euo pipefail
PREFIX=${1:-cs-}

sky jobs queue 2>/dev/null | awk -v p="$PREFIX" '$3 ~ "^"p {print $1, $3}' | sort -k2 |
while read -r ID NAME; do
  sky jobs logs "$ID" --no-follow 2>/dev/null |
    sed 's/\x1b\[[0-9;]*m//g' |
    awk -v cell="$NAME" '
      /----- RESULT / { label=$3; seed=$4; sub("seed=", "", seed); getline
        if (match($0, /success_rate.: [0-9.]+/)) {
          rate=substr($0, RSTART+15, RLENGTH-15)
          printf "%-18s %-12s %-4s %s\n", cell, label, seed, rate
        } }'
done
