#!/usr/bin/env bash
# Start the Dyna-uniJ loop once all four base Cube cells are locked (their lock_*.out carry an n=6 test row).
cd "$(dirname "$0")/../../.."; SP=$PWD/scratchpad; N=${1:-3}
CELLS="rlp-cu-uniJ-h100-20260921 rlp-cu-uniJ-h100-p-20260921 rlp-cu-uniJ-h25-20260921 rlp-cu-uniJ-h25-p-20260921"
while :; do ok=0; for c in $CELLS; do grep -q "n=6 test" $SP/lock_$c.out 2>/dev/null && ok=$((ok+1)); done
  [ "$ok" = 4 ] && break; echo "[$(date -u +%m%d-%H:%M)] base locks: $ok/4"; sleep 600; done
echo "[$(date -u +%m%d-%H:%M)] all four base Cube cells locked -> Dyna-uniJ $N iterations"
SP=$SP exec bash scripts/sky/dyna2/run_dyna_uniJ.sh $N 20260923
