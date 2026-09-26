#!/usr/bin/env bash
# Lock one Cube cell under the shared-stop protocol: dump the cell's driver logs from the volume (base tag + -s345 +
# -s5 seed splits), pick the (teacher, actor step) pair with the best mean validation over the six seeds, require
# 36 pairs on every seed, then stage + test-evaluate that pair for all six seeds (scripts/cube_fixed_report.sh).
#   scripts/cube_lock_cell.sh <cell-tag> <base> <hz>          e.g. rlp-cu-uniJ-h100-20260921 lewm 100
#   env: DYNA_WM/DYNA_IT forwarded to the report job; WAIT=1 polls every 20 min until the grid is complete.
set -uo pipefail
CELL=$1; BASE=$2; HZ=$3; SP=${SP:-/tmp}
STEM=${CELL%-*}; DATE=${CELL##*-}                 # rlp-cu-uniJ-h100 / 20260921
TAGS="$CELL ${STEM}-s345-${DATE} ${STEM}-s5-${DATE}"
jid(){ grep -o "sky jobs logs [0-9]*" | grep -o "[0-9]*$" | head -1; }
wait_job(){ local id=$1 st=""; while :; do st=$(sky jobs queue 2>/dev/null | awk -v j=$id '$1==j' | grep -oE 'SUCCEEDED|FAILED[A-Z_]*|CANCELLED' | head -1); [ -n "$st" ] && { echo "$st"; return; }; sleep 60; done; }
while :; do
  RID=$(sky jobs launch -y -d --name "read-${CELL#rlp-cu-uniJ-}" --env TAGS="$TAGS" scripts/sky/tools/ckptval_read.yaml 2>&1 | jid)
  [ -n "$RID" ] || { echo "reader launch failed"; exit 1; }
  st=$(wait_job $RID); [ "$st" = SUCCEEDED ] || { echo "reader $RID $st"; exit 1; }
  sky jobs logs $RID --no-follow 2>&1 | perl -pe 's/^.*?rank=0\)\e\[0m //' > $SP/lock_${CELL}.dump
  DRIVER_DUMP=$SP/lock_${CELL}.dump scripts/collect_ckpt_select.sh 2>/dev/null > $SP/lock_${CELL}.sel
  grep ^SNAPH $SP/lock_${CELL}.sel > $SP/lock_${CELL}.snaph; grep -v ^SNAPH $SP/lock_${CELL}.sel > $SP/lock_${CELL}.dep
  PICK=$(python3 scripts/fixed_stop_select.py $SP/lock_${CELL}.snaph $SP/lock_${CELL}.dep 2>/dev/null | grep "^${CELL}:")
  echo "$PICK"
  if echo "$PICK" | grep -q "6/6 seeds" && ! echo "$PICK" | grep -q PRELIMINARY; then break; fi
  [ "${WAIT:-0}" = 1 ] || { echo "cell not complete; rerun with WAIT=1 to poll"; exit 2; }
  echo "[$(date -u +%H:%M)] not complete yet; polling again in 20 min"; sleep 1200
done
T=$(echo "$PICK" | grep -oE "FIXED teacher [0-9]+" | grep -oE "[0-9]+$"); STEP=$(echo "$PICK" | grep -oE "actor [0-9a-z]+" | cut -d' ' -f2)
ROW=unig_ctrl_a2.5; [ "$T" != 18000 ] && ROW="unig_ctrl_a2.5_t$T"
SEEDMAP="0:$CELL 1:$CELL 2:$CELL 3:${STEM}-s345-${DATE} 4:${STEM}-s345-${DATE} 5:${STEM}-s5-${DATE}"
echo "LOCK $CELL: row=$ROW step=$STEP"
SP=$SP scripts/cube_fixed_report.sh "$CELL" "$BASE" "$HZ" "$ROW" "$STEP" "$SEEDMAP"
