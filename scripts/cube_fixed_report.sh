#!/usr/bin/env bash
# Cube shared-stop lock-in: stage the chosen (teacher, actor-step) snapshot of ALL six seeds into one import tag,
# evaluate them on the report draws 42-44 (REUSE_ONLY import job, EVAL_TAG=rh5fixed, no training), read the
# per-seed [summary] json back from the volume and print the n=6 row.
#   scripts/cube_fixed_report.sh <cell> <base> <hz 25|100> <row> <step|final> "<seed>:<srctag> ..." [DYNA_WM=<dir> DYNA_IT=<k>]
#   e.g. scripts/cube_fixed_report.sh rlp-cu-uniJ-h100-20260921 lewm 100 unig_ctrl_a2.5_t9000 2000 \
#          "0:rlp-cu-uniJ-h100-20260921 1:rlp-cu-uniJ-h100-20260921 2:rlp-cu-uniJ-h100-20260921 3:rlp-cu-uniJ-h100-s345-20260921 4:rlp-cu-uniJ-h100-s345-20260921 5:rlp-cu-uniJ-h100-s5-20260921"
# Every seed is re-evaluated (uniform provenance eval/rh5fixed), even where the per-seed winner was the same snapshot.
set -uo pipefail
CELL=$1; BASE=$2; HZ=$3; ROW=$4; STEP=$5; SEEDMAP=$6
IMPORT="${CELL}-fixedimport"; DATE=${DATE:-$(echo "$CELL" | grep -oE '[0-9]{8}$')}
SP=${SP:-/tmp}; LOG=$SP/cube_fixed_${CELL}.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
jid(){ grep -o "sky jobs logs [0-9]*" | grep -o "[0-9]*$" | head -1; }
wait_job(){ local id=$1 st=""; while :; do st=$(sky jobs queue --limit 400 2>/dev/null | awk -v j=$id '$1==j' | grep -oE 'SUCCEEDED|FAILED[A-Z_]*|CANCELLED' | head -1); [ -n "$st" ] && { echo "$st"; return; }; sleep 60; done; }
ITEMS=""; SEEDS=""
for kv in $SEEDMAP; do s=${kv%%:*}; t=${kv##*:}; ITEMS="$ITEMS $t:cube:$BASE:$ROW:$s:$STEP"; SEEDS="$SEEDS $s"; done
log "cell=$CELL base=$BASE h$HZ row=$ROW step=$STEP seeds=$SEEDS -> import $IMPORT"
SID=$(sky jobs launch -y -d --name "fixed-stage-${CELL#rlp-cu-uniJ-}" --env ITEMS="$ITEMS" --env DEST_TAG="$IMPORT" scripts/sky/tools/fixed_stage.yaml 2>&1 | jid)
[ -n "$SID" ] || { log "stage launch failed"; exit 1; }; log "stage job $SID"; st=$(wait_job $SID); log "stage $st"; [ "$st" = SUCCEEDED ] || exit 1
sky jobs logs $SID --no-follow 2>/dev/null | grep -E "\[stage\]" | sed 's/^.*\[stage\]/[stage]/' > $SP/stage_${CELL}.txt; cat $SP/stage_${CELL}.txt >> "$LOG"
NMISS=$(grep -c MISSING $SP/stage_${CELL}.txt || true); [ "${NMISS:-0}" = 0 ] || { log "staging reported $NMISS MISSING file(s)"; exit 1; }
NST=$(grep -c " -> " $SP/stage_${CELL}.txt || true); log "staged $NST actor file(s)"
# report job via the launcher's FIXED_ROW mode (envs from recipes/uniJ.yaml; gamma per stack)
BASES_ARG=$BASE; SFX_ARG=""
RID=$(ENVS=cube HZ=$HZ ONLY_BASES=$BASE GPUS=4 DATE=$DATE FIXED_ROW=$ROW FIXED_IMPORT=$IMPORT FIXED_SEEDS="$(echo $SEEDS)" \
      DYNA_WM=${DYNA_WM:-} DYNA_IT=${DYNA_IT:-} bash scripts/sky/unigamma/launch_uniB.sh 2>&1 | tee -a "$LOG" | jid)
if [ -z "$RID" ]; then   # fallback: the launcher output was filtered -- find the newest job by name
  sleep 20; RID=$(sky jobs queue --limit 100 2>/dev/null | awk '($1 ~ /^[0-9]+$/) {for(i=1;i<=NF;i++) if ($i ~ /^rlp-cu-uniJ-.*-fixed$/) print $1, $i}' | grep -E " rlp-cu-uniJ-h${HZ}(-p)?(-dyna[0-9]+)?-fixed$" | sort -n | tail -1 | cut -d' ' -f1)
fi
[ -n "$RID" ] || { log "report launch failed (no job id)"; exit 1; }; log "report job $RID"; st=$(wait_job $RID); log "report $st"
FT=$(ENVS=cube HZ=$HZ ONLY_BASES=$BASE DATE=$DATE FIXED_ROW=$ROW FIXED_IMPORT=$IMPORT DRY=1 DYNA_WM=${DYNA_WM:-} DYNA_IT=${DYNA_IT:-} bash scripts/sky/unigamma/launch_uniB.sh 2>&1 | grep -oE "EXPERIMENT_TAG=rlp-cu-uniJ[^ ]*-fixed-[0-9]+" | tail -1 | cut -d= -f2)
# read the per-seed [summary] json from the volume
DID=$(sky jobs launch -y -d --name "read-${FT#rlp-cu-uniJ-}" --env TAGS="$FT" scripts/sky/tools/ckptval_read.yaml 2>&1 | jid); st=$(wait_job $DID); log "reader $DID $st"
sky jobs logs $DID --no-follow 2>&1 | perl -pe 's/^.*?rank=0\)\e\[0m //' > $SP/fixed_${CELL}.dump
python3 - "$SP/fixed_${CELL}.dump" "$HZ" "$CELL" "$ROW" "$STEP" <<'PY'
import sys, re, json, statistics
dump, hz, cell, row, step = sys.argv[1:]
res = {}
cur = None
for line in open(dump):
    m = re.match(r"^=== EVAL (\S+) (\S+) (\d+)", line)
    if m: cur = (m.group(2), int(m.group(3))); continue
    if line.startswith("=== END EVAL"): cur = None; continue
    if cur and "[summary]" in line:
        js = json.loads(line[line.index("{"):line.rindex("}")+1])
        res[cur[1]] = {k: v for k, v in js.items() if k.startswith(f"rh5_h{hz}_s")}
print(f"\n### {cell}: shared stop {row} / actor {step} (h{hz}) -- test draws 42-44 (eval/rh5fixed)")
means = []
for s in sorted(res):
    d = res[s]; vals = [d.get(f"rh5_h{hz}_s{k}") for k in (42, 43, 44)]
    if None in vals: print(f"  s{s}: incomplete {d}"); continue
    m = statistics.mean(vals); means.append(m)
    print(f"  s{s}: {vals[0]:.0f} / {vals[1]:.0f} / {vals[2]:.0f}  mean {m:.1f}")
if len(means) == 6:
    print(f"  ==> n=6 test: median {statistics.median(means):.1f}  mean {statistics.mean(means):.2f}")
else:
    print(f"  ==> only {len(means)}/6 seeds evaluated")
PY
