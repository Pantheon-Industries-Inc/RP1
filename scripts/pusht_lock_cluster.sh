#!/usr/bin/env bash
# PushT PLDM (cluster) lock-in: dump the tag's driver log from the volume, pick the shared pair (6 seeds x 36 pairs),
# launch six MODE=report jobs (draws 42-44), read the per-seed csv rows back, print the n=6 row.
#   scripts/pusht_lock_cluster.sh [TAG=rlp-pt-uniJ-h25-p-20260923] ; env WAIT=1 polls every 20 min until complete
set -uo pipefail
TAG=${1:-rlp-pt-uniJ-h25-p-20260923}; SP=${SP:-/tmp}; BASE=pldm; WM=${PT_PLDM_WM:-/newcheckpoints/armin@pantheon.inc/pusht-pldm-wm-20260920/wm_epoch19}
jid(){ grep -o "sky jobs logs [0-9]*" | grep -o "[0-9]*$" | head -1; }
wait_job(){ local id=$1 st=""; while :; do st=$(sky jobs queue --limit 400 2>/dev/null | awk -v j=$id '$1==j' | grep -oE 'SUCCEEDED|FAILED[A-Z_]*|CANCELLED' | head -1); [ -n "$st" ] && { echo "$st"; return; }; sleep 60; done; }
while :; do
  RID=$(sky jobs launch -y -d --name "read-${TAG#rlp-}" --env TAGS="$TAG" scripts/sky/tools/ckptval_read.yaml 2>&1 | jid); st=$(wait_job $RID); [ "$st" = SUCCEEDED ] || { echo "reader $RID $st"; exit 1; }
  sky jobs logs $RID --no-follow 2>&1 | perl -pe 's/^.*?rank=0\)\e\[0m //' > $SP/lock_${TAG}.dump
  DRIVER_DUMP=$SP/lock_${TAG}.dump scripts/collect_ckpt_select.sh 2>/dev/null | grep " $TAG \|^$TAG " > $SP/lock_${TAG}.sel
  grep ^SNAPH $SP/lock_${TAG}.sel > $SP/lock_${TAG}.snaph; grep -v ^SNAPH $SP/lock_${TAG}.sel > $SP/lock_${TAG}.dep
  PICK=$(python3 scripts/fixed_stop_select.py $SP/lock_${TAG}.snaph $SP/lock_${TAG}.dep 2>/dev/null | grep "^$TAG:" | grep -v WARNING); echo "$PICK"
  if echo "$PICK" | grep -q "6/6 seeds" && ! echo "$PICK" | grep -q PRELIMINARY; then break; fi
  [ "${WAIT:-0}" = 1 ] || { echo "not complete; WAIT=1 to poll"; exit 2; }; echo "[$(date -u +%H:%M)] not complete; polling in 20 min"; sleep 1200
done
T=$(echo "$PICK" | grep -oE "FIXED teacher [0-9]+" | grep -oE "[0-9]+$"); STEP=$(echo "$PICK" | grep -oE "actor [0-9a-z]+" | cut -d' ' -f2); [ "$STEP" = final ] && STEP=12000
echo "LOCK $TAG: row=t$T step=$STEP"
ENVS=pusht ONLY_BASES=$BASE DATE=${TAG##*-} PT_PLDM_WM=$WM PT_REPORT=t$T:$STEP bash scripts/sky/unigamma/launch_uniB.sh 2>&1 | grep -E "==>|Submitted"
sleep 60
# wait for the six report jobs (by name), then dump the csv rows via the reader (report/ dir is not in the reader -> read the driver log's [cell] lines instead)
while :; do n=$(sky jobs queue --limit 200 2>/dev/null | awk -v t="${TAG%-*}" '($1 ~ /^[0-9]+$/) {for(i=1;i<=NF;i++) if ($i ~ "^"t"-s[0-5]-report$") {for(k=1;k<=NF;k++) if ($k=="SUCCEEDED" || $k ~ /^FAILED/) c++}} END{print c+0}'); [ "${n:-0}" -ge 6 ] && break; sleep 120; done
RID=$(sky jobs launch -y -d --name "read-${TAG#rlp-}-rep" --env TAGS="$TAG" scripts/sky/tools/ckptval_read.yaml 2>&1 | jid); st=$(wait_job $RID)
sky jobs logs $RID --no-follow 2>&1 | perl -pe 's/^.*?rank=0\)\e\[0m //' > $SP/lock_${TAG}.rep
python3 - "$SP/lock_${TAG}.rep" "$TAG" "t$T" "$STEP" <<'PY'
import sys, re, statistics, collections
dump, tag, row, step = sys.argv[1:]; rows = collections.defaultdict(dict)
# [cell] rh5fixed_h25_s42 = 74.0   (unig_ctrl_a2.5_t3000 s0 step6000)   -- written to the driver log by MODE=report
for line in open(dump):
    m = re.search(r"\[cell\] rh5fixed_h25_s(\d+) = ([0-9.]+|FAIL)\s+\((\S+) s(\d) step(\d+)\)", line)
    if m and m.group(3).endswith(row) and m.group(5) == step and m.group(2) != "FAIL": rows[int(m.group(4))][int(m.group(1))] = float(m.group(2))
means = []
print(f"\n### {tag} (PushT PLDM h25): shared stop teacher {row[1:]} / actor {step} -- test draws 42-44")
for s in sorted(rows):
    v = [rows[s].get(k) for k in (42, 43, 44)]
    if None in v: print(f"  s{s}: incomplete {rows[s]}"); continue
    m = statistics.mean(v); means.append(m); print(f"  s{s}: {v[0]:.0f} / {v[1]:.0f} / {v[2]:.0f}  mean {m:.1f}")
print(f"  ==> n=6 test: median {statistics.median(means):.1f} mean {statistics.mean(means):.2f}" if len(means) == 6 else f"  ==> {len(means)}/6 seeds")
PY
