#!/usr/bin/env bash
# PushT shared-stop lock-in on a pod: collect the [ckpt-select] lines of TAG, pick the pair (six seeds x 36 pairs), launch
# six MODE=report jobs (draws 42-44) via the launcher's PT_REPORT mode (one per GPU), print the n=6 row.
#   POD=root@host:port HZ=25|100 BASE=lewm|pldm TAG=rlp-pt-uniJ-h<HZ>[-p]-<date> scripts/pusht_lock.sh   (WAIT=1 polls)
set -uo pipefail
POD=${POD:-root@69.30.85.162:22186}; HZ=${HZ:-25}; BASE=${BASE:-lewm}; SP=${SP:-/tmp}
P=""; [ "$BASE" = pldm ] && P="-p"; TAG=${TAG:-rlp-pt-uniJ-h${HZ}${P}-20260923}; DATE=${TAG##*-}
H=${POD%%:*}; PT=${POD##*:}
collect(){ ssh -o BatchMode=yes -o ConnectTimeout=20 -p "$PT" -i ~/.ssh/id_ed25519 "$H" "cat /mnt/raid0/rlp-pusht/armin@pantheon.inc/$TAG/grid_driver.log 2>/dev/null" 2>/dev/null; }
while :; do
  # reuse the collector's parser on the pod's driver log
  collect > $SP/pusht_${TAG}.drv; printf '=== DRIVER %s\n' "$TAG" > $SP/pusht_${TAG}.dump; cat $SP/pusht_${TAG}.drv >> $SP/pusht_${TAG}.dump; echo "=== END DRIVER" >> $SP/pusht_${TAG}.dump
  DRIVER_DUMP=$SP/pusht_${TAG}.dump scripts/collect_ckpt_select.sh 2>/dev/null | grep " $TAG \|^$TAG " > $SP/pusht_${TAG}.sel
  grep ^SNAPH $SP/pusht_${TAG}.sel > $SP/pusht_${TAG}.snaph; grep -v ^SNAPH $SP/pusht_${TAG}.sel > $SP/pusht_${TAG}.dep
  PICK=$(python3 scripts/fixed_stop_select.py $SP/pusht_${TAG}.snaph $SP/pusht_${TAG}.dep 2>/dev/null | grep "^$TAG:" | grep -v WARNING); echo "$PICK"
  if echo "$PICK" | grep -q "6/6 seeds" && ! echo "$PICK" | grep -q PRELIMINARY; then break; fi
  [ "${WAIT:-0}" = 1 ] || { echo "PushT grid $TAG not complete"; exit 2; }; echo "[$(date -u +%H:%M)] not complete; polling in 20 min"; sleep 1200
done
T=$(echo "$PICK" | grep -oE "FIXED teacher [0-9]+" | grep -oE "[0-9]+$"); STEP=$(echo "$PICK" | grep -oE "actor [0-9a-z]+" | cut -d' ' -f2); [ "$STEP" = final ] && STEP=12000
echo "LOCK $TAG: row=t$T step=$STEP"
ENVS=pusht HZ=$HZ ONLY_BASES=$BASE POD=$POD DATE=$DATE PT_REPORT=t$T:$STEP bash scripts/sky/unigamma/launch_uniB.sh 2>&1 | grep -E "==>|pod-runner|rror"
NAME=rlp-pt-uniJ-h${HZ}${P}
while :; do n=$(ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/id_ed25519 -p $PT $H "ls /root/podjobs/${NAME}-s*-report/exit_code 2>/dev/null | wc -l"); [ "${n:-0}" -ge 6 ] && break; sleep 120; done
ssh -o BatchMode=yes -o ConnectTimeout=25 -i ~/.ssh/id_ed25519 -p $PT $H "cat /root/podjobs/${NAME}-s*-report/exit_code | tr '\n' ' '; echo; cat /mnt/raid0/rlp-pusht/armin@pantheon.inc/$TAG/report/*_t${T}_s*_step${STEP}_rh5fixed.csv 2>/dev/null" > $SP/pusht_${TAG}.report.csv
python3 - $SP/pusht_${TAG}.report.csv "$TAG" "$T" "$STEP" "$HZ" <<'PY'
import sys, statistics, collections
f, tag, T, step, hz = sys.argv[1:]; rows = collections.defaultdict(dict)
for line in open(f):
    p = line.strip().split(",")
    if len(p) == 5 and p[4] != "FAIL": rows[p[1]][p[3]] = float(p[4])
means = []
print(f"### {tag} (PushT h{hz}): shared stop teacher {T} / actor {step} -- test draws 42-44")
for s in sorted(rows):
    v = [rows[s].get(f"e{k}") for k in (42, 43, 44)]
    if None in v: print(f"  {s}: incomplete {rows[s]}"); continue
    m = statistics.mean(v); means.append(m); print(f"  {s}: {v[0]:.0f} / {v[1]:.0f} / {v[2]:.0f}  mean {m:.1f}")
print(f"  ==> n=6 test: median {statistics.median(means):.1f} mean {statistics.mean(means):.2f}" if len(means) == 6 else f"  ==> {len(means)}/6 seeds")
PY
