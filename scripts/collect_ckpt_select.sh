#!/usr/bin/env bash
# Gather the yaml's "[ckpt-select] ... winner|final wins (val X)" lines = the DEPLOYED snapshot's
# selection score per (tag, row, seed), from pods (ssh) and cluster jobs (sky jobs logs).
# Output lines: <tag> <row> <seed> <val> <deployed-step>   and   SNAPH <tag> <row> <seed> <step|final> <h25|na> <h100|na>
# (step is taken from the snapshot FILENAME -- never from the order of the ckptval log, which sorts
#  lexicographically: step10000 < step2000 ... and mislabelled the 12k-cap runs on 2026-09-22)
#   scripts/collect_ckpt_select.sh > /tmp/ckpt_select.txt
set -u
parse(){ # $1 = tag ; stdin = log.  Emits:  <tag> <row> <seed> <val>          (deployed snapshot)
  #                                  and:  SNAP <tag> <row> <seed> <step|final> <val>  (every snapshot)
  local t=$1; sed 's/.*rank=0)..//' | grep -E "\[ckpt-select\]" | while IFS= read -r l; do
    case "$l" in
      *" winner "*) echo "$l" | sed -E "s/.*\[ckpt-select\] ((unig_ctrl_a2\.5|rs_x0_r0)(_t[0-9]+)?) s([0-9]+) winner [^ ]*(_step|\.step)([0-9]+)\.pt \(val ([0-9.]+)\).*/$t \1 \4 \7 \6/";;
      *"final wins"*) echo "$l" | sed -E "s/.*\[ckpt-select\] ((unig_ctrl_a2\.5|rs_x0_r0)(_t[0-9]+)?) s([0-9]+) final wins \(val ([0-9.]+)\).*/$t \1 \4 \5 final/";;
      *" val="*) echo "$l" | sed -E "s/.*\[ckpt-select\] ((unig_ctrl_a2\.5|rs_x0_r0)(_t[0-9]+)?) s([0-9]+) ([^ ]+) val=([0-9.a-z]+)( h25=([0-9.a-z]+) h100=([0-9.a-z]+))?.*/$t \1 \4 \5 \6 \8 \9/" \
        | awk '{ f=$4; st="final"; if (match(f, /(_step|\.step)[0-9]+/)) { st=substr(f, RSTART, RLENGTH); sub(/^(_step|\.step)/, "", st) }
                 h25=$6; h100=$7; if (h25=="" || h25=="na") h25="na"; if (h100=="" || h100=="na") h100="na";
                 # legacy (no h25/h100 fields): the single val is the h25 score for reacher/-h25 tags, else mean_all (unusable per horizon)
                 if ($6=="" ) { if ($1 ~ /^rlp-re-/ || $1 ~ /-h25/) h25=$5; }
                 # PushT rows written before 2026-09-23 16:30Z carry val x100 (success_rate was already a percentage)
                 if ($1 ~ /^rlp-pt-/) { if (h25!="na" && h25+0>100) h25=h25/100; if ($5+0>100) $5=$5/100 }
                 print "SNAPH", $1, $2, $3, st, h25, h100 }';;
    esac
  done
}
# pods: <host:port> <jobname>=<tag> ...
for spec in "root@69.30.85.162:22186 rlp-tw-uniJ-h100=rlp-tw-uniJ-h100-20260921 rlp-tw-uniJ-h25-p=rlp-tw-uniJ-h25-p-20260921 rlp-tw-uniJ-h100-p-s345=rlp-tw-uniJ-h100-p-s345-20260921" \
            "root@216.81.245.23:44445 rlp-tw-uniJ-h100-p=rlp-tw-uniJ-h100-p-20260921 rlp-tw-uniJ-h25=rlp-tw-uniJ-h25-20260921"; do
  set -- $spec; hp=$1; shift
  for jt in "$@"; do j=${jt%%=*}; t=${jt##*=}
    # the persistent grid_driver.log survives relaunches of the same tag (run.log is overwritten by launch.sh)
    ssh -o BatchMode=yes -o ConnectTimeout=20 -p "${hp##*:}" -i ~/.ssh/id_ed25519 "${hp%%:*}" \
      "if [ -f /checkpoints/armin@pantheon.inc/$t/grid_driver.log ]; then cat /checkpoints/armin@pantheon.inc/$t/grid_driver.log; else cat /root/podjobs/$j/run.log 2>/dev/null; fi" 2>/dev/null | parse "$t"
  done
done
# PushT on the old pod (scripts/sky/pusht_uniJ.yaml): OUT_ROOT=/mnt/raid0/rlp-pusht; same [ckpt-select] line format
for spec in "root@69.30.85.162:22186 rlp-pt-uniJ-h25-20260923"; do
  set -- $spec; hp=$1; shift
  for t in "$@"; do
    ssh -o BatchMode=yes -o ConnectTimeout=20 -p "${hp##*:}" -i ~/.ssh/id_ed25519 "${hp%%:*}" "cat /mnt/raid0/rlp-pusht/armin@pantheon.inc/$t/grid_driver.log 2>/dev/null" 2>/dev/null | parse "$t"
  done
done
# cluster: preferred = a driver dump from scripts/sky/tools/ckptval_read.yaml
#   sky jobs logs <reader-id> --no-follow | perl -pe 's/^.*?rank=0\)\e\[0m //' > dump.txt ; DRIVER_DUMP=dump.txt scripts/collect_ckpt_select.sh
# (sections "=== DRIVER <tag>" ... "=== END DRIVER"); fallback = `sky jobs logs` per job id (slow, can hang).
if [ -n "${DRIVER_DUMP:-}" ]; then
  awk '/^=== DRIVER /{t=$3; next} /^=== END DRIVER/{t=""; next} t!=""{print t"\t"$0}' "$DRIVER_DUMP" | while IFS=$'\t' read -r t l; do echo "$l" | parse "$t"; done
else
# cluster: <jobid>=<tag>
for jt in 26813=rlp-cu-uniJ-h100-20260921 26814=rlp-cu-uniJ-h100-p-20260921 26815=rlp-cu-uniJ-h100-s345-20260921 26816=rlp-cu-uniJ-h100-p-s345-20260921 \
          26817=rlp-cu-uniJ-h25-20260921 26818=rlp-cu-uniJ-h25-p-20260921 26819=rlp-cu-uniJ-h25-s345-20260921 26820=rlp-cu-uniJ-h25-p-s345-20260921 \
          26840=rlp-re-uniJ-h25-20260922 26842=rlp-re-uniJ-h25-p-20260922 26843=rlp-re-uniJ-h25-s345-20260922 26844=rlp-re-uniJ-h25-p-s345-20260922; do
  j=${jt%%=*}; t=${jt##*=}
  sky jobs logs "$j" --no-follow 2>/dev/null | parse "$t"
done
fi
