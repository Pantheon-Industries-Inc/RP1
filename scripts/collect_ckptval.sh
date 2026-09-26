#!/usr/bin/env bash
# Per-snapshot, per-horizon selection scores from the yaml's ckptval logs
# ($PERSIST/logs/g_<row>_s<seed>_ckptval.log: one "[summary] {...}" per snapshot, in the order the
# yaml scores them: step2000, step4000, ..., final). Output:
#   SNAPH <tag> <row> <seed> <step|final> <h25|na> <h100|na>
# Pods are read over ssh; cluster tags need the ckptval logs fetched from the results volume first
# (scripts/sky/ckptval_read.yaml -> its job log), pass those files as extra args.
#   scripts/collect_ckptval.sh [cluster_ckptval_dump.txt ...]
set -u
parse_dump(){ # stdin: "=== <tag> <row> <seed>" header lines followed by that log's content
  awk '
    /^=== END/{ for (i=1;i<=n;i++){ st=(i==n)?"final":(2000*i); print "SNAPH", tag, row, seed, st, steps[i] } n=0; next }
    /^=== /{tag=$2; row=$3; seed=$4; n=0; next}
    /\[summary\] \{/{ n++; h25="na"; h100="na"
      if (match($0, /"mean_rh5_h25": [0-9.]+/)) { h25=substr($0, RSTART+16, RLENGTH-16) }
      if (match($0, /"mean_rh5_h100": [0-9.]+/)) { h100=substr($0, RSTART+17, RLENGTH-17) }
      steps[n]=h25 " " h100 }
  '
}
dump_pod(){ # $1 host:port  $2.. tags   (remote script via bash -s: no nested-quoting)
  local hp=$1; shift
  for t in "$@"; do
    ssh -o BatchMode=yes -o ConnectTimeout=20 -p "${hp##*:}" -i ~/.ssh/id_ed25519 "${hp%%:*}" "TAG=$t bash -s" 2>/dev/null <<'REMOTE'
for f in /checkpoints/armin@pantheon.inc/$TAG/logs/g_*_ckptval.log; do
  [ -f "$f" ] || continue
  b=$(basename "$f" _ckptval.log); row=${b#g_}; row=${row%_s*}; seed=${b##*_s}
  echo "=== $TAG $row $seed"; grep -oE '\[summary\] \{[^}]*\}' "$f"; echo "=== END"
done
REMOTE
  done | parse_dump
}
dump_pod root@69.30.85.162:22186 rlp-tw-uniJ-h100-20260921 rlp-tw-uniJ-h25-p-20260921 rlp-tw-uniJ-h100-p-s345-20260921
dump_pod root@216.81.245.23:44445 rlp-tw-uniJ-h100-p-20260921 rlp-tw-uniJ-h25-20260921
for f in "$@"; do parse_dump < "$f"; done
