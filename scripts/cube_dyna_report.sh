#!/usr/bin/env bash
# Read a fixed-config cell (Dyna retrain) off the volume: per-seed validation (draws 48-51) and test (42-44) means
# of the deployed actors' report evals (<tag>/logs/g_<row>_s<seed>_eval.log), n=6 medians.
#   scripts/cube_dyna_report.sh <tag> <hz>          e.g. rlp-cu-uniJ-h25-dyna1-20260923 25
set -uo pipefail; TAG=$1; HZ=$2; SP=${SP:-/tmp}
jid(){ grep -o "sky jobs logs [0-9]*" | grep -o "[0-9]*$" | head -1; }
wait_job(){ local id=$1 st=""; while :; do st=$(sky jobs queue --limit 400 2>/dev/null | awk -v j=$id '$1==j' | grep -oE 'SUCCEEDED|FAILED[A-Z_]*|CANCELLED' | head -1); [ -n "$st" ] && { echo "$st"; return; }; sleep 60; done; }
WSARG=""; [ -n "${WORKSPACE:-}" ] && WSARG="--workspace $WORKSPACE"
RID=$(sky jobs launch -y -d $WSARG --name "read-${TAG#rlp-cu-uniJ-}" --env TAGS="$TAG" scripts/sky/tools/ckptval_read.yaml 2>&1 | jid); st=$(wait_job $RID); [ "$st" = SUCCEEDED ] || { echo "reader $RID $st"; exit 1; }
sky jobs logs $RID --no-follow 2>&1 | perl -pe 's/^.*?rank=0\)\e\[0m //' > $SP/dyna_${TAG}.dump
python3 - "$SP/dyna_${TAG}.dump" "$HZ" "$TAG" <<'PY'
import sys, re, json, statistics
dump, hz, tag = sys.argv[1:]; res = {}; cur = None
for line in open(dump):
    m = re.match(r"^=== EVAL (\S+) (\S+) (\d+)", line)
    if m: cur = (m.group(2), int(m.group(3))); continue
    if line.startswith("=== END EVAL"): cur = None; continue
    if cur and "[summary]" in line:
        js = json.loads(line[line.index("{"):line.rindex("}")+1]); res[cur] = js
val, test = [], []
print(f"### {tag} (h{hz}) fixed config, per seed: val(48-51) | test(42-44)")
for (row, s) in sorted(res, key=lambda k: k[1]):
    js = res[(row, s)]
    v = [js.get(f"rh5_h{hz}_s{k}") for k in (48, 49, 50, 51)]; t = [js.get(f"rh5_h{hz}_s{k}") for k in (42, 43, 44)]
    if None in v or None in t: print(f"  {row} s{s}: incomplete"); continue
    val.append(statistics.mean(v)); test.append(statistics.mean(t))
    print(f"  {row} s{s}: val {statistics.mean(v):.1f} | test {t[0]:.0f}/{t[1]:.0f}/{t[2]:.0f} mean {statistics.mean(t):.1f}")
if len(test) == 6:
    print(f"  ==> n=6: val mean {statistics.mean(val):.2f} | test median {statistics.median(test):.1f} mean {statistics.mean(test):.2f}")
else: print(f"  ==> {len(test)}/6 seeds")
PY
