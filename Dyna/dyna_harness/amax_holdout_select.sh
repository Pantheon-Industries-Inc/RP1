#!/bin/bash
# Held-out model selection for the amax sweep -- see ../DATA_SPLIT_POLICY.md
# § REQUIRED CHANGES #2.
#
# Selecting and reporting on the same draws is tuning on test: with n=150 binary
# trials SE ~ 2.9 pts, and a max over ~6 candidates buys ~3 pts of optimism.
# The fix costs zero compute because the per-draw cells already exist:
#   SELECT the winning amax on draw 42 alone;  REPORT it on draws 43+44 only.
#
# Still imperfect -- draws 43/44 remain inside LIP's ~2M-pair training pool, so
# this removes the SELECTION bias, not the no-holdout problem. Only the episode
# split (collect 0-7999 / eval 8000-9999) fixes the latter.
#
# Standalone by design: reads only the summary CSV, so it can run while a sweep
# is in flight or long after. Safe to re-run.
#
# Usage: amax_holdout_select.sh [SUMMARY_CSV] [AMAXES...]
set -u
SUM=${1:-/workspace/results/summary_amaxsweep.csv}
shift 2>/dev/null || true
AMAXES=${*:-"1.0 1.2 1.4 1.6 1.8 2.2 2.6 3.0 3.5"}
SEEDS="0 1 2"
[ -f "$SUM" ] || { echo "no summary csv at $SUM"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

mean(){ # mean EXPECTED_N cell...   -> NA unless exactly N usable cells are present.
  # Must check the count: empty cells vanish in "$*" word-splitting, so averaging
  # whatever survived would silently report a partial column as complete.
  local want=$1; shift
  awk -v want="$want" 'BEGIN{n=0;s=0
    for(i=1;i<ARGC;i++){v=ARGV[i]
      if(v==""||v=="FAIL"||v=="NA"){print "NA";exit}
      s+=v;n++}
    if(n!=want){print "NA";exit}
    printf "%.1f",s/n}' "$@"
}

echo "=== amax: held-out selection (select on draw 42, report on 43+44) ==="
echo "    csv: $SUM"
printf "  %-6s %-14s %-16s %s\n" amax "sel(42)" "held-out(43,44)" "all-3 (biased)"
best=""; bestsel=""
for am in $AMAXES; do
  t=${am/./}
  # per-seed cells, collected as arrays so empty cells stay countable
  sel=(); ho=(); allc=()
  for s in $SEEDS; do
    sel+=("$(sc amsw_a${t}_s${s}_e42)")
    ho+=("$(sc amsw_a${t}_s${s}_e43)" "$(sc amsw_a${t}_s${s}_e44)")
    allc+=("$(sc amsw_a${t}_s${s}_e42)" "$(sc amsw_a${t}_s${s}_e43)" "$(sc amsw_a${t}_s${s}_e44)")
  done
  S=$(mean 3 "${sel[@]}"); H=$(mean 6 "${ho[@]}"); A=$(mean 9 "${allc[@]}")
  [ "$S" = NA ] && [ "$H" = NA ] && continue          # amax not swept at all
  printf "  %-6s %-14s %-16s %s\n" "$am" "$S" "$H" "$A"
  if [ "$S" != NA ]; then
    if [ -z "$bestsel" ] || awk -v s="$S" -v b="$bestsel" 'BEGIN{exit !(s>b)}'; then
      bestsel=$S; best=$am
    fi
  fi
done

if [ -z "$best" ]; then echo "  no complete draw-42 column yet -- rerun when the sweep advances"; exit 0; fi
t=${best/./}
ho=(); for s in $SEEDS; do ho+=("$(sc amsw_a${t}_s${s}_e43)" "$(sc amsw_a${t}_s${s}_e44)"); done
H=$(mean 6 "${ho[@]}")
echo
echo "  SELECTED amax=$best   (best draw-42 score: $bestsel)"
echo "  >>> REPORT: $H   (3 seeds x held-out draws 43,44) <<<"
echo "  Do NOT report the all-3-draw mean for amax=$best -- draw 42 chose it."
echo "  Gap (all-3 minus held-out) is the visible selection bias for this winner."
