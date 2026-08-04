"""Fix the shared-knob selection in run_joint_sweep.sh.

The user's constraint: LIP knobs (amax, actor-lr) may differ per base, but TD /
dataset knobs (replay-prob, expand-weight, expectile) must be SHARED by PLDM and
LeWM. Stage 1 violated that -- it probed replay/expand on PLDM alone and took
PLDM's local best, ignoring lejepa's already-measured numbers for the same 2x2.

That mattered, because the two bases disagree sharply:

    replay / expand      lejepa    pldm    cross-base mean
    0     / 0             56.7     18.7        37.7
    0.5   / 0             43.3     31.7        37.5
    0     / 1.0           25.0     31.7        28.4
    0.5   / 1.0           20.0     34.0        27.0      <- what stage 1 picked

PLDM likes value expansion at both replay levels; lejepa is destroyed by it. The
picked cell is the WORST of the four cross-base, so stage 2 trained every lejepa
cell in its catastrophic regime.

Two changes:

1. EXPAND-WEIGHT is fixed to the cross-base winner (0). The gap is decisive
   (37.7/37.5 vs 28.4/27.0). This is a real cost to PLDM -- it would rather have
   1.0 -- and that cost is the honest price of the shared-knob constraint, so it
   gets reported rather than tuned away.

2. REPLAY-PROB is NOT decided here. 37.7 vs 37.5 is a tie at n=3 training seeds
   x 2 selection seeds, and a coin-flip on the shared axis is exactly the kind of
   choice that should be resolved by the full sweep rather than by a small probe.
   Both values become a shared axis of stage 2, and stage 3 picks the shared
   (expectile, replay) pair by cross-base mean -- same rule, more evidence.

Stage 2 therefore goes 2 bases x 2 expectile x 2 amax x 2 lr x 2 replay x 3
seeds = 96 trainings. The extra compute is spent on the one axis where the
evidence is genuinely ambiguous.
"""

import re
import subprocess

P = "/workspace/run_joint_sweep.sh"
s = open(P).read()

if "CROSS-BASE MEAN" in s:
    print("already patched")
    raise SystemExit

# ---------------------------------------------------------------- stage 1 selection
OLD1 = '''BEST_RP=0; BEST_EW=0; BESTV=0
for rp in 0 0.5; do for ew in 0 1.0; do
  t="p_r${rp//./}e${ew//./}"; v=$(cfgmean "jp_${t}")
  log "  PLDM replay ${rp} / expand ${ew}: ${v}"
  awk "BEGIN{exit !($v > $BESTV)}" && { BESTV=$v; BEST_RP=$rp; BEST_EW=$ew; }
done; done
log "SHARED replay/expand = ${BEST_RP} / ${BEST_EW} (PLDM best ${BESTV}); lejepa best was 0/0"
[ "$BEST_RP" = "0" ] && [ "$BEST_EW" = "0" ] && log "  -> both bases agree: OGBench settings do NOT transfer to reacher"'''

NEW1 = '''# lejepa measured the identical 2x2 under the identical protocol
# (results/summary_rxe_lejepa.csv). The knob is SHARED, so it must be scored by
# the CROSS-BASE MEAN -- PLDM's local best is not admissible evidence on its own.
lejepa_rxe(){ case "$1_$2" in
  0_0)     echo 56.7;; 0_1.0)   echo 25.0;;
  0.5_0)   echo 43.3;; 0.5_1.0) echo 20.0;; *) echo 0;; esac; }

log "--- replay x expand, scored CROSS-BASE MEAN (the knob is shared) ---"
BEST_EW=0; BEST_EWV=0
for ew in 0 1.0; do
  tot=0; k=0
  for rp in 0 0.5; do
    t="p_r${rp//./}e${ew//./}"; vp=$(cfgmean "jp_${t}"); vl=$(lejepa_rxe "$rp" "$ew")
    m=$(awk "BEGIN{printf \\"%.1f\\", ($vp+$vl)/2}")
    log "  replay ${rp} / expand ${ew}: pldm ${vp} | lejepa ${vl} | cross-base ${m}"
    tot=$(awk "BEGIN{print $tot+$m}"); k=$((k+1))
  done
  em=$(awk "BEGIN{printf \\"%.1f\\", ($k ? $tot/$k : 0)}")
  awk "BEGIN{exit !($em > $BEST_EWV)}" && { BEST_EWV=$em; BEST_EW=$ew; }
done
log "SHARED expand-weight = ${BEST_EW} (cross-base ${BEST_EWV})"
log "  NOTE: PLDM alone prefers expand 1.0 at BOTH replay levels; lejepa is"
log "  destroyed by it. The shared constraint costs PLDM here -- report it."

# replay is a tie cross-base (37.7 vs 37.5 at n=3x2), so it is NOT decided by
# this probe: both values enter stage 2 as a shared axis and stage 3 picks the
# (expectile, replay) pair by the same cross-base rule.
RP_LIST="0 0.5"
log "SHARED replay = undecided by probe (tie); carrying {${RP_LIST}} into stage 2"'''

assert s.count(OLD1) == 1, f"stage1 anchor x{s.count(OLD1)}"
s = s.replace(OLD1, NEW1)

# ---------------------------------------------------------------- stage 2: add replay axis
OLD2 = '''for B in lejepa pldm; do for EX in 005 01; do for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  for sd in 0 1 2; do
    t="${B}_e${EX}a${am//./}l${lr//e-/e}"
    train "$B" /workspace/metrics/window3_${B}_e${EX}.pt "$am" "$lr" "$BEST_RP" "$BEST_EW" "$sd" \\
          /workspace/actors/lip4_js_${t}_s${sd}.pt $((i % 6)) &
    i=$((i + 1)); [ $((i % 18)) -eq 0 ] && wait
  done
done; done; done; done'''

NEW2 = '''for B in lejepa pldm; do for EX in 005 01; do for rp in $RP_LIST; do
for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  for sd in 0 1 2; do
    t="${B}_e${EX}a${am//./}l${lr//e-/e}r${rp//./}"
    train "$B" /workspace/metrics/window3_${B}_e${EX}.pt "$am" "$lr" "$rp" "$BEST_EW" "$sd" \\
          /workspace/actors/lip4_js_${t}_s${sd}.pt $((i % 6)) &
    i=$((i + 1)); [ $((i % 18)) -eq 0 ] && wait
  done
done; done; done; done; done'''

assert s.count(OLD2) == 1, f"stage2 train anchor x{s.count(OLD2)}"
s = s.replace(OLD2, NEW2)

OLD3 = '''for B in lejepa pldm; do for EX in 005 01; do for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  t="${B}_e${EX}a${am//./}l${lr//e-/e}"
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_js_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $((i % 6)) "$B" "js_${t}_s${sd}_sel${s}" $s "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done; done; done'''

NEW3 = '''for B in lejepa pldm; do for EX in 005 01; do for rp in $RP_LIST; do
for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  t="${B}_e${EX}a${am//./}l${lr//e-/e}r${rp//./}"
  ( for sd in 0 1 2; do
      A=/workspace/actors/lip4_js_${t}_s${sd}.pt; [ -f "$A" ] || continue
      for s in $SEL; do ev $((i % 6)) "$B" "js_${t}_s${sd}_sel${s}" $s "$A"; done
    done ) &
  i=$((i + 1)); [ $((i % 6)) -eq 0 ] && wait
done; done; done; done; done'''

assert s.count(OLD3) == 1, f"stage2 eval anchor x{s.count(OLD3)}"
s = s.replace(OLD3, NEW3)

# ---------------------------------------------------------------- stage 3: screen rows carry replay
OLD4 = '''for B in lejepa pldm; do for EX in 005 01; do for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  t="${B}_e${EX}a${am//./}l${lr//e-/e}"; v=$(cfgmean "js_${t}")
  [ "$v" = "0.0" ] && continue
  echo "$v $B $EX $am $lr $t" >> /workspace/results/joint_screen.txt
  log "  ${B} expectile${EX} amax${am} lr${lr}: ${v}"
done; done; done; done'''

NEW4 = '''for B in lejepa pldm; do for EX in 005 01; do for rp in $RP_LIST; do
for am in 1.8 2.2; do for lr in 3e-4 1e-3; do
  t="${B}_e${EX}a${am//./}l${lr//e-/e}r${rp//./}"; v=$(cfgmean "js_${t}")
  [ "$v" = "0.0" ] && continue
  echo "$v $B $EX $am $lr $rp $t" >> /workspace/results/joint_screen.txt
  log "  ${B} expectile${EX} replay${rp} amax${am} lr${lr}: ${v}"
done; done; done; done; done'''

assert s.count(OLD4) == 1, f"stage3 screen anchor x{s.count(OLD4)}"
s = s.replace(OLD4, NEW4)

# ---------------------------------------------------------------- stage 3: joint (expectile, replay) choice
OLD5 = '''BEST_EX=""; BEST_EXV=0
for EX in 005 01; do
  tot=0; k=0
  for B in lejepa pldm; do
    b=$(awk -v b="$B" -v e="$EX" '$2==b && $3==e {print $1}' /workspace/results/joint_screen.txt | sort -rn | head -1)
    [ -z "$b" ] && continue
    tot=$(awk "BEGIN{print $tot+$b}"); k=$((k+1))
  done
  [ "$k" -eq 0 ] && continue
  m=$(awk "BEGIN{printf \\"%.1f\\", $tot/$k}")
  log "  shared expectile ${EX}: cross-base mean of per-base bests = ${m}"
  awk "BEGIN{exit !($m > $BEST_EXV)}" && { BEST_EXV=$m; BEST_EX=$EX; }
done
log "SHARED expectile = ${BEST_EX} (cross-base ${BEST_EXV})"'''

NEW5 = '''# Both expectile and replay are SHARED, so they are chosen together as a pair,
# by the cross-base mean of each base's best (amax, lr) at that pair. amax/lr
# stay free per base -- that is the only asymmetry the user allowed.
BEST_EX=""; BEST_RP=""; BEST_EXV=0
for EX in 005 01; do for rp in $RP_LIST; do
  tot=0; k=0
  for B in lejepa pldm; do
    b=$(awk -v b="$B" -v e="$EX" -v r="$rp" '$2==b && $3==e && $6==r {print $1}' \\
        /workspace/results/joint_screen.txt | sort -rn | head -1)
    [ -z "$b" ] && continue
    tot=$(awk "BEGIN{print $tot+$b}"); k=$((k+1))
  done
  [ "$k" -lt 2 ] && { log "  pair expectile${EX}/replay${rp}: incomplete (${k}/2 bases)"; continue; }
  m=$(awk "BEGIN{printf \\"%.1f\\", $tot/$k}")
  log "  shared pair expectile${EX} replay${rp}: cross-base mean of per-base bests = ${m}"
  awk "BEGIN{exit !($m > $BEST_EXV)}" && { BEST_EXV=$m; BEST_EX=$EX; BEST_RP=$rp; }
done; done
log "SHARED expectile = ${BEST_EX}, replay = ${BEST_RP}, expand = ${BEST_EW} (cross-base ${BEST_EXV})"'''

assert s.count(OLD5) == 1, f"stage3 pair anchor x{s.count(OLD5)}"
s = s.replace(OLD5, NEW5)

# ---------------------------------------------------------------- stage 3: winner row indices shift
OLD6 = '''  line=$(awk -v b="$B" -v e="$BEST_EX" '$2==b && $3==e' /workspace/results/joint_screen.txt | sort -rn | head -1)
  [ -z "$line" ] && { log "no winner for ${B}"; continue; }
  set -- $line; sv=$1; am=$4; lr=$5; t=$6'''

NEW6 = '''  line=$(awk -v b="$B" -v e="$BEST_EX" -v r="$BEST_RP" '$2==b && $3==e && $6==r' \\
         /workspace/results/joint_screen.txt | sort -rn | head -1)
  [ -z "$line" ] && { log "no winner for ${B}"; continue; }
  set -- $line; sv=$1; am=$4; lr=$5; t=$7'''

assert s.count(OLD6) == 1, f"stage3 winner anchor x{s.count(OLD6)}"
s = s.replace(OLD6, NEW6)

open(P, "w").write(s)
r = subprocess.run(["bash", "-n", P], capture_output=True, text=True)
assert r.returncode == 0, f"syntax error:\n{r.stderr}"
n = len(re.findall(r"RP_LIST", s))
print(f"patched: expand fixed cross-base, replay carried as shared axis "
      f"(RP_LIST x{n}); bash -n clean")
print("stage 2 is now 2 bases x 2 expectile x 2 replay x 2 amax x 2 lr x 3 seeds "
      "= 96 trainings")
