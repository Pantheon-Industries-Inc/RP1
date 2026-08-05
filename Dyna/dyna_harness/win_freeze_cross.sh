#!/bin/bash
# WINNER x FREEZE: is expand-weight underdosed?
#
# THE OBSERVATION THAT MOTIVATES THIS. --expand-weight only ever acts inside
# critic_step() (train_lip_ac.py:345-350), and critic_step() stops being called
# at freeze_at. Every measurement of the factorial winner so far was taken at
# --freeze-critic-frac 0.05, i.e. freeze_at = 300 of 6000 steps. So PLDM's
# +10.6 was bought with the expand term ACTIVE FOR 5% OF TRAINING and switched
# off for the other 95%. At frac 0.8 it runs for 4800 steps -- 16x the dose.
#
# The pair is not symmetric in this respect, which is what makes the ladder a
# clean instrument:
#   --replay-prob   acts in actor_step(), which runs all 6000 steps regardless
#                   of the freeze point. Dose is INDEPENDENT of the fraction.
#   --expand-weight acts in critic_step(), which runs only until freeze_at.
#                   Dose is PROPORTIONAL to the fraction.
# So sweeping the fraction with the winner flags on is, to first order, an
# expand-weight DOSE-RESPONSE curve with replay held fixed.
#
# THREE OUTCOMES, all informative:
#   rises with the fraction  -> expand was underdosed; the factorial found a
#                               real term but measured it at 1/16 strength, and
#                               the winner's true ceiling is above 83.3.
#   flat                     -> 300 steps of critic adaptation is already
#                               saturating; expand is a fast correction, not a
#                               slow one.
#   falls                    -> the term is only safe in small doses; long
#                               co-adaptation lets the actor bend the critic
#                               toward its own imagined states (the exact
#                               failure mode freezing was meant to prevent).
#
# DESIGN. 2x4: {plain, winner} x freeze_at {300, 2000, 3000, 4800}, on BOTH
# bases, 3 seeds x 3 draws each. The plain row comes from lewm_freeze_ladder.sh
# and pldm_freeze_ladder.sh, already running; this script adds the winner row.
# Two rungs are already banked and are NOT retrained:
#     LeWM   frac 0.05 winner = lewmwin2       90.00 (3 seeds)
#     PLDM   frac 0.05 winner = fx_batch256.replay50.expand10  83.3 (1 seed)
# The PLDM one has a single seed, so that rung IS rerun at 3 seeds (pwin05) --
# 83.3 is also the max of 16 factorial cells, so winner's curse applies to it
# and a seeded value is needed regardless.
#
# WIN_FLAGS is byte-identical to the take-2 arm, verified against the script
# that produced lewmwin2: --batch 256 --replay-prob 0.5 --expand-weight 1.0.
#
# 21 trains + 63 evals, sharing the 12-slot pool with the two plain ladders.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_pldmgrid_egl.csv
DRV=$L/driver_wincross.log
V2WM=/workspace/models/v2WM
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
LF1=/workspace/caches/v2_tr8000_fs1.pt;   LF5=/workspace/caches/v2_tr8000_fs5.pt
LTD=/workspace/metrics/v2_tr8000_TD.pt
QF1=/workspace/caches/pldm_tr8000_fs1.pt; QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
WIN_FLAGS="--batch 256 --replay-prob 0.5 --expand-weight 1.0"
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"; SEEDS="0 1 2"
LAMAX=1.6; PAMAX=4.5; NSLOT=12; NGPU=8; SLOTDIR=/tmp/frzslots
mkdir -p "$L" "$R" "$SLOTDIR"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"
acquire(){ local s; while :; do
    for s in $(seq 0 $((NSLOT-1))); do
      mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }
    done; sleep 3; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
# shared pool with both plain ladders -- never clear slots others may hold

# tag|base|frac
ARMS=(
  "lwin2k|lewm|0.3334" "lwin3k|lewm|0.5" "lwin48|lewm|0.8"
  "pwin05|pldm|0.05" "pwin2k|pldm|0.3334" "pwin3k|pldm|0.5" "pwin48|pldm|0.8"
)

log "=== WINNER x FREEZE CROSS (pid $$): ${#ARMS[@]} arms x 3 seeds ==="
log "  WIN_FLAGS: $WIN_FLAGS"
log "  banked: LeWM frac0.05 winner = lewmwin2 90.00 (3 seeds)"
for f in "$LF1" "$LF5" "$LTD" "$QF1" "$QF5" "$QTD" "$AH5"; do
  [ -e "$f" ] || die "missing $f"; done

# ------------------------------------------------------------------- P1 train
log "P1: ${#ARMS[@]}x3 trains (queued behind the plain ladders on the shared pool)"
for e in "${ARMS[@]}"; do
  IFS='|' read -r t base fr <<< "$e"
  case "$base" in
    lewm) wm=$V2WM; c5=$LF5; c1=$LF1; td=$LTD; am=$LAMAX ;;
    pldm) wm=$PLDM; c5=$QF5; c1=$QF1; td=$QTD; am=$PAMAX ;;
  esac
  for s in $SEEDS; do
    out=/workspace/actors/lip4_${t}_s${s}.pt
    [ -f "$out" ] && { log "  $t/s$s present"; continue; }
    slot=$(acquire); gpu=$(( slot % NGPU ))
    (
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$gpu timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$c5" --cache-td "$c1" --h5 "$AH5" --wm "$wm" --init-value "$td" \
        --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$am" \
        --actor-lr 3e-4 --actor-lr-final 3e-5 \
        --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
        --arch v4 --seed "$s" --freeze-critic-frac "$fr" $WIN_FLAGS \
        --out "$out" --out-value "${out%.pt}_value.pt" > "$L/wx_${t}_s${s}.log" 2>&1
      release "$slot"
    ) &
    log "  -> $t/s$s ($base, frac $fr, amax $am) on gpu $gpu (slot $slot)"
  done
done
wait
log "P1: trains done"

for e in "${ARMS[@]}"; do
  IFS='|' read -r t base fr <<< "$e"
  want=$(awk -v f="$fr" 'BEGIN{printf "%d", int(f*6000)}')
  got=$(grep -oE "^step [0-9]+: critic\+teacher frozen" "$L/wx_${t}_s0.log" \
        | grep -oE "[0-9]+" | head -1)
  [ "$got" = "$want" ] || die "$t froze at '$got', expected $want"
  log "  gate ok: $t froze at step $got"
done

# -------------------------------------------------------------------- P2 eval
ev(){ # slot arm base seed draw
  local slot=$1 arm=$2 base=$3 s=$4 d=$5 gpu=$(( $1 % NGPU )) nm="${2}_s${4}_e${5}"
  local pol; case "$base" in lewm) pol=$V2WM ;; *) pol=$PLDM ;; esac
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { release "$slot"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu timeout 7200 \
    python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$pol" solver=lip "solver.actor_path=/workspace/actors/lip4_${arm}_s${s}.pt" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  local sr=""
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
    sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
  log "  [gpu$gpu] $nm = ${sr:-FAIL}"
  release "$slot"
}
log "P2: 63 eval cells"
for e in "${ARMS[@]}"; do
  IFS='|' read -r t base fr <<< "$e"
  for s in $SEEDS; do for d in $DRAWS; do
    slot=$(acquire); ev "$slot" "$t" "$base" "$s" "$d" &
  done; done
done
wait
log "P2: evals done"

# The 2x4 card needs the PLAIN row from the two ladder scripts. Wait for them
# (they were launched first and are shorter), but do not block forever.
log "P3: waiting for the plain ladders so the card is complete"
T0=$(date +%s)
until grep -q "LEWM_FRZ_DONE" "$L/driver_lewmfrz.log" 2>/dev/null \
   && grep -q "PLDM_FRZ_DONE" "$L/driver_pldmfrz.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 7200 ] && { log "WARN: plain rows incomplete after 2h"; break; }
  sleep 120
done

# -------------------------------------------------------------------- P4 card
python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, math
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
mean = lambda xs: sum(xs)/len(xs)
def sd(xs):
    m = mean(xs); return math.sqrt(sum((x-m)**2 for x in xs)/max(len(xs)-1, 1))
def arm(a):
    o = []
    for s in (0, 1, 2):
        vs = [rows.get(f"{a}_s{s}_e{d}") for d in (42, 43, 44)]
        if all(v is not None for v in vs): o.append(mean(vs))
    return o
# base -> [(freeze_at, plain_tag, winner_tag)]
GRID = {
 "LeWM (v2WM, amax 1.6)": [(300,"lewmbase","lewmwin2"), (2000,"frz2k","lwin2k"),
                           (3000,"frz3k","lwin3k"),    (4800,"frz48","lwin48")],
 "PLDM (amax 4.5)":       [(300,"pfrz05","pwin05"),    (2000,"pfrz2k","pwin2k"),
                           (3000,"pfrz3k","pwin3k"),   (4800,"pfrz48","pwin48")],
}
print("=== WINNER x CRITIC-FREEZE, both bases (held-out 8000:10000, EGL, 3 seeds x 3 draws) ===")
print("    winner = --batch 256 --replay-prob 0.5 --expand-weight 1.0")
print("    freeze_at also sets expand-weight's ACTIVE WINDOW: the term runs in")
print("    critic_step(), which stops at freeze_at. 300 -> 5% dose, 4800 -> 80%.")
for base, rungs in GRID.items():
    print(f"\n  --- {base} ---")
    print(f"  {'freeze_at':>9s} {'expand dose':>11s} {'plain':>7s} {'winner':>7s} "
          f"{'delta':>7s} {'w-spread':>9s}")
    deltas = []
    for at, pt, wt in rungs:
        pv, wv = arm(pt), arm(wt)
        pm = f"{mean(pv):7.2f}" if pv else "      -"
        wm_ = f"{mean(wv):7.2f}" if wv else "      -"
        if pv and wv:
            d = mean(wv) - mean(pv); deltas.append((at, d, pv, wv)); ds = f"{d:+7.2f}"
        else: ds = "      -"
        sp = f"{max(wv)-min(wv):9.1f}" if wv else "        -"
        print(f"  {at:9d} {100*at/6000:10.0f}% {pm} {wm_} {ds} {sp}")
    if len(deltas) >= 2:
        lo, hi = deltas[0], deltas[-1]
        print(f"    winner benefit at {lo[0]} steps: {lo[1]:+.2f}   at {hi[0]} steps: {hi[1]:+.2f}"
              f"   trend {hi[1]-lo[1]:+.2f}")
        if len(lo[3]) == 3 and len(hi[3]) == 3:
            dd = [x-y for x, y in zip(hi[3], lo[3])]
            m, s = mean(dd), sd(dd); t = m/(s/math.sqrt(3)) if s > 0 else float("inf")
            print(f"    winner {hi[0]} vs {lo[0]} (paired): {m:+.2f} sd {s:.2f} t {t:.2f}"
                  f"  {'SIGNIFICANT' if abs(t) > 2.92 else 'not significant'}")
        best = max(deltas, key=lambda x: x[1])
        print(f"    largest winner benefit at freeze_at={best[0]} ({best[1]:+.2f})")
print("\n  READING. Winner benefit GROWING with the dose => expand was underdosed and")
print("  the factorial measured a real term at 1/16 strength; 83.3 is not its ceiling.")
print("  FLAT => 300 steps already saturates it, and it is a fast correction.")
print("  SHRINKING/negative => the term is only safe in small doses -- long")
print("  co-adaptation lets the actor bend the critic toward its own imagined")
print("  states, which is precisely what the early freeze was preventing.")
print("  Note replay-prob's dose does NOT scale with the fraction (it acts in")
print("  actor_step, all 6000 steps), so the trend is attributable to expand.")
print("  PLDM refs: latent+CEM 66.7 | TD+CEM 73.3 (75.3 @ g.98) | LIP 20-seed 72.8")
print("  LeWM refs: LIP historical 87.8 (3 seeds) / 87.1 (6 seeds)")
print("  3 seeds resolves ~3 pts at best; the PLDM 20-seed spread was 68.0-79.3.")
print("WIN_FREEZE_CROSS_DONE")
PY
log "done"
