#!/bin/bash
# PHASE A: replicate every min-objective arm to 3 seeds.
# PHASE B: 2-D replay-curriculum grid at 3 seeds.
# 6000 steps, LeWM, I3 argmin readout at rh=1, rh=5 regression column.
#
# WHY THIS SHAPE. The 20-arm screen resolved exactly one thing: the objective.
# All ten `min` arms landed 83.33-88.00 at rh=1 while nine of ten `last` arms
# landed 78.67-85.33 -- a ~4-point systematic split against a 2.43 sd across
# arms. NO hyper was resolved: every hyper delta was 1-3 points against a 1-seed
# se of ~3.5, and min_mw03 proved it by falling from 86.00 (1 seed) to
# 83.33 +/- 2.67 (3 seeds). So phase A replicates the whole OFAT under the
# winning objective, and phase B sweeps only the axis that showed signal.
#
# WHY THE REPLAY AXIS. Under `min` the screen ordered replay-prob monotonically
# (0.25 -> 88.00, 0.5 -> 86.00, 0.75 -> 83.33) and preferred stride 1 over 5
# (87.33 vs 86.00). Both are the replay curriculum, which is mechanistically the
# right place to look: at rh=1 the deployed replan queries a state 5 primitive
# steps in, so how often mid-task states are injected and at what stride is
# exactly what should matter. amax, expand-weight, mean-weight and iters all
# looked flat or negative and are NOT swept -- they get replication only.
#
# NOTE --replay-stride 0 (default, legacy tr[:, -3:]) is IDENTICAL to 5 by
# construction, so the grid's stride-5 column reuses the phase-A arms directly.
#
# 26 actors (~50 min each, 4 at a time => ~5.4 h) + ~160 cells (~40 min).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_repgrid.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
WM=/workspace/models/v2WM
TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
STEPS=6000; DRAWS="42 43 44"
NGPU=4; SLOTDIR=/tmp/rgslots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
NSLOT=4
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 10; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd "$CODE"

python3 - <<'PY' || exit 1
import subprocess, sys, inspect
sys.path.insert(0, "/workspace/code/stable-worldmodel")
from stable_worldmodel.solver import LIPSolver
h = subprocess.run([sys.executable, "/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac.py",
                    "--help"], capture_output=True, text=True).stdout
ok = ("align_mode" in inspect.signature(LIPSolver.__init__).parameters
      and "--term-index" in h and "--replay-stride" in h)
print("[preflight] argmin readout + trainer levers:", ok); sys.exit(0 if ok else 1)
PY
for f in "$C5" "$C1" "$TD" "$AH5" "$WM/config.json"; do [ -e "$f" ] || die "missing $f"; done
for i in $(seq 1 90); do pgrep -f "eval_w[m].py" >/dev/null || break; sleep 20; done

BASE="--arch v4 --horizon 5 --iters 8 --n-step 50 --gamma 0.98 --batch 256 \
--amax 1.6 --replay-prob 0.5 --expand-weight 3.0 --freeze-critic-frac 0.5 \
--actor-lr 3e-4 --actor-lr-final 3e-5 --critic-lr 1e-3 --critic-lr-final 1e-4 \
--expectile 0.1 --expectile-final 0.03 --mean-weight 0.1 --term-index min"

train_one(){ # tag extra seed
  local tag=$1 extra=$2 s=$3
  local out=/workspace/actors/lip4_sc_${tag}_s${s}.pt
  [ -f "$out" ] && return 0
  local slot; slot=$(acquire)
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
      --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
      $BASE --steps "$STEPS" $extra --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$L/sc_${tag}_s${s}.log" 2>&1
    release "$slot" ) &
  log "  -> $tag/s$s ($extra)"
}
eval_one(){ # tag seed mode
  local tag=$1 s=$2 mode=$3 d
  local A=/workspace/actors/lip4_sc_${tag}_s${s}.pt
  [ -f "$A" ] || { log "  WARN no actor $A"; return 0; }
  for d in $DRAWS; do
    local nm="f30_lip_sc${tag}_${mode}_pre_lewm_s${s}_e${d}"
    local cc; cc=$(sc "$nm"); [ -n "$cc" ] && [ "$cc" != FAIL ] && continue
    local EX
    if [ "$mode" = rh1 ]; then
      EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
          "+solver.align_deadline=true" "+solver.align_mode=argmin")
    else
      EX=(plan_config.receding_horizon=5)
    fi
    local slot; slot=$(acquire)
    (
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) MUJOCO_EGL_DEVICE_ID=$(( slot % NGPU )) \
      timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
        eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
        eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=8000:10000" \
        "${EX[@]}" policy="$WM" solver=lip "solver.actor_path=$A" \
        output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
      sr=""
      grep -q "ep_range 8000:10000" "$L/eval_${nm}.log" && \
        sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
      flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
      log "  $nm = ${sr:-FAIL}"; release "$slot" ) &
  done
}

# tag | extra   (min objective is already in BASE)
PHASE_A=(
  "min_base|"
  "min_amax12|--amax 1.2"
  "min_amax22|--amax 2.2"
  "min_exp10|--expand-weight 1.0"
  "min_exp60|--expand-weight 6.0"
  "min_rp075|--replay-prob 0.75"
  "min_it16|--iters 16"
)
# already at 3 seeds from the screen: min_rp025, min_stride1, min_mw03
PHASE_B=(
  "min_rp0|--replay-prob 0.0"
  "min_rp0125|--replay-prob 0.125"
  "min_rp0125_st1|--replay-prob 0.125 --replay-stride 1"
  "min_rp025_st1|--replay-prob 0.25 --replay-stride 1"
)

log "=== PHASE A: replicate 7 min arms to 3 seeds (14 trains)"
for e in "${PHASE_A[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in 1 2; do train_one "$tag" "$extra" "$s"; done
done
wait
log "PHASE A: trains done"

log "=== PHASE B: 4 new replay-grid configs x 3 seeds (12 trains)"
for e in "${PHASE_B[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in 0 1 2; do train_one "$tag" "$extra" "$s"; done
done
wait
na=$(ls /workspace/actors/lip4_sc_min_*_s[0-9].pt 2>/dev/null | wc -l)
log "PHASE B: trains done; $na min-objective actors on disk"

NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
log "=== EVALS"
for e in "${PHASE_A[@]}" "${PHASE_B[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in 0 1 2; do eval_one "$tag" "$s" rh1; eval_one "$tag" "$s" rh5; done
done
wait
log "EVALS: done"

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys, statistics as st
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"):
        try: rows[k] = float(v)
        except ValueError: pass
D = (42, 43, 44); S = (0, 1, 2)
FL = st.mean([rows["f30_nomove_h25_e%d" % d] for d in D])
def dr(tag, mode, s):
    v = [rows.get("f30_lip_sc%s_%s_pre_lewm_s%d_e%d" % (tag, mode, s, d)) for d in D]
    return None if any(x is None for x in v) else st.mean(v)
def agg(tag, mode):
    o = [dr(tag, mode, s) for s in S]; o = [x for x in o if x is not None]
    if not o: return None
    return st.mean(o), (st.stdev(o) if len(o) > 1 else 0.0), len(o), o
def hard(v): return None if v is None else 100.0*(v-FL)/(100.0-FL)
OFAT = [("min_base","recipe values"),("min_amax12","amax 1.2"),("min_amax22","amax 2.2"),
        ("min_exp10","expand 1.0"),("min_exp60","expand 6.0"),("min_rp025","replay-prob 0.25"),
        ("min_rp075","replay-prob 0.75"),("min_it16","iters 16"),
        ("min_stride1","replay-stride 1"),("min_mw03","mean-weight 0.3")]
print("\n=== REPLICATED OFAT under the min objective, 3 seeds, 6000 steps ===")
print("  %-20s%10s%7s%4s%10s%10s" % ("arm", "rh1", "sd", "n", "rh1 hard", "rh5"))
ctl = agg("min_base", "rh1")
for tag, lab in OFAT:
    a, b = agg(tag, "rh1"), agg(tag, "rh5")
    if not a: print("  %-20s%10s" % (lab, "pending")); continue
    print("  %-20s%10.2f%7.2f%4d%10.1f%10s   [%s]" % (
        lab, a[0], a[1], a[2], hard(a[0]), ("%.2f" % b[0]) if b else "-",
        " ".join("%.2f" % x for x in a[3])))
if ctl:
    print("\n  paired vs min_base (recipe values), rh1, same seeds:")
    for tag, lab in OFAT:
        if tag == "min_base": continue
        a = agg(tag, "rh1")
        if not a or a[2] != ctl[2]: continue
        ds = [a[3][i] - ctl[3][i] for i in range(len(ctl[3]))]
        if len(ds) < 2: continue
        m, sd = st.mean(ds), st.stdev(ds)
        t = m/(sd/len(ds)**0.5) if sd > 0 else float("inf")
        print("    %-20s %+6.2f  sd %5.2f  t %6.2f  %d/%d pos%s" % (
            lab, m, sd, t, sum(1 for x in ds if x > 0), len(ds),
            "  *" if abs(t) > 4.30 else ""))
    print("    (* = |t| > 4.30 = df=2 crit; 9 comparisons, so expect ~0.5 false positives)")
print("\n=== PHASE B: replay curriculum grid (min objective, 3 seeds), rh1 ===")
print("  %-14s%12s%12s" % ("replay-prob", "stride 5", "stride 1"))
GRID = [("0.0", "min_rp0", None), ("0.125", "min_rp0125", "min_rp0125_st1"),
        ("0.25", "min_rp025", "min_rp025_st1"), ("0.5", "min_base", "min_stride1")]
for lab, t5, t1 in GRID:
    a = agg(t5, "rh1") if t5 else None
    b = agg(t1, "rh1") if t1 else None
    f = lambda x: ("%12.2f" % x[0]) if x else "%12s" % ("n/a" if x is None else "-")
    print("  %-14s%s%s" % (lab, f(a), f(b) if t1 else "%12s" % "(same)"))
print("\n  ANCHORS: rh5 committed 90.00 | rh1 unaligned 73.33 |")
print("           rh1 argmin on production (last obj) 82.22 | rh1 deadline@40 84.89")
print("REPGRID_DONE")
PY
log "all done"
