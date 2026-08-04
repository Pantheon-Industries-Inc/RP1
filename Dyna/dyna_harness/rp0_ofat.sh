#!/bin/bash
# OFAT around the NEW baseline: min objective + replay-prob 0. 3 seeds, 6000 steps.
# LeWM, I3 argmin readout at rh=1, rh=5 regression column.
#
# WHY replay-prob 0 IS THE BASELINE NOW. At 3 seeds, replay-prob 0 = 87.11 +/- 1.02
# vs 0.25 = 87.56 +/- 2.04; paired -0.44, sd 1.02, t = -0.76, 1/3 positive -- a
# null. The screen's apparent monotone trend (0.25 -> 88.00, 0.5 -> 86.00,
# 0.75 -> 83.33) was 1-seed noise. Since the curriculum's PRESENCE does not
# matter, its stride cannot be what was helping either, so --replay-stride is
# retired along with it. Dropping replay also makes the recipe simpler and
# removes a term whose banked window (post-full-plan) was mismatched to rh=1
# anyway.
#
# 3 SEEDS DIRECTLY, NO SCREEN. The 20-arm 1-seed screen resolved nothing about any
# hyper (se ~3.5 vs deltas of 1-3), and min_mw03 fell from 86.00 at 1 seed to
# 83.33 +/- 2.67 at 3. Screening again would just re-pay for noise.
#
# --expand-weight 0.0 is included deliberately: with the replay curriculum off,
# expand is the only remaining term that trains on the planner's OWN imagined
# rollouts, so "does either self-generated signal matter at rh=1" is a coherent
# pair of questions rather than one.
#
# 21 actors (~50 min each, 4 at a time => ~5 h) + 126 cells (~30 min).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_rp0ofat.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM; TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
AH5=/workspace/datasets/expert_actions.h5
NGPU=4; SLOTDIR=/tmp/rp0oslots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
NSLOT=4
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 10; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd /workspace/code/stable-worldmodel

python3 - <<'PY' || exit 1
import subprocess, sys, inspect
sys.path.insert(0, "/workspace/code/stable-worldmodel")
from stable_worldmodel.solver import LIPSolver
h = subprocess.run([sys.executable, "/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac.py",
                    "--help"], capture_output=True, text=True).stdout
ok = ("align_mode" in inspect.signature(LIPSolver.__init__).parameters and "--term-index" in h)
print("[preflight] argmin readout + min objective:", ok); sys.exit(0 if ok else 1)
PY
for f in "$C5" "$C1" "$TD" "$AH5" "$WM/config.json"; do [ -e "$f" ] || die "missing $f"; done
for i in $(seq 1 90); do pgrep -f "eval_w[m].py" >/dev/null || break; sleep 20; done

# replay-prob 0 is in BASE; --replay-stride is retired (moot with no replay)
BASE="--arch v4 --horizon 5 --iters 8 --n-step 50 --gamma 0.98 --batch 256 \
--amax 1.6 --replay-prob 0.0 --expand-weight 3.0 --freeze-critic-frac 0.5 \
--actor-lr 3e-4 --actor-lr-final 3e-5 --critic-lr 1e-3 --critic-lr-final 1e-4 \
--expectile 0.1 --expectile-final 0.03 --mean-weight 0.1 --term-index min"

# tag | one change from BASE
ARMS=(
  "amax12|--amax 1.2"
  "amax22|--amax 2.2"
  "exp00|--expand-weight 0.0"
  "exp10|--expand-weight 1.0"
  "exp60|--expand-weight 6.0"
  "it16|--iters 16"
  "mw03|--mean-weight 0.3"
)

log "=== OFAT around min + replay-prob 0: 7 arms x 3 seeds @ 6000 steps"
for e in "${ARMS[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in 0 1 2; do
    out=/workspace/actors/lip4_sc_rp0_${tag}_s${s}.pt
    [ -f "$out" ] && { log "  $tag/s$s present"; continue; }
    slot=$(acquire)
    ( # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 43200 python3 "$P/train_lip_ac.py" \
        --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
        $BASE --steps 6000 $extra --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/sc_rp0_${tag}_s${s}.log" 2>&1
      release "$slot" ) &
    log "  -> $tag/s$s ($extra)"
  done
done
wait
na=$(ls /workspace/actors/lip4_sc_rp0_*_s[0-9].pt 2>/dev/null | wc -l)
log "trains done: $na/21"
[ "$na" -ge 3 ] || die "too few actors ($na)"

NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
log "=== evals: rh1 argmin + rh5 regression"
for e in "${ARMS[@]}"; do
  IFS='|' read -r tag extra <<< "$e"
  for s in 0 1 2; do
    A=/workspace/actors/lip4_sc_rp0_${tag}_s${s}.pt
    [ -f "$A" ] || { log "  WARN no actor $tag/s$s"; continue; }
    for mode in rh1 rh5; do
      for d in 42 43 44; do
        nm="f30_lip_scrp0_${tag}_${mode}_pre_lewm_s${s}_e${d}"
        c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
        if [ "$mode" = rh1 ]; then
          EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
              "+solver.align_deadline=true" "+solver.align_mode=argmin")
        else
          EX=(plan_config.receding_horizon=5)
        fi
        slot=$(acquire)
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
          log "  $nm = ${sr:-FAIL}"; release "$slot"
        ) &
      done
    done
  done
done
wait
log "evals done"

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
def dr(key, mode, s):
    v = [rows.get("f30_lip_%s_%s_pre_lewm_s%d_e%d" % (key, mode, s, d)) for d in D]
    return None if any(x is None for x in v) else st.mean(v)
def agg(key, mode):
    o = [dr(key, mode, s) for s in S]; o = [x for x in o if x is not None]
    if not o: return None
    return st.mean(o), (st.stdev(o) if len(o) > 1 else 0.0), len(o), o
def hard(v): return None if v is None else 100.0*(v-FL)/(100.0-FL)
ctl = agg("scmin_rp0", "rh1"); ctl5 = agg("scmin_rp0", "rh5")
ARMS = [("amax12","amax 1.2"),("amax22","amax 2.2"),("exp00","expand 0.0"),
        ("exp10","expand 1.0"),("exp60","expand 6.0"),("it16","iters 16"),
        ("mw03","mean-weight 0.3")]
print("\n=== OFAT around min + replay-prob 0, 3 seeds, 6000 steps ===")
print("  %-20s%10s%7s%4s%10s%10s" % ("arm", "rh1", "sd", "n", "rh1 hard", "rh5"))
if ctl:
    print("  %-20s%10.2f%7.2f%4d%10.1f%10s   [%s]  <- BASELINE" % (
        "replay 0 (base)", ctl[0], ctl[1], ctl[2], hard(ctl[0]),
        ("%.2f" % ctl5[0]) if ctl5 else "-", " ".join("%.2f" % x for x in ctl[3])))
for tag, lab in ARMS:
    a, b = agg("scrp0_" + tag, "rh1"), agg("scrp0_" + tag, "rh5")
    if not a: print("  %-20s%10s" % (lab, "pending")); continue
    print("  %-20s%10.2f%7.2f%4d%10.1f%10s   [%s]" % (
        lab, a[0], a[1], a[2], hard(a[0]), ("%.2f" % b[0]) if b else "-",
        " ".join("%.2f" % x for x in a[3])))
if ctl:
    print("\n  paired vs replay-0 baseline, rh1, same seeds:")
    for tag, lab in ARMS:
        a = agg("scrp0_" + tag, "rh1")
        if not a or a[2] != ctl[2]: continue
        ds = [a[3][i] - ctl[3][i] for i in range(ctl[2])]
        if len(ds) < 2: continue
        m, sd = st.mean(ds), st.stdev(ds)
        t = m/(sd/len(ds)**0.5) if sd > 0 else float("inf")
        print("    %-20s %+6.2f  sd %5.2f  t %6.2f  %d/%d pos%s" % (
            lab, m, sd, t, sum(1 for x in ds if x > 0), len(ds),
            "  *" if abs(t) > 4.30 else ""))
    print("    (* = |t| > 4.30 = df=2 crit; 7 comparisons => expect ~0.35 false positives)")
print("\n  ANCHORS rh1: unaligned 73.33 | argmin on production(last obj) 82.22 |")
print("               deadline@40 84.89 | best so far 87.56 (replay 0.25)")
print("  ANCHOR  rh5: committed 90.00")
print("RP0OFAT_DONE")
PY
log "all done"
