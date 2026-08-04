#!/bin/bash
# THE ADAPTED BASELINE: the 90.00 rh=5 recipe, changed only where rh=1 demands it.
# Waits for the hyper OFAT to finish so it gets whole GPUs.
#
# WHAT "ADAPTED" MEANS -- three changes from the shipped recipe, nothing else:
#   1. --term-index min     grade min_j V(traj_j, zg) instead of only chunk H-1.
#                           Matches the argmin readout AND the task's own scoring
#                           (terminate_at_goal fires the moment the cube is inside
#                           0.04 m at ANY step).
#   2. --replay-prob 0.0    the curriculum banked post-FULL-plan states, the rh=5
#                           query. Measured a null at 3 seeds (87.11 vs 87.56,
#                           t = -0.76), so it is removed rather than retuned.
#   3. --warm-train         two-pass training: refine from A_0 = 0, then advance the
#                           state ONE chunk and re-refine from the shifted plan.
#                           Fixes both halves of the rh=1 mismatch -- nonzero warm
#                           start A_0, and a mid-task state one chunk in.
# Everything else is the 90.00 recipe verbatim: arch v4, horizon 5, iters 8,
# amax 1.6, expand-weight 3.0, mean-weight 0.1, freeze-critic-frac 0.5, gamma 0.98,
# n-step 50, batch 256, expectile 0.1->0.03, critic-lr 1e-3->1e-4,
# actor-lr 3e-4->3e-5, teacher up_lewmpre_g98s12k_s0.
#
# EVAL, three ways per actor, so warm start is isolated from the objective:
#   rh1w  rh=1, argmin readout, warm_init=TRUE   <- matches how it was trained
#   rh1   rh=1, argmin readout, warm_init=false  <- warm training, cold deploy
#   rh5   the committed protocol, regression check
# Plus warm_init=TRUE on the EXISTING replay-0 actors (trained cold): the
# eval-only contrast that shows whether warm start needs training at all. Until
# today LIPSolver accepted init_action and dropped it, so no warm-start number
# anyone has quoted was real.
#
# 3 actors at ~1.8x cost (~90 min, all parallel) + 36 cells (~12 min).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_warm.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM; TD=/workspace/metrics/up_lewmpre_g98s12k_s0.pt
C5=/workspace/caches/v2_tr8000_fs5.pt; C1=/workspace/caches/v2_tr8000_fs1.pt
AH5=/workspace/datasets/expert_actions.h5
NGPU=4; NSLOT=4; SLOTDIR=/tmp/warmslots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
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
p = inspect.signature(LIPSolver.__init__).parameters
ok = ("warm_init" in p and "align_mode" in p and "--warm-train" in h and "--term-index" in h)
print("[preflight] warm_init + warm-train + argmin:", ok); sys.exit(0 if ok else 1)
PY

log "waiting for the hyper OFAT to finish (up to 8 h)"
for i in $(seq 1 480); do
  grep -q RP0OFAT_DONE "$L/driver_rp0ofat.log" 2>/dev/null && break
  grep -q FATAL "$L/driver_rp0ofat.log" 2>/dev/null && die "the OFAT reported FATAL; not proceeding"
  sleep 60
done
grep -q RP0OFAT_DONE "$L/driver_rp0ofat.log" 2>/dev/null || die "timed out waiting for the OFAT"
log "OFAT finished; starting the adapted baseline"

BASE="--arch v4 --horizon 5 --iters 8 --n-step 50 --gamma 0.98 --batch 256 \
--amax 1.6 --replay-prob 0.0 --expand-weight 3.0 --freeze-critic-frac 0.5 \
--actor-lr 3e-4 --actor-lr-final 3e-5 --critic-lr 1e-3 --critic-lr-final 1e-4 \
--expectile 0.1 --expectile-final 0.03 --mean-weight 0.1 --term-index min --warm-train"

log "=== TRAIN: adapted baseline, 3 seeds, 6000 steps (~1.8x cost)"
for s in 0 1 2; do
  out=/workspace/actors/lip4_sc_wt_base_s${s}.pt
  [ -f "$out" ] && { log "  s$s present"; continue; }
  slot=$(acquire)
  ( # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$(( slot % NGPU )) timeout 86400 python3 "$P/train_lip_ac.py" \
      --cache "$C5" --cache-td "$C1" --h5 "$AH5" --wm "$WM" --init-value "$TD" \
      $BASE --steps 6000 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" > "$L/sc_wt_base_s${s}.log" 2>&1
    release "$slot" ) &
  log "  -> s$s"
done
wait
log "trains done: $(ls /workspace/actors/lip4_sc_wt_base_s[0-9].pt 2>/dev/null | wc -l)/3"

NSLOT=8; rmdir "$SLOTDIR"/* 2>/dev/null || true
ev(){ # actorpath cellprefix seed mode
  local A=$1 pre=$2 s=$3 mode=$4 d
  [ -f "$A" ] || { log "  WARN no actor $A"; return 0; }
  for d in 42 43 44; do
    local nm="${pre}_${mode}_pre_lewm_s${s}_e${d}"
    local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && continue
    local EX
    case "$mode" in
      rh1w) EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
                "+solver.align_deadline=true" "+solver.align_mode=argmin"
                "+solver.warm_init=true") ;;
      rh1)  EX=(plan_config.receding_horizon=1 "+plan_config.eval_budget=50"
                "+solver.align_deadline=true" "+solver.align_mode=argmin") ;;
      rh5)  EX=(plan_config.receding_horizon=5) ;;
    esac
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
      log "  $nm = ${sr:-FAIL}"; release "$slot"
    ) &
  done
}
log "=== EVAL: adapted baseline 3 ways + warm_init on the cold-trained replay-0 actors"
for s in 0 1 2; do
  ev "/workspace/actors/lip4_sc_wt_base_s${s}.pt" "f30_lip_scwt_base" "$s" rh1w
  ev "/workspace/actors/lip4_sc_wt_base_s${s}.pt" "f30_lip_scwt_base" "$s" rh1
  ev "/workspace/actors/lip4_sc_wt_base_s${s}.pt" "f30_lip_scwt_base" "$s" rh5
  ev "/workspace/actors/lip4_sc_min_rp0_s${s}.pt" "f30_lip_scmin_rp0"  "$s" rh1w
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
def dr(pre, mode, s):
    v = [rows.get("%s_%s_pre_lewm_s%d_e%d" % (pre, mode, s, d)) for d in D]
    return None if any(x is None for x in v) else st.mean(v)
def agg(pre, mode):
    o = [dr(pre, mode, s) for s in S]; o = [x for x in o if x is not None]
    if not o: return None
    return st.mean(o), (st.stdev(o) if len(o) > 1 else 0.0), len(o), o
def hard(v): return None if v is None else 100.0*(v-FL)/(100.0-FL)
f = lambda x: "%9.2f" % x[0] if x else "%9s" % "-"
g = lambda x: "%6.2f" % x[1] if x else "%6s" % "-"
W, C = "f30_lip_scwt_base", "f30_lip_scmin_rp0"
print("\n=== ADAPTED BASELINE: min objective + replay 0 + warm-train ===")
print("  %-34s%9s%6s%10s" % ("cell", "score", "sd", "hard"))
for pre, mode, lab in ((W, "rh1w", "warm-trained, warm deploy  (rh1)"),
                       (W, "rh1",  "warm-trained, cold deploy  (rh1)"),
                       (W, "rh5",  "warm-trained, committed    (rh5)"),
                       (C, "rh1w", "cold-trained,  warm deploy (rh1)"),
                       (C, "rh1",  "cold-trained,  cold deploy (rh1)")):
    a = agg(pre, mode)
    print("  %-34s%s%s%s   [%s]" % (lab, f(a), g(a),
          ("%10.1f" % hard(a[0])) if a else "%10s" % "-",
          " ".join("%.2f" % x for x in a[3]) if a else ""))
def paired(x, y, lab):
    a, b = x, y
    if not a or not b or a[2] != b[2] or a[2] < 2: return
    ds = [a[3][i] - b[3][i] for i in range(a[2])]
    m, sd = st.mean(ds), st.stdev(ds)
    t = m/(sd/len(ds)**0.5) if sd > 0 else float("inf")
    print("    %-40s %+6.2f  sd %5.2f  t %6.2f  %d/%d pos%s" % (
        lab, m, sd, t, sum(1 for v in ds if v > 0), len(ds), "  *" if abs(t) > 4.30 else ""))
print("\n  paired contrasts (same seeds):")
paired(agg(W, "rh1w"), agg(C, "rh1"), "adapted(warm) - cold baseline        ")
paired(agg(W, "rh1w"), agg(W, "rh1"),  "warm deploy - cold deploy (warm-trained)")
paired(agg(C, "rh1w"), agg(C, "rh1"),  "warm deploy - cold deploy (cold-trained)")
print("    (* = |t| > 4.30 = df=2 crit)")
print("\n  ANCHORS rh1: unaligned 73.33 | argmin on the 90.00 recipe 82.22 |")
print("               deadline@40 84.89 | best so far 87.56")
print("  ANCHOR  rh5: committed 90.00")
print("  READING: the third contrast is the one that says whether warm start needed")
print("  training. If cold-trained warm deploy is flat or worse while warm-trained")
print("  warm deploy gains, the two-pass training is what unlocked it.")
print("WARM_DONE")
PY
log "all done"
