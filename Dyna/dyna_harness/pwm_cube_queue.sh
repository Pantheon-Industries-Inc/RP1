#!/bin/bash
# PWM arm on cube, both bases, 3 seeds -- adds the "policy actor-critic" row
# to the cross-base table. Uses the parallel session's committed PWM stack
# (train_pwm_ac.py + PWMSolver, contract-fixed in 11eace9) read-only, with
# the tworoom_pwm.sh invocation as precedent: trainer defaults except amax
# (per-base LIP-tuned operating point: LeWM 1.6, PLDM 4.5 -- PWM's own amax
# optimum is unexplored, first-pass caveat) and the campaign eval protocol
# (plan_config defaults h5/r5, solver.batch_size=10, held-out 8000:10000).
# Assets: the existing per-base tr8000 caches + TD teachers; PWM needs no
# action h5 (closed-loop latent policy).
# Waits for the PLDM Dyna loop. 2 waves (~70 min each, steps 8000) + 18
# cells => ~3h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_dynasplit.csv
DRV=$L/driver_pwm.log
V2WM=/workspace/models/v2WM; PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; SEEDS="0 1 2"; DRAWS="42 43 44"
LF1=/workspace/caches/v2_tr${EPHI}_fs1.pt;  LF5=/workspace/caches/v2_tr${EPHI}_fs5.pt
LTD=/workspace/metrics/v2_tr${EPHI}_TD.pt
QF1=/workspace/caches/pldm_tr${EPHI}_fs1.pt; QF5=/workspace/caches/pldm_tr${EPHI}_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PWM cube queue (pid $$): waiting for the PLDM Dyna loop ==="
T0=$(date +%s)
until grep -q "PLDM_DYNA_DONE" "$L/driver_pldmdyna.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 43200 ] && die "PLDM Dyna loop never finished in 12h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_pwm_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
for f in "$LF1" "$LF5" "$LTD" "$QF1" "$QF5" "$QTD"; do [ -e "$f" ] || die "missing $f"; done
[ -f "$P/train_pwm_ac.py" ] || die "train_pwm_ac.py not on pod (sync the repo code)"
grep -q "PWMSolver" "$CODE/stable_worldmodel/solver/pwm.py" || die "PWMSolver missing"
log "P0: preflight OK, box quiet"

pwm_wave(){ # base wm fs5 fs1 td amax
  local base=$1 wm=$2 c5=$3 c1=$4 td=$5 amax=$6
  log "P1: PWM wave $base (amax $amax, GPUs 0-2)"
  local g=0 s
  for s in $SEEDS; do
    out=/workspace/actors/pwm_${base}_s${s}.pt
    if [ ! -f "$out" ]; then
      CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_pwm_ac.py" \
        --cache "$c5" --cache-td "$c1" --wm "$wm" --init-value "$td" \
        --amax "$amax" --seed "$s" \
        --out "$out" --out-value "${out%.pt}_value.pt" \
        > "$L/pwm_train_${base}_s${s}.log" 2>&1 || log "  $base/s$s TRAIN FAILED" &
    fi
    g=$((g+1))
  done; wait
  log "P1: $base wave done"
}
pwm_wave lewm "$V2WM" "$LF5" "$LF1" "$LTD" 1.6
pwm_wave pldm "$PLDM" "$QF5" "$QF1" "$QTD" 4.5

log "P2: eval queue (18 cells, sequential, egl, held-out $EVAL_RANGE)"
while pgrep -f "train_pwm_a[c]|train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 60; done
run_eval(){ # name policy actor draw
  local nm=$1 pol=$2 actor=$3 d=$4
  local c; c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; return 0; }
  [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$pol" solver=pwm "solver.actor_path=$actor" solver.batch_size=10 \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
}
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "pwm_lewm_s${s}_e${d}" "$V2WM" "/workspace/actors/pwm_lewm_s${s}.pt" "$d"
done; done
for s in $SEEDS; do for d in $DRAWS; do
  run_eval "pwm_pldm_s${s}_e${d}" "$PLDM" "/workspace/actors/pwm_pldm_s${s}.pt" "$d"
done; done

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
def sm(arm, s):
    vs = [rows.get(f"{arm}_s{s}_e{d}") for d in (42, 43, 44)]
    return None if any(v is None for v in vs) else sum(vs) / 3
print("=== PWM CARD (frozen WMs, held-out 8000:10000, egl, 3 seeds) ===")
for base, ref in (("lewm", "refs: latent+CEM 76.0 TD+CEM 78.0 LIP 87.8"),
                  ("pldm", "refs: latent+CEM 66.0 TD+CEM 73.3 LIP 75.3")):
    ms = [sm(f"pwm_{base}", s) for s in (0, 1, 2)]
    if all(m is not None for m in ms):
        print(f"  PWM {base}: {' '.join(f'{m:5.1f}' for m in ms)}   mean {sum(ms)/3:.1f}   ({ref})")
print("  caveats: PWM at each base's LIP-tuned amax (own optimum unexplored);")
print("  trainer defaults otherwise (steps 8000, max-delta 12, gamma 0.99).")
print("PWM_CUBE_DONE")
PY
log "done"
