#!/bin/bash
# The one grid cell with a mechanism: h3 x amax 4.5 on PLDM. Shorter
# imagination shrinks the exploitation surface, so it should tolerate a
# larger clip -- the interaction corner of the two best arms (h5@4.5 75.3,
# h3@3.5 74.4). PRE-REGISTERED decision rule: < 77 at 3 seeds => stop tuning
# pure LIP on PLDM; the tie with TD+CEM is the result and the next lever is
# the equal-compute hybrid. Waits for the PWM queue. ~1.2h.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; SUM=$R/summary_dynasplit.csv
DRV=$L/driver_pldm_h3a45.log
PLDM=/workspace/models/PLDM_OgBench_lewm
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
AH5=/workspace/datasets/expert_actions.h5
QF1=/workspace/caches/pldm_tr8000_fs1.pt
QF5=/workspace/caches/pldm_tr8000_fs5.pt
QTD=/workspace/metrics/pldm_TD.pt
EPHI=8000; EVAL_RANGE="${EPHI}:10000"; DRAWS="42 43 44"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
cd "$CODE"

log "=== PLDM h3@4.5 cell (pid $$): waiting for the PWM queue ==="
T0=$(date +%s)
until grep -q "PWM_CUBE_DONE" "$L/driver_pwm.log" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 64800 ] && die "PWM queue never finished in 18h"
  sleep 300
done
while pgrep -f "train_lip_a[c]|train_pwm_a[c]|train_metri[c]|lewm_exper[t]|eval_w[m]" >/dev/null; do sleep 120; done
for f in "$QF1" "$QF5" "$QTD" "$AH5"; do [ -e "$f" ] || die "missing $f"; done
log "P0: box quiet"

log "P1: h3@4.5 wave (GPUs 0-2)"
g=0; for s in 0 1 2; do
  out=/workspace/actors/lip4_pldm_h3a45_s${s}.pt
  if [ ! -f "$out" ]; then
    CUDA_VISIBLE_DEVICES=$g timeout 28800 python3 "$P/train_lip_ac.py" \
      --cache "$QF5" --cache-td "$QF1" --h5 "$AH5" --wm "$PLDM" --init-value "$QTD" \
      --horizon 3 --iters 8 --steps 6000 --n-step 50 --amax 4.5 \
      --expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 \
      --actor-lr 3e-4 --actor-lr-final 3e-5 --arch v4 --seed "$s" \
      --out "$out" --out-value "${out%.pt}_value.pt" \
      > "$L/pldm_lip_h3a45_s${s}.log" 2>&1 || log "  h3a45/s$s TRAIN FAILED" &
  fi
  g=$((g+1))
done; wait
log "P1: wave done"

log "P2: 9 cells (matched solver horizon 3)"
while pgrep -f "train_lip_a[c]|train_metri[c]" >/dev/null; do sleep 60; done
for s in 0 1 2; do for d in $DRAWS; do
  nm="pldm_h3a45_s${s}_e${d}"
  c=$(sc "$nm"); [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm cached ($c)"; continue; }
  actor=/workspace/actors/lip4_pldm_h3a45_s${s}.pt
  [ -f "$actor" ] || { log "  $nm: actor missing, FAIL"; echo "${nm},FAIL" >> "$SUM"; continue; }
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" \
    policy="$PLDM" solver=lip "solver.actor_path=$actor" \
    plan_config.horizon=3 plan_config.receding_horizon=3 \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" \
    || { log "  $nm: ep_range NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; continue; }
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"; log "  $nm = ${sr:-FAIL}"
done; done

python3 - "$SUM" 2>&1 <<'PY' | tee -a "$DRV"
import sys
rows = {}
for ln in open(sys.argv[1]):
    if "," not in ln: continue
    k, v = ln.strip().split(",")[:2]
    if v not in ("", "FAIL"): rows[k] = float(v)
ms = []
for s in (0, 1, 2):
    vs = [rows.get(f"pldm_h3a45_s{s}_e{d}") for d in (42, 43, 44)]
    ms.append(sum(vs)/3 if all(v is not None for v in vs) else None)
print("=== PLDM h3@4.5 CELL (held-out, matched horizon, 3 seeds) ===")
if all(m is not None for m in ms):
    mean = sum(ms)/3
    print(f"  h3@4.5: {' '.join(f'{m:5.1f}' for m in ms)}   mean {mean:.1f}  spread {max(ms)-min(ms):.1f}")
    print(f"  neighbors: h5@4.5 75.3 | h3@3.5 74.4 | TD+CEM 73.3")
    print(f"  decision rule: {'INTERACTION REAL - extend seeds' if mean >= 77 else 'flat mound confirmed - STOP tuning pure LIP; hybrid or writeup next'}")
print("PLDM_H3A45_DONE")
PY
log "done"
