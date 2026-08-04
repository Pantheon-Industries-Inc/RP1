#!/bin/bash
# Deadline-aligned RLP on LeWM. rh5align is the INERTNESS CONTROL: at
# receding_horizon == horizon the plan already ends on the budget cap, so
# term_idx is H-1 at every solve and this must reproduce the banked 90.00
# exactly. rh1align is the test: does alignment recover the -16.67 that the
# cadence change cost?
#
# Written as a file rather than an inline heredoc: the first attempt escaped the
# dollar in "${sr:-FAIL}" inside a quoted heredoc, so the value never expanded
# and all 18 cells banked the literal string FAIL while the evals themselves ran
# fine. Keep the substitution unescaped and let it expand in the local shell.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
SUM=$R/summary_pldmgrid_egl.csv; DRV=$L/driver_align.log
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
WM=/workspace/models/v2WM
EPHI=8000
NSLOT=8; NGPU=4; SLOTDIR=/tmp/alignslots
mkdir -p "$SLOTDIR" "$L" "$R"; touch "$SUM" "$SUM.lock"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }
acquire(){ local s; while :; do for s in $(seq 0 $((NSLOT-1))); do
  mkdir "$SLOTDIR/$s" 2>/dev/null && { echo "$s"; return; }; done; sleep 5; done; }
release(){ rmdir "$SLOTDIR/$1" 2>/dev/null || true; }
cd /workspace/code/stable-worldmodel

# fail fast if the patch is not actually live, rather than silently measuring
# the unaligned solver and reporting it as aligned
python3 - <<'PY' || exit 1
import sys, inspect
sys.path.insert(0, "/workspace/code/stable-worldmodel")
from stable_worldmodel.solver import LIPSolver
import stable_worldmodel as swm
ok = ("align_deadline" in inspect.signature(LIPSolver.__init__).parameters
      and hasattr(LIPSolver, "set_align_remaining")
      and "eval_budget" in swm.PlanConfig.__dataclass_fields__)
print("[preflight] alignment patch live:", ok)
sys.exit(0 if ok else 1)
PY
log "=== RLP deadline-aligned (pid $$)"

for rh in 1 5; do
  for s in 0 1 2; do
    for d in 42 43 44; do
      nm="f30_lip_rh${rh}align_pre_lewm_s${s}_e${d}"
      c=$(sc "$nm")
      [ -n "$c" ] && [ "$c" != FAIL ] && { log "  $nm banked ($c)"; continue; }
      slot=$(acquire); g=$(( slot % NGPU ))
      (
        CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g timeout 14400 \
          python3 "$P/eval_wm.py" --config-name cube seed=$d \
          eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
          eval.goal_offset_steps=25 eval.eval_budget=50 "+eval.ep_range=${EPHI}:10000" \
          plan_config.receding_horizon=$rh "+plan_config.eval_budget=50" \
          policy="$WM" solver=lip "+solver.align_deadline=true" \
          "solver.actor_path=/workspace/actors/lip4_re_lewm_exp30_s${s}.pt" \
          output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
        sr=""
        grep -q "ep_range ${EPHI}:10000" "$L/eval_${nm}.log" && \
          sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
        flock "$SUM.lock" -c "echo '${nm},${sr:-FAIL}' >> '$SUM'"
        log "  $nm = ${sr:-FAIL}"
        release "$slot"
      ) &
    done
  done
done
wait
log "cells done"
