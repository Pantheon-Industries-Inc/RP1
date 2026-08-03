#!/bin/bash
# GPU 0, strictly sequential EGL work. Priority order:
#  1. fx96_* LIP actors  -- LIP with EVERY upstream fault fixed: correct action
#     convention, correct returns, and the std96k critic that beats the LeWM
#     teacher on both unconfounded probes. This is the campaign question.
#  2. TD+CEM with std96k -- critic quality -> planning, with no actor confound.
#  3. fx_*_s3000         -- unscored; their earlier FAILs were GPU-0 contention.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_final.csv
mkdir -p "$R" /workspace/videos_scratch; touch "$SUM"
cd /workspace/code/stable-worldmodel
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_final.log"; }

ev(){ # name  extra_hydra_args...
  local nm=$1; shift
  grep -q "^${nm}," "$SUM" && { log "  $nm cached"; return 0; }
  local t0 t1 sr
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm_dino.py" --config-name cube \
    eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" policy="$CKPT" \
    "+video_dir=/workspace/videos_scratch/$nm" output.filename="${nm}.txt" "$@" \
    > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s)
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"
  log "  $nm = ${sr:-FAIL} ($(( (t1-t0)/60 ))m)"
  if [ -z "$sr" ]; then
    log "    ERR $(grep -E 'Error|Traceback' "$L/eval_${nm}.log" | tail -1)"
  fi
}

log "=== 1. LIP on the std96k critic (refs: plain CEM 80.0, prior LIP 69.3) ==="
for c in ctrl exp0.5 rep0.25; do
  A=/workspace/actors/fx96_${c}_s3000.pt
  if [ ! -f "$A" ]; then log "  fx96_${c}_s3000 missing, skip"; continue; fi
  for d in 44 42 43; do
    ev "lip96_${c}_e${d}" seed=$d solver=lip "solver.actor_path=$A"
  done
done

log "=== 2. TD+CEM with std96k (ref: prior TD+CEM 76.0) ==="
for d in 44 42 43; do
  ev "tdcem96_e${d}" seed=$d solver=cem \
     "+metric=/workspace/metrics/dinopool_td_std96k.pt"
done

log "=== 3. leftover fx_*_s3000 (pre-critic-upgrade LIP) ==="
for c in ctrl a12 a20; do
  A=/workspace/actors/fx_${c}_s3000.pt
  if [ -f "$A" ]; then
    for d in 44 42 43; do
      ev "fxs3_${c}_e${d}" seed=$d solver=lip "solver.actor_path=$A"
    done
  fi
done

log "FINAL_EVALS_DONE"
sort "$SUM" | tee -a "$L/driver_final.log"
