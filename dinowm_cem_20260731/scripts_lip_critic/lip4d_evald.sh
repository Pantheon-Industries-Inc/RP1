#!/bin/bash
# Eval daemon for the LIPv4-dino sweep. GPU 0 only, strictly sequential (EGL).
# Scores every checkpoint the trainers drop (draw 44 as the screen; the final
# s3000 of each cell also gets draws 42/43). Idempotent via the CSV cache;
# exits when no trainer is alive and everything present has been scored.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_lip4d.csv
mkdir -p "$L" "$R" /workspace/videos_scratch; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_lip4d_eval.log"; }

ev(){ # actor_path draw
  local ap=$1 d=$2 nm
  nm="$(basename "$ap" .pt)_e${d}"
  grep -q "^${nm}," "$SUM" && return 0
  local t0 t1 sr
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 "$P/eval_wm_dino.py" --config-name cube \
    seed=$d eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" policy="$CKPT" solver=lip \
    "solver.actor_path=$ap" "+video_dir=/workspace/videos_scratch/$nm" \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s)
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL},$((t1-t0))" >> "$SUM"
  log "  $nm = ${sr:-FAIL}  ($((t1-t0))s)   [refs: plainCEM d44=78 d42=82 d43=80 | v1-LIP 70]"
}

log "=== lip4d eval daemon up ==="
while true; do
  did=0
  for ap in /workspace/actors/lip4d_*_s*.pt; do
    [ -f "$ap" ] || continue
    ev "$ap" 44; did=1
    case "$ap" in (*_s3000.pt|*_s6000.pt|*_s2000.pt) ev "$ap" 42; ev "$ap" 43;; esac
  done
  if [ -f /workspace/STOP_EVALD ]; then
    # persist across rounds; only an explicit sentinel stops the daemon
    for ap in /workspace/actors/lip4d_*_s*.pt; do
      [ -f "$ap" ] || continue
      ev "$ap" 44
      case "$ap" in (*_s3000.pt|*_s6000.pt|*_s2000.pt) ev "$ap" 42; ev "$ap" 43;; esac
    done
    log "=== trainers done, all checkpoints scored ==="
    sort -t, -k2 -rn "$SUM" | head -8 | tee -a "$L/driver_lip4d_eval.log"
    log "LIP4D_EVAL_DONE"
    exit 0
  fi
  [ "$did" = 0 ] && sleep 180 || sleep 30
done
