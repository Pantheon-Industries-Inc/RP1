#!/bin/bash
# TD+CEM on the FULL-token DINO-WM, then the all-10k CEM replication attempt.
# Strictly sequential: concurrent EGL evals corrupt each other.
#
# Teacher: dinofull_e0.1_lr0.0003, the sweep winner (spearman 0.483,
# pair_acc 0.693, monotone 0.678; the LeWM reference is 0.484/0.691/0.923).
# No +metric_pool_dim is passed: the hook auto-detects flatten-vs-pool from the
# head width (metric_in 75264 == 196*384 -> flatten).
#
# Arm 1 (tdcem_*)   held-out eps 8000-9999 -> compare to plain CEM 80.0
# Arm 2 (dinoall_*) ALL 10k episodes       -> compare to repo baseline 86
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
TD=/workspace/metrics/dinofull_e0.1_lr0.0003.pt
SUM=$R/summary_dino_tdcem.csv
mkdir -p "$L" "$R" /workspace/videos_scratch; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_tdcem.log"; }

ev(){ # name extra_args...
  local nm=$1; shift
  grep -q "^${nm}," "$SUM" && { log "$nm cached"; return 0; }
  local t0 t1 sr
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$P/eval_wm_dino.py" --config-name cube \
    eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 policy="$CKPT" solver=cem \
    "+video_dir=/workspace/videos_scratch/$nm" output.filename="${nm}.txt" "$@" \
    > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s)
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"
  log "  $nm = ${sr:-FAIL}  ($((t1-t0))s)  $(grep -oE 'metric-hook.*' "$L/eval_${nm}.log" | head -1)"
  if [ -z "$sr" ]; then
    log "    ERR: $(grep -E 'Error|Traceback' "$L/eval_${nm}.log" | tail -2 | tr '\n' ' ')"
  fi
}

log "=== ARM 1: TD+CEM, held-out eps 8000-9999 (vs plain CEM 80.0) ==="
for d in 42 43 44; do
  ev "tdcem_e${d}" seed=$d "+eval.ep_range=8000:10000" "+metric=$TD"
done

log "=== ARM 2: plain CEM, ALL 10k episodes (vs repo baseline 86) ==="
for d in 42 43 44; do
  ev "dinoall_e${d}" seed=$d
done

log "TDCEM_DONE"
cat "$SUM" | tee -a "$L/driver_tdcem.log"
