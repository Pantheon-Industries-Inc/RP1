#!/bin/bash
# TD+CEM sweep, REPLACEMENT MODE ONLY (value metric alone, no latent cost).
#
# The first teacher (e0.1_lr3e-4, best pair_acc) gave 78/80/70 = 76.0 vs plain
# CEM 80.0. This sweeps teachers spanning the WHOLE probe range on the single
# most discriminating draw (44: plain 78.0, that teacher 70.0), to answer two
# things at once:
#
#   1. does any teacher actually plan better than 76.0 / beat plain CEM 80.0?
#   2. does ANY offline probe metric predict planning performance?
#
# Question 2 is why the list deliberately includes bad teachers, and one
# DEGENERATE control (e0.03_lr0.003: spearman 0.010, monotone 0.000 -- the
# metric collapsed during training). If a collapsed metric plans about as well
# as the best one, then CEM is barely using the metric signal, and the -4.0 is
# simply the cost of discarding latent MSE rather than anything about the TD
# teacher. That would redirect the whole stage.
#
# Gated on GPU idle: concurrent EGL evals corrupt each other.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results; M=/workspace/metrics
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_tdcem_sweep.csv
DRAW=44
mkdir -p "$L" "$R" /workspace/videos_scratch; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_tdcem_sweep.log"; }

# name:metric  -- ordered best-probe-first, with the controls last
CELLS="
e0.03_lr0.0003:$M/dinofull_e0.03_lr0.0003.pt
hidden1024:$M/dinoarch_hidden1024.pt
e0.1_lr0.001:$M/dinofull_e0.1_lr0.001.pt
e0.03_lr0.001:$M/dinofull_e0.03_lr0.001.pt
e0.01_lr0.0003:$M/dinofull_e0.01_lr0.0003.pt
e0.5_lr0.0003:$M/dinofull_e0.5_lr0.0003.pt
e0.01_lr0.001:$M/dinofull_e0.01_lr0.001.pt
COLLAPSED_e0.03_lr0.003:$M/dinofull_e0.03_lr0.003.pt
"

log "waiting for the running eval chain to finish (EGL must be sequential)"
while pgrep -f "eval_wm_din[o]" >/dev/null; do sleep 60; done
log "GPU free; TD+CEM sweep on draw $DRAW, replacement mode"
log "  references on draw 44: plain CEM 78.0 | e0.1_lr3e-4 replacement 70.0"

for cell in $CELLS; do
  nm="${cell%%:*}"; mp="${cell#*:}"
  [ -f "$mp" ] || { log "  $nm: metric missing, skip"; continue; }
  grep -q "^${nm}," "$SUM" && { log "  $nm cached"; continue; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$P/eval_wm_dino.py" --config-name cube \
    seed=$DRAW eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" policy="$CKPT" solver=cem \
    "+metric=$mp" "+video_dir=/workspace/videos_scratch/tdsw_$nm" \
    output.filename="tdsw_${nm}.txt" > "$L/eval_tdsw_${nm}.log" 2>&1
  t1=$(date +%s)
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_tdsw_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"
  log "  $nm = ${sr:-FAIL}  ($(( (t1-t0)/60 )) min)"
  if [ -z "$sr" ]; then
    log "    ERR: $(grep -E 'Error|Traceback' "$L/eval_tdsw_${nm}.log" | tail -2 | tr '\n' ' ')"
  fi
done

log "TDCEM_SWEEP_DONE"
cat "$SUM" | tee -a "$L/driver_tdcem_sweep.log"
