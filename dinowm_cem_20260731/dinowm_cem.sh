#!/bin/bash
# Upstream DINO-WM (noprop, cube, epoch 10) + CEM on the canonical cube protocol.
# fp32 deliberately: ++bf16=true crashes dino models here (CEM feeds fp32 actions
# into a bf16-cast action Embedder Conv1d).
# img_size 196: the checkpoint's predictor has 588 = 3 x 196 tokens, i.e. 14x14
# patches of 14 px. 224 px yields 768 tokens and raises.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; R=/workspace/results
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_dinowm_cem.csv
mkdir -p "$L" "$R"; touch "$SUM"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_dinowm_cem.log"; }

run(){ # drawseed
  local d=$1 nm="dinocem_e${d}" t0 t1
  grep -q "^${nm}," "$SUM" && { log "$nm cached"; return 0; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 28800 python3 "$P/eval_wm_dino.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" policy="$CKPT" solver=cem \
    output.filename="${nm}.txt" > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s)
  grep -q "ep_range 8000:10000" "$L/eval_${nm}.log" || { log "$nm: EP_RANGE NOT APPLIED"; echo "${nm},FAIL" >> "$SUM"; return 0; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"
  log "$nm = ${sr:-FAIL}  ($((t1-t0))s)"
}

log "=== upstream DINO-WM noprop + CEM, held-out eps 8000-9999, fp32, img 196 ==="
for d in 42 43 44; do run $d; done
log "DINOWM_CEM_DONE"
grep "^dinocem" "$SUM" | tee -a "$L/driver_dinowm_cem.log"
