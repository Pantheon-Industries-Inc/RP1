#!/bin/bash
# REAL reproduction of the 3-parallel-eval SIGABRT, and the MUJOCO_EGL_DEVICE_ID fix.
#
# Runs the same 3 concurrent 50-env evals that collect_r1.sh launches at PAR=3,
# once WITHOUT MUJOCO_EGL_DEVICE_ID (current production) and once WITH it.
# Reports per-arm exit codes -- 134 = SIGABRT, which is the failure the driver
# can only see as "rc != 0".
#
# Deliberately CEM (solver=cem, no actor/value needed) so this depends on
# nothing but the WM + expert lance. eval_budget kept small for turnaround.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
WM=/workspace/models/dyna_r1_5050b
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
LOGS=/workspace/logs/egltest
BUDGET=${1:-10}          # env steps per episode; 10 is enough to build all contexts
mkdir -p "$LOGS"

mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' | '; }

one_eval(){ # gpu seed set_egl_id
  local gpu=$1 seed=$2 set_id=$3
  local extra=""
  [ "$set_id" = 1 ] && extra="MUJOCO_EGL_DEVICE_ID=$gpu"
  env $extra CUDA_VISIBLE_DEVICES=$gpu timeout 900 python3 "$PLAN/eval_wm.py" \
    --config-name cube seed="$seed" eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=$BUDGET \
    policy="$WM" solver=cem output.filename="egltest_g${gpu}_${set_id}.txt" \
    > "$LOGS/eval_g${gpu}_set${set_id}.log" 2>&1
  echo "$gpu:$?"
}

run_arm(){ # label set_egl_id
  local label=$1 set_id=$2
  echo "=================================================================="
  echo "ARM: $label"
  echo "baseline mem: $(mem)"
  local t0=$SECONDS
  local out
  out=$( { one_eval 0 42 "$set_id" & one_eval 1 43 "$set_id" & one_eval 2 44 "$set_id" & wait; } )
  echo "exit codes (gpu:rc)  -- 134 = SIGABRT:"
  echo "$out" | sort
  echo "elapsed: $((SECONDS - t0))s"
  echo "mem after: $(mem)"
  for g in 0 1 2; do
    local L="$LOGS/eval_g${g}_set${set_id}.log"
    local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L" 2>/dev/null | tail -1)
    echo "  gpu$g: ${sr:-<no success_rate>}"
    grep -iE "abort|GLError|EGL_BAD|out of memory|Segmentation|Failed to make" "$L" 2>/dev/null \
      | grep -v "Exception ignored" | head -2 | sed 's/^/      /'
  done
}

run_arm "MUJOCO_EGL_DEVICE_ID UNSET (current production behaviour)" 0
sleep 10
run_arm "MUJOCO_EGL_DEVICE_ID=\$gpu (proposed fix)" 1
echo "=================================================================="
echo "DONE"
