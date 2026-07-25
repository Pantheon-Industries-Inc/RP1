#!/usr/bin/env bash
# Decisive test for the 3-parallel-eval SIGABRT hypothesis.
#
# Runs 3 concurrent processes x N EGL contexts (mimicking 3 concurrent 50-env
# evals), once WITHOUT MUJOCO_EGL_DEVICE_ID and once WITH it, and reports where
# the GPU memory landed.
#
# Predicted: without the var, all 3 processes pile onto ONE physical GPU
# (memory rises only there); with it, memory spreads across GPU 0/1/2.
#
# Usage: ./egl_device_test.sh [n_contexts] [hold_seconds]
set -u
N=${1:-50}
HOLD=${2:-25}
cd /workspace

mem(){ nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' | '; }

run_arm(){ # label  set_egl_id(0|1)
  local label=$1 set_id=$2
  echo "=================================================================="
  echo "ARM: $label   (${N} contexts x 3 processes)"
  echo "baseline mem: $(mem)"
  local pids=() rc=0
  for g in 0 1 2; do
    if [ "$set_id" = 1 ]; then
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g \
        python3 egl_device_probe.py "$N" "$HOLD" 2>/tmp/egl_err_$g.txt | grep '^{' &
    else
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$g \
        python3 egl_device_probe.py "$N" "$HOLD" 2>/tmp/egl_err_$g.txt | grep '^{' &
    fi
    pids+=($!)
  done
  sleep $(( HOLD / 2 ))
  echo "MEM WHILE HELD: $(mem)"
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "after: $(mem)"
  echo "exit-status-nonzero: $rc"
  for g in 0 1 2; do
    if [ -s /tmp/egl_err_$g.txt ]; then
      echo "--- stderr gpu$g (first 3 lines) ---"; head -3 /tmp/egl_err_$g.txt
    fi
  done
}

run_arm "MUJOCO_EGL_DEVICE_ID UNSET (current production behaviour)" 0
sleep 5
run_arm "MUJOCO_EGL_DEVICE_ID=\$gpu (proposed fix)" 1
echo "=================================================================="
echo "DONE"
