#!/bin/bash
# h50 probe of the two best v3 arms: does attention pay rent at longer horizon?
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1 MUJOCO_GL=osmesa
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
WM=/workspace/ckpts/ogbench_cube_single_v2WM
ev() { # name gpu actor
  grep -q "^$1," /workspace/results/summary.csv && return
  CUDA_VISIBLE_DEVICES=$2 python3 /workspace/code/stable-worldmodel/scripts/plan/eval_wm.py \
    --config-name cube seed=42 eval.dataset_name=$H5 ++bf16=true eval.img_size=224 \
    policy=$WM eval.goal_offset_steps=50 eval.eval_budget=100 \
    solver=lip solver.actor_path=$3 output.filename=$1.txt > /workspace/logs/eval_$1.log 2>&1
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" /workspace/logs/eval_$1.log | tail -1 | grep -oE "[0-9.]+$")
  echo "$1,${sr:-FAIL}" >> /workspace/results/summary.csv
  echo "[$(date +%H:%M:%S)] eval $1: ${sr:-FAIL}" >> /workspace/logs/driver_probe.log
}
ev probe_g_s2_h50_s42 0 /workspace/actors/lip_ac90g_v3g_s2.pt &
ev probe_h_s3_h50_s42 1 /workspace/actors/lip_ac90h_v3h_s3.pt &
wait
echo "[$(date +%H:%M:%S)] PROBE DONE" >> /workspace/logs/driver_probe.log
