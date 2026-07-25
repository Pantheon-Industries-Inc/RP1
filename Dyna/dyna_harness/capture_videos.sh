#!/bin/bash
# Capture failure-case videos (agent|demo|goal panels) for analysis.
#   h25  lip4_r2w_s1 on e44 (fails tasks 10,11,12) — residual grasp failures
#   h200 lip4_r2w_s1 on e42 (all fail)             — long-horizon collapse
# egl render on idle pod, GPU0/GPU1 in parallel (<=2 evals ok).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1 OMP_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
WM=/workspace/models/wm2_e2; ACTOR=/workspace/actors/lip4_r2w_s1.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
LOGS=/workspace/logs
cd "$CODE"

( CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$PLAN/eval_wm.py" --config-name cube \
    seed=44 eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 policy="$WM" solver=lip \
    "solver.actor_path=$ACTOR" +video_dir=/workspace/videos/h25_s1_e44 \
    output.filename=vid_h25.txt > "$LOGS/vid_h25.log" 2>&1
  echo "H25_VID_DONE" >> "$LOGS/vid_h25.log" ) &

( CUDA_VISIBLE_DEVICES=1 timeout 10800 python3 "$PLAN/eval_wm.py" --config-name cube \
    seed=42 eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=200 eval.eval_budget=400 eval.num_eval=8 policy="$WM" solver=lip \
    "solver.actor_path=$ACTOR" +video_dir=/workspace/videos/h200_s1_e42 \
    output.filename=vid_h200.txt > "$LOGS/vid_h200.log" 2>&1
  echo "H200_VID_DONE" >> "$LOGS/vid_h200.log" ) &
wait
echo "CAPTURE_DONE" >> "$LOGS/vid_h25.log"
ls /workspace/videos/h25_s1_e44/ /workspace/videos/h200_s1_e42/ 2>/dev/null | head
