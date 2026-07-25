#!/bin/bash
# Capture clean AGENT pixels via the SWM_RECORD_PATH hook (proven clean, unlike
# the harness video panel). h25 (lip4_r2w_s1 e44) then h200 (e42). osmesa, GPU0.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel; PLAN=$CODE/scripts/plan
WM=/workspace/models/wm2_e2; ACTOR=/workspace/actors/lip4_r2w_s1.pt
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
LOGS=/workspace/logs
cd "$CODE"
rm -rf /workspace/videos/rec_h25.lance /workspace/videos/rec_h200.lance

SWM_RECORD_PATH=/workspace/videos/rec_h25.lance CUDA_VISIBLE_DEVICES=0 timeout 5400 \
  python3 "$PLAN/eval_wm.py" --config-name cube seed=44 eval.dataset_name="$EXPERT" \
  ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
  policy="$WM" solver=lip "solver.actor_path=$ACTOR" output.filename=rec_h25.txt \
  > "$LOGS/rec_h25.log" 2>&1
echo "H25 rec: $(grep -oE 'success_rate[^0-9]*[0-9.]+' $LOGS/rec_h25.log | tail -1) | $(grep -oE '\[record\].*' $LOGS/rec_h25.log | tail -1)" >> "$LOGS/rec_capture.log"

SWM_RECORD_PATH=/workspace/videos/rec_h200.lance CUDA_VISIBLE_DEVICES=0 timeout 10800 \
  python3 "$PLAN/eval_wm.py" --config-name cube seed=42 eval.dataset_name="$EXPERT" \
  ++bf16=true eval.img_size=224 eval.goal_offset_steps=200 eval.eval_budget=400 eval.num_eval=8 \
  policy="$WM" solver=lip "solver.actor_path=$ACTOR" output.filename=rec_h200.txt \
  > "$LOGS/rec_h200.log" 2>&1
echo "H200 rec: $(grep -oE 'success_rate[^0-9]*[0-9.]+' $LOGS/rec_h200.log | tail -1) | $(grep -oE '\[record\].*' $LOGS/rec_h200.log | tail -1)" >> "$LOGS/rec_capture.log"
echo "REC_ALL_DONE" >> "$LOGS/rec_capture.log"
