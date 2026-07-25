#!/bin/bash
# Pod B eval-path smoke: waits for expert lance download, runs a 2-episode CEM
# eval on the e14 smoke model. Success marker: /workspace/B_EVAL_SMOKE_OK.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
mkdir -p /workspace/swm_home /workspace/logs
L=/workspace/logs/smoke_eval_b.log
for i in $(seq 1 240); do grep -q "DOWNLOADED" /workspace/logs/hf_download.log 2>/dev/null && break; sleep 30; done
grep -q "DOWNLOADED" /workspace/logs/hf_download.log || { echo "download never finished" > $L; exit 1; }
LANCE=$(find /workspace/datasets/ogb_cube_single -name "*.lance" -maxdepth 2 -type d | head -1)
echo "using dataset: $LANCE" > $L
cd /workspace/code/stable-worldmodel
CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 scripts/plan/eval_wm.py \
  --config-name cube seed=42 eval.dataset_name=$LANCE ++bf16=true \
  eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=2 eval.num_eval=2 \
  policy=/workspace/models/mix_plain_roll_e14_smoke solver=cem \
  output.filename=smoke_b.txt >> $L 2>&1
grep -oE "success_rate[^0-9]*[0-9.]+" $L | tail -1 > /workspace/B_EVAL_SMOKE_OK 2>/dev/null || echo "EVAL FAILED, see $L"
[ -s /workspace/B_EVAL_SMOKE_OK ] && echo "SMOKE OK: $(cat /workspace/B_EVAL_SMOKE_OK)" >> $L
