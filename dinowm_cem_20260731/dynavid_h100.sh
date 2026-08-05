#!/bin/bash
# Long-horizon (h100) PRE vs POST-Dyna rollouts, same episode-disjoint split.
# goal = frame t+100 of the demo, 200-step budget (the h25 cells use 25/50, the
# earlier long-horizon capture used 200/400). NOTE: both actors were trained and
# tuned at the 25-step offset, so h100 is out of distribution for both arms.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; V=/workspace/videos_h100
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EVAL_RANGE=8000:10000
mkdir -p "$V" "$L"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_h100.log"; }

run(){ # name wm actor drawseed
  local nm=$1 md=$2 A=$3 d=$4 t0 t1
  [ -f "$V/$nm/env_0.mp4" ] && { log "$nm already rendered, skip"; return 0; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 14400 python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=100 \
    eval.eval_budget=200 "+eval.ep_range=$EVAL_RANGE" policy="$md" solver=lip \
    "solver.actor_path=$A" "+video_dir=$V/$nm" output.filename="${nm}.txt" \
    > "$L/vidh100_${nm}.log" 2>&1
  t1=$(date +%s)
  grep -q "ep_range 8000:10000" "$L/vidh100_${nm}.log" || { log "$nm: EP_RANGE NOT APPLIED"; return 1; }
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/vidh100_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  log "$nm = ${sr:-FAIL}  ($((t1-t0))s, $(ls "$V/$nm"/*.mp4 2>/dev/null | wc -l) mp4s)"
}

log "=== h100 capture start (goal_offset 100, budget 200) ==="
run pre_s2_e44  /workspace/models/v2WM            /workspace/actors/lip4_dsp_pre_s2.pt  44
run post_s2_e44 /workspace/models/dyna_full_5050  /workspace/actors/lip4_rep_full_s2.pt 44
run pre_s1_e42  /workspace/models/v2WM            /workspace/actors/lip4_dsp_pre_s1.pt  42
run post_s1_e42 /workspace/models/dyna_full_5050  /workspace/actors/lip4_rep_full_s1.pt 42
log "H100_CAPTURE_DONE"
