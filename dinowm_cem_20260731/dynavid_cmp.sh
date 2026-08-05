#!/bin/bash
# PRE vs POST-Dyna rollout videos, episode-disjoint split (eval eps 8000-9999).
# Same draw seed for both arms => identical start states + goals => paired video.
# Sequential (egl evals must never overlap). Cells chosen from the campaign
# per-task success diff: s2/e44 = 80.0 -> 92.0, six clean gains, zero regressions.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs; V=/workspace/videos_dynacmp
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
EVAL_RANGE=8000:10000
mkdir -p "$V" "$L"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_dynavid.log"; }

run(){ # name wm actor drawseed
  local nm=$1 md=$2 A=$3 d=$4 t0 t1
  [ -f "$V/$nm/env_0.mp4" ] && { log "$nm already rendered, skip"; return 0; }
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm.py" --config-name cube seed=$d \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=$EVAL_RANGE" policy="$md" solver=lip \
    "solver.actor_path=$A" "+video_dir=$V/$nm" output.filename="${nm}.txt" \
    > "$L/vidcmp_${nm}.log" 2>&1
  t1=$(date +%s)
  grep -q "ep_range 8000:10000" "$L/vidcmp_${nm}.log" || { log "$nm: EP_RANGE NOT APPLIED"; return 1; }
  grep -q "MUJOCO_GL=egl" "$L/vidcmp_${nm}.log" || log "$nm: WARNING not egl"
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/vidcmp_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  log "$nm = ${sr:-FAIL}  ($((t1-t0))s, $(ls "$V/$nm"/*.mp4 2>/dev/null | wc -l) mp4s)"
}

log "=== dyna video capture start ==="
run pre_s2_e44  /workspace/models/v2WM            /workspace/actors/lip4_dsp_pre_s2.pt  44
run post_s2_e44 /workspace/models/dyna_full_5050  /workspace/actors/lip4_rep_full_s2.pt 44
run pre_s1_e42  /workspace/models/v2WM            /workspace/actors/lip4_dsp_pre_s1.pt  42
run post_s1_e42 /workspace/models/dyna_full_5050  /workspace/actors/lip4_rep_full_s1.pt 42
log "DYNAVID_CAPTURE_DONE"
