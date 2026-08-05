#!/bin/bash
# THE MISSING CONTROL: what does this eval score with NO planning at all?
#
# Every LIP configuration -- ~20 of them across amax, lr, steps, expand, replay,
# and two different critics -- lands in a narrow 62-72 band, and draw 43 returns
# exactly 62.0 for three differently-trained actors. TD+CEM returns exactly 76.0
# with a critic 17% better than the one that also returned 76.0. That pattern is
# what you see when the learned components barely move the score.
#
# Notes from the earlier DINO-WM retrain campaign record "20k ckpt = 50.0 =
# FLOOR" on a comparable cube eval, so the floor here may be HIGH: with a 25-step
# goal offset the cube sometimes starts near its goal, and success is judged on
# the final frame.
#
#   floor ~60  -> LIP at 66 is barely above chance; the actor contributes almost
#                 nothing, and every LIP/TD+CEM comparison in this campaign has
#                 been resolving noise near a floor.
#   floor ~10  -> the learned planners are doing real work and the ordering
#                 (80 > 76 > 66) is a genuine ranking.
#
# Also evaluates a barely-trained (30-step) actor: if it matches the fully
# trained ones, actor training is irrelevant to the score, which is the same
# question from the other side.
#
# Waits for the in-flight eval to clear -- concurrent EGL evals corrupt silently.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
P=/workspace/code/stable-worldmodel/scripts/plan; L=/workspace/logs; R=/workspace/results
EXPERT=/root/datasets/ogb_cube_single/ogb_cube_single.lance
CKPT=/workspace/ckpts/dinowm_noprop_cube
SUM=$R/summary_floor.csv
mkdir -p "$R" /workspace/videos_scratch; touch "$SUM"
cd /workspace/code/stable-worldmodel
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_floor.log"; }

log "=== waiting for the in-flight eval queue to clear (EGL is exclusive) ==="
for i in $(seq 1 240); do
  pgrep -f "eval_wm_din[o]" >/dev/null || break
  sleep 60
done
log "  GPU free"

ev(){ # name  extra...
  local nm=$1; shift
  grep -q "^${nm}," "$SUM" && { log "  $nm cached"; return 0; }
  local t0 t1 sr
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 timeout 7200 python3 "$P/eval_wm_dino.py" --config-name cube \
    eval.dataset_name="$EXPERT" eval.img_size=196 eval.goal_offset_steps=25 \
    eval.eval_budget=50 "+eval.ep_range=8000:10000" \
    "+video_dir=/workspace/videos_scratch/$nm" output.filename="${nm}.txt" "$@" \
    > "$L/eval_${nm}.log" 2>&1
  t1=$(date +%s)
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$L/eval_${nm}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${nm},${sr:-FAIL}" >> "$SUM"
  log "  $nm = ${sr:-FAIL} ($(( (t1-t0)/60 ))m)"
  if [ -z "$sr" ]; then
    log "    ERR $(grep -E 'Error|Traceback' "$L/eval_${nm}.log" | tail -1)"
  fi
}

log "=== FLOORS: no planning at all (refs: plainCEM 80.0, TD+CEM 76.0, LIP 66.0) ==="
for d in 44 42 43; do ev "nomove_e${d}" seed=$d policy=nomove; done
for d in 44 42 43; do ev "random_e${d}" seed=$d policy=random; done

log "=== UNTRAINED-ACTOR CONTROL: 30-step actor vs the 3000-step ones ==="
if [ -f /tmp/fix_smoke.pt ]; then
  for d in 44 42 43; do
    ev "lip30step_e${d}" seed=$d policy="$CKPT" solver=lip \
       "solver.actor_path=/tmp/fix_smoke.pt"
  done
else
  log "  /tmp/fix_smoke.pt gone (container recycled); skipping"
fi

log "FLOOR_DONE"
sort "$SUM" | tee -a "$L/driver_floor.log"
