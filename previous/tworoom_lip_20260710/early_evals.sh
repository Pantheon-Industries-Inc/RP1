#!/bin/bash
# Early selection evals for the 6 already-trained lv1 actors (rows are
# name-compatible with run_hypers.sh, which will see them as cached).
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results; ACT=/workspace/actors
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/hypers.log"; }
run_eval() {
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  grep -q "^${name}," "$RES/summary.csv" && return 0
  local hard=(); [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu python3 "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "+video_dir=$RES/videos_${name}" \
    "${hard[@]}" "$@" > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}
gpu0() { for tag in lv1_t-e01n50 lv1_t-e003n25 lv1_t-ed256; do
  run_eval "${tag}_hard_h50_s43" 0 43 50 100 hard solver=lip "solver.actor_path=$ACT/${tag}.pt"
  run_eval "${tag}_hard_h50_s42" 0 42 50 100 hard solver=lip "solver.actor_path=$ACT/${tag}.pt"
  run_eval "${tag}_std_h50_s42"  0 42 50 100 std  solver=lip "solver.actor_path=$ACT/${tag}.pt"
done; }
gpu1() { for tag in lv1_t-pc05 lv1_r-k12 lv1_r-16k; do
  run_eval "${tag}_hard_h50_s43" 1 43 50 100 hard solver=lip "solver.actor_path=$ACT/${tag}.pt"
  run_eval "${tag}_hard_h50_s42" 1 42 50 100 hard solver=lip "solver.actor_path=$ACT/${tag}.pt"
  run_eval "${tag}_std_h50_s42"  1 42 50 100 std  solver=lip "solver.actor_path=$ACT/${tag}.pt"
done; }
gpu0 & gpu1 & wait
log "early evals done"
