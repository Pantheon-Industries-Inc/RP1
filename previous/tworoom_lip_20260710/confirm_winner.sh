#!/bin/bash
# Pre-run the 12-cell confirmation of the hyper winner (lv1_t-e01n50) on GPU0.
# Row names identical to run_hypers.sh's confirm block -> it will see them as cached.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results
A=/workspace/actors/lv1_t-e01n50.pt
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/hypers.log"; }
run_eval() {
  local name=$1 seed=$2 offset=$3 budget=$4 surface=$5
  grep -q "^${name}," "$RES/summary.csv" && return 0
  local hard=(); [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=0 python3 "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "+video_dir=$RES/videos_${name}" \
    "${hard[@]}" solver=lip "solver.actor_path=$A" > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}
B=lv1_t-e01n50
for surface in hard std; do
  run_eval "lv1win_${B}_${surface}_h25_s42" 42 25 50 "$surface"
  run_eval "lv1win_${B}_${surface}_h25_s43" 43 25 50 "$surface"
  run_eval "lv1win_${B}_${surface}_h25_s44" 44 25 50 "$surface"
  run_eval "lv1win_${B}_${surface}_h50_s42" 42 50 100 "$surface"
  run_eval "lv1win_${B}_${surface}_h50_s43" 43 50 100 "$surface"
  run_eval "lv1win_${B}_${surface}_h50_s44" 44 50 100 "$surface"
done
log "winner pre-confirm done"
