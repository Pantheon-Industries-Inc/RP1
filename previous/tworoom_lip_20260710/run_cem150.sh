#!/bin/bash
# CEM with 150 samples (half budget): latent+CEM and TD+CEM on all 12 cells.
# Runs strictly one eval at a time on GPU1 to stay within safe concurrency.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results
TD=/workspace/metrics/td_e0.03_n50.pt   # same value as the table's TD+CEM row
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/cem150.log"; }

run_eval() { # name seed offset budget surface extra...
  local name=$1 seed=$2 offset=$3 budget=$4 surface=$5; shift 5
  grep -q "^${name}," "$RES/summary.csv" && return 0
  local hard=(); [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=1 python3 "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 solver.num_samples=150 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}

log "CEM-150 round start (latent + TD, 12 cells each)"
for surface in std hard; do
  for seed in 42 43 44; do
    run_eval "latent150_${surface}_h25_s${seed}" "$seed" 25 50 "$surface"
    run_eval "latent150_${surface}_h50_s${seed}" "$seed" 50 100 "$surface"
    run_eval "tdcem150_${surface}_h25_s${seed}" "$seed" 25 50 "$surface" "+metric=$TD"
    run_eval "tdcem150_${surface}_h50_s${seed}" "$seed" 50 100 "$surface" "+metric=$TD"
  done
done
log "CEM150 DONE"
grep -E "^(latent150|tdcem150)" "$RES/summary.csv" | sort
