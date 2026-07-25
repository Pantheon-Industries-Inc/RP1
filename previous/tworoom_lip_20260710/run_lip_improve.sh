#!/bin/bash
# LIP improvement round: close the h50 gap to TD+CEM.
#   A: restart-selection eval of the existing winner (no retraining)
#   B: train h50-native LIP (horizon 10 blocks) + longer-trained K8 (16k steps)
#   C: eval both new actors on the h50 cells (+h25 spot checks)
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
H5=/workspace/caches/tworoom_play.h5
CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TDWIN=$MET/td_e0.03_n50.pt
LIPWIN=$ACT/lip_k8_lr3e-4_st8000.pt

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/improve.log"; }

run_eval() { # name gpu seed offset budget surface(std|hard) extra...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (see $LOGS/eval_${name}.log)"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1
  fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

# ---------------- B: trainings first so GPU1 works through phase A ------------
train_b() {
  # h50-native LIP: plan horizon 10 blocks (=50 primitive steps)
  if [ ! -f "$ACT/lip_h10_k8_lr3e-4_st8000.pt" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
      --wm lewm_tworoom --value "$TDWIN" --horizon 10 --iters 8 \
      --steps 8000 --lr 3e-4 --out "$ACT/lip_h10_k8_lr3e-4_st8000.pt" \
      > "$LOGS/lip_h10_k8_lr3e-4_st8000.log" 2>&1 \
      && log "trained lip_h10_k8" || log "lip_h10_k8 train FAILED"
  fi
  # same recipe as winner, 2x steps
  if [ ! -f "$ACT/lip_k8_lr3e-4_st16000.pt" ]; then
    CUDA_VISIBLE_DEVICES=1 $PY "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
      --wm lewm_tworoom --value "$TDWIN" --horizon 5 --iters 8 \
      --steps 16000 --lr 3e-4 --out "$ACT/lip_k8_lr3e-4_st16000.pt" \
      > "$LOGS/lip_k8_lr3e-4_st16000.log" 2>&1 \
      && log "trained lip_k8_16k" || log "lip_k8_16k train FAILED"
  fi
}
log "improvement round start"
train_b &
TRAIN_PID=$!

# ---------------- A: restart-selection on the existing winner (GPU0) ----------
log "A: restarts=8 argmin-V selection on existing winner"
run_eval "lipR8_k8_hard_h50_s43" 0 43 50 100 hard solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
run_eval "lipR8_k8_hard_h50_s42" 0 42 50 100 hard solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
run_eval "lipR8_k8_hard_h50_s44" 0 44 50 100 hard solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
run_eval "lipR8_k8_std_h50_s42" 0 42 50 100 std solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
run_eval "lipR8_k8_std_h50_s43" 0 43 50 100 std solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
run_eval "lipR8_k8_std_h50_s44" 0 44 50 100 std solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
# sanity: restarts must not regress h25
run_eval "lipR8_k8_hard_h25_s42" 0 42 25 50 hard solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8
# robustified selection on the worst cell
run_eval "lipR8m4_k8_hard_h50_s43" 0 43 50 100 hard solver=lip "solver.actor_path=$LIPWIN" solver.restarts=8 solver.robust_m=4

wait "$TRAIN_PID"
log "trainings done"

# ---------------- C: eval the new actors ------------------------------------
H10=$ACT/lip_h10_k8_lr3e-4_st8000.pt
K16=$ACT/lip_k8_lr3e-4_st16000.pt
if [ -f "$H10" ]; then
  for seed in 42 43 44; do
    run_eval "lip_h10_hard_h50_s${seed}" 0 "$seed" 50 100 hard solver=lip "solver.actor_path=$H10" plan_config.horizon=10 &
    run_eval "lip_h10_std_h50_s${seed}" 1 "$seed" 50 100 std solver=lip "solver.actor_path=$H10" plan_config.horizon=10 &
    wait
  done
  run_eval "lip_h10_hard_h25_s42" 0 42 25 50 hard solver=lip "solver.actor_path=$H10" plan_config.horizon=10
fi
if [ -f "$K16" ]; then
  for seed in 42 43 44; do
    run_eval "lip_k16k_hard_h50_s${seed}" 0 "$seed" 50 100 hard solver=lip "solver.actor_path=$K16" &
    run_eval "lip_k16k_std_h50_s${seed}" 1 "$seed" 50 100 std solver=lip "solver.actor_path=$K16" &
    wait
  done
fi
# best-of: restarts on the h10 actor for the hard h50 cells
if [ -f "$H10" ]; then
  for seed in 42 43 44; do
    run_eval "lipR8_h10_hard_h50_s${seed}" 0 "$seed" 50 100 hard solver=lip "solver.actor_path=$H10" plan_config.horizon=10 solver.restarts=8 &
    run_eval "lipR8_h10_std_h50_s${seed}" 1 "$seed" 50 100 std solver=lip "solver.actor_path=$H10" plan_config.horizon=10 solver.restarts=8 &
    wait
  done
fi

log "IMPROVE DONE"
grep -E "^lipR8|^lip_h10|^lip_k16k" "$RES/summary.csv" | sort
