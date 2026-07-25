#!/bin/bash
# Pure LIP-v1 hyper sweep to close the h50 gap (no restarts/selection/MPPI at deploy).
# Axis 1 (teacher): LIP recipe fixed K8 lr3e-4 8k, vary the TD value it distills.
# Axis 2 (recipe):  teacher fixed td_e0.03_n50, vary LIP training hypers.
# Selection: sum over {hard h50 s42, hard h50 s43, std h50 s42} (s43-hard is the weak draw).
# Reference: current winner lip_k8_lr3e-4_st8000 scores 100+94+98 = 292 on these cells.
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
CACHE1=/workspace/caches/tworoom_lewm_fs1.pt
CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TD0=$MET/td_e0.03_n50.pt

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/hypers.log"; }

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

lip_train() { # out gpu value extra-args...
  local out=$1 gpu=$2 value=$3; shift 3
  [ -f "$out" ] && { log "train $(basename "$out"): cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip.py" --cache "$CACHE5" --h5 "$H5" \
    --wm lewm_tworoom --value "$value" --lr 3e-4 "$@" --out "$out" \
    > "$LOGS/$(basename "$out" .pt).log" 2>&1 \
    && log "trained $(basename "$out")" || log "train $(basename "$out") FAILED"
}

# ---------------- TD variants (teacher axis) ---------------------------------
log "TD variant training"
if [ ! -f "$MET/td_e0.03_n50_ed256st12k.pt" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$PLAN/train_metric.py" --cache "$CACHE1" --learner td \
    --head quasimetric --expectile 0.03 --n-step 50 --embed-dim 256 --steps 12000 \
    --out "$MET/td_e0.03_n50_ed256st12k.pt" > "$LOGS/td_ed256st12k.log" 2>&1 \
    && log "trained td ed256st12k" || log "td ed256st12k FAILED"
fi &
if [ ! -f "$MET/td_e0.03_n50_pc05.pt" ]; then
  CUDA_VISIBLE_DEVICES=1 $PY "$PLAN/train_metric.py" --cache "$CACHE1" --learner td \
    --head quasimetric --expectile 0.03 --n-step 50 --p-cross 0.5 --steps 6000 \
    --out "$MET/td_e0.03_n50_pc05.pt" > "$LOGS/td_pc05.log" 2>&1 \
    && log "trained td pc05" || log "td pc05 FAILED"
fi &
wait

# ---------------- LIP trainings (all v1: --feed none is the default) ----------
log "LIP trainings: teacher axis (recipe K8 lr3e-4 8k)"
lip_train "$ACT/lv1_t-e01n50.pt"  0 "$MET/td_e0.1_n50.pt"             --horizon 5 --iters 8 --steps 8000 &
lip_train "$ACT/lv1_t-e003n25.pt" 1 "$MET/td_e0.03_n25.pt"            --horizon 5 --iters 8 --steps 8000 &
wait
lip_train "$ACT/lv1_t-ed256.pt"   0 "$MET/td_e0.03_n50_ed256st12k.pt" --horizon 5 --iters 8 --steps 8000 &
lip_train "$ACT/lv1_t-pc05.pt"    1 "$MET/td_e0.03_n50_pc05.pt"       --horizon 5 --iters 8 --steps 8000 &
wait
log "LIP trainings: recipe axis (teacher td_e0.03_n50)"
lip_train "$ACT/lv1_r-k12.pt"     0 "$TD0" --horizon 5  --iters 12 --steps 8000 &
lip_train "$ACT/lv1_r-16k.pt"     1 "$TD0" --horizon 5  --iters 8  --steps 16000 &
wait
lip_train "$ACT/lv1_r-md12.pt"    0 "$TD0" --horizon 5  --iters 8  --steps 8000 --max-delta 12 &
lip_train "$ACT/lv1_r-h10.pt"     1 "$TD0" --horizon 10 --iters 8  --steps 8000 &
wait
log "all trainings done"

# ---------------- selection evals ---------------------------------------------
log "selection evals (hard h50 s42+s43, std h50 s42)"
sel_eval() { # actor-basename extra...
  local tag=$1; shift
  local a="$ACT/${tag}.pt"
  [ -f "$a" ] || { log "skip ${tag} (no actor)"; return 0; }
  run_eval "${tag}_hard_h50_s42" 0 42 50 100 hard solver=lip "solver.actor_path=$a" "$@" &
  run_eval "${tag}_hard_h50_s43" 1 43 50 100 hard solver=lip "solver.actor_path=$a" "$@" &
  wait
  run_eval "${tag}_std_h50_s42" 0 42 50 100 std solver=lip "solver.actor_path=$a" "$@"
}
sel_eval lv1_t-e01n50
sel_eval lv1_t-e003n25
sel_eval lv1_t-ed256
sel_eval lv1_t-pc05
sel_eval lv1_r-k12
sel_eval lv1_r-16k
sel_eval lv1_r-md12
sel_eval lv1_r-h10 plan_config.horizon=10

# ---------------- winner: best selection sum vs reference 292 -----------------
best=""; best_score=291  # must beat the current winner's 292 minus nothing: > 291 keeps ties out
for tag in lv1_t-e01n50 lv1_t-e003n25 lv1_t-ed256 lv1_t-pc05 lv1_r-k12 lv1_r-16k lv1_r-md12 lv1_r-h10; do
  a=$(grep "^${tag}_hard_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  b=$(grep "^${tag}_hard_h50_s43," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  c=$(grep "^${tag}_std_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  { [ -z "$a" ] || [ "$a" = "FAIL" ] || [ -z "$b" ] || [ "$b" = "FAIL" ] || [ -z "$c" ] || [ "$c" = "FAIL" ]; } && continue
  s=$(awk "BEGIN{print $a + $b + $c}")
  log "candidate ${tag}: selection sum ${s}"
  if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best=$tag; fi
done

if [ -z "$best" ]; then
  log "NO candidate beat the reference (292) — current winner stands"
else
  log "hyper winner: ${best} (selection sum ${best_score}); confirming on all cells"
  EXTRA=()
  [ "$best" = "lv1_r-h10" ] && EXTRA=(plan_config.horizon=10)
  A="$ACT/${best}.pt"
  for surface in hard std; do
    run_eval "lv1win_${best}_${surface}_h25_s42" 0 42 25 50 "$surface" solver=lip "solver.actor_path=$A" "${EXTRA[@]}" &
    run_eval "lv1win_${best}_${surface}_h25_s43" 1 43 25 50 "$surface" solver=lip "solver.actor_path=$A" "${EXTRA[@]}" &
    wait
    run_eval "lv1win_${best}_${surface}_h25_s44" 0 44 25 50 "$surface" solver=lip "solver.actor_path=$A" "${EXTRA[@]}" &
    run_eval "lv1win_${best}_${surface}_h50_s42" 1 42 50 100 "$surface" solver=lip "solver.actor_path=$A" "${EXTRA[@]}" &
    wait
    run_eval "lv1win_${best}_${surface}_h50_s43" 0 43 50 100 "$surface" solver=lip "solver.actor_path=$A" "${EXTRA[@]}" &
    run_eval "lv1win_${best}_${surface}_h50_s44" 1 44 50 100 "$surface" solver=lip "solver.actor_path=$A" "${EXTRA[@]}" &
    wait
  done
fi

log "HYPERS DONE"
grep -E "^lv1" "$RES/summary.csv" | sort
