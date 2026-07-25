#!/bin/bash
# Cube h25 >90 push, round 4: actor ARCHITECTURE change (PlannerNetV3, kind='lip3').
# Trajectory-aligned refiner: transformer over the H=5 plan blocks, token t =
# [A_t, grad_t, proj(z_t), proj(zg - z_t), E] + iteration embedding, per-token
# dA_t and gate. Fixes v1's single-scalar gate / flat-MLP limits; realigns the
# v2 feed per token. Same schedamax teacher recipe (winner of rounds 1-2).
# NO restarts anywhere (user constraint; also cube-negative).
#
# Round-2 lesson: 14pt training-seed spread on the 2-cell sum -> run 3 seeds of
# v3-base and compare RECIPE means, not single draws. Incumbent: schedamax
# seeds {0,1,2} = {168,154,168}, mean 163.3, best 168 (confirmed 88/96/80).
# Arms: v3 seeds 0/1/2 (w256 L2) + v3big (w384 L3, seed 0).
# Selection s42+s44; confirm best arm on 3 draws + h50 s42 if it beats 168.
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
WM=/workspace/ckpts/ogbench_cube_single_v2WM
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
CACHE1=/workspace/caches/cube_full_fs1.pt
CACHE5=/workspace/caches/cube_full_fs5.pt
TD_WIN=$MET/cf_dE_t003n50.pt
EVAL_TIMEOUT=14400

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90d.log"; }
die() { log "FATAL: $*"; exit 1; }

run_eval() { # name gpu seed offset budget extra...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    policy="$WM" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$RES/summary.csv"
    return 1
  fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}
sc() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }
sum2() {
  local x y; x=$(sc "$1"); y=$(sc "$2")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && { echo -1; return; }
  awk "BEGIN{print $x + $y}"
}

ac_train() { # arm gpu extra...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac90d_${arm}.pt" outv="$MET/lip_ac90d_${arm}_value.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM" \
    --out "$out" --out-value "$outv" --init-value "$TD_WIN" \
    --horizon 5 --iters 8 --steps 8000 --n-step 50 --arch traj --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 "$@" \
    > "$LOGS/train_lip_ac90d_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
log "round 4: v3.1-sep (quasimetric goal conditioning) seeds 0/1/2 + v3big"
ac_train v3sep_s0 0 --goal-mode sep --seed 0 &
ac_train v3sep_s1 1 --goal-mode sep --seed 1 &
ac_train v3sep_s2 2 --goal-mode sep --seed 2 &
ac_train v3sep_w384 3 --goal-mode sep --seed 0 --width 384 --layers 3 &
wait
log "round 4 trained"

ARMS="v3sep_s0 v3sep_s1 v3sep_s2 v3sep_w384"
for s in 42 44; do
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac90d_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac90d_${arm}_h25_s${s}" "$gpu" "$s" 25 50 solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

best="schedamax"; best_score=168; BEST_PATH="$ACT/lip_ac90_schedamax.pt"
# lift the incumbent to round-3's best if it beat schedamax
for arm3 in v3_s0 v3_s1 v3_s2 v3big; do
  s3=$(sum2 "lipac90c_${arm3}_h25_s42" "lipac90c_${arm3}_h25_s44")
  if awk "BEGIN{exit !($s3 > $best_score)}"; then
    best_score=$s3; best="r3_${arm3}"; BEST_PATH="$ACT/lip_ac90c_${arm3}.pt"
  fi
done
log "incumbent entering round 4: ${best} (${best_score})"
tot=0; n=0
for arm in $ARMS; do
  s=$(sum2 "lipac90d_${arm}_h25_s42" "lipac90d_${arm}_h25_s44")
  log "arm ${arm}: s42+s44 = ${s}"
  if [ "$arm" != "v3sep_w384" ] && awk "BEGIN{exit !($s >= 0)}"; then
    tot=$(awk "BEGIN{print $tot + $s}"); n=$((n + 1))
  fi
  if awk "BEGIN{exit !($s > $best_score)}"; then
    best_score=$s; best=$arm; BEST_PATH="$ACT/lip_ac90d_${arm}.pt"
  fi
done
[ "$n" -gt 0 ] && log "v3-base recipe mean over $n seeds: $(awk "BEGIN{print $tot / $n}") (schedamax 3-seed mean: 163.3)"
log "round 4 winner: ${best} (s42+s44 = ${best_score}, incumbent schedamax = 168)"

if [ "$best" != "schedamax" ]; then
  gpu=0
  for s in 42 43 44; do
    run_eval "win90d_${best}_h25_s${s}" "$gpu" "$s" 25 50 \
      solver=lip "solver.actor_path=$BEST_PATH" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  run_eval "win90d_${best}_h50_s42" 3 42 50 100 solver=lip "solver.actor_path=$BEST_PATH" &
  wait
  m=$(awk "BEGIN{print ($(sc win90d_${best}_h25_s42) + $(sc win90d_${best}_h25_s43) + $(sc win90d_${best}_h25_s44)) / 3}")
  log "DONE. round-4 winner ${best} h25 mean = ${m} (incumbent schedamax = 88.0; target >90)"
else
  log "DONE. no v3 arm beat schedamax's 168; architecture change did not clear the plateau"
fi
