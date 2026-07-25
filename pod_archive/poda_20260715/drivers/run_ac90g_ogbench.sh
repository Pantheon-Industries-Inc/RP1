#!/bin/bash
# Cube h25 >90 push, round G: the elegant-init resolution (small-init, no priors).
# Identical vonly tokens + SCALAR iteration k/(K-1) (new default) + plain v1-form
# v3.2: tokens = [A_t, ghat_t, z_t] (242d); E and k enter via learned global bias
# (Linear(2->256) added to all tokens, dedicated gradient path); per-step V dropped.
# Fixes: (1) init -> --head-scale 0.01 near-identity, live grads; (2) scale imbalance
# -> --feat-norm (RMS-normalized grad feature, E/25, token LayerNorm).
# = near-identity start with live gradients everywhere. No GD prior, no residual
# split, no zero-init dead paths, no embedding patches. 4 seeds, schedamax teacher.
# Compares against: round V (default init, iter-emb) and round Z (zero-init, iter-emb)
# on the same s42+s44 selection; incumbent best = schedamax 168 / mean 163.3.
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

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90g.log"; }
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
  local out="$ACT/lip_ac90g_${arm}.pt" outv="$MET/lip_ac90g_${arm}_value.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM" \
    --out "$out" --out-value "$outv" --init-value "$TD_WIN" \
    --horizon 5 --iters 8 --steps 8000 --n-step 50 \
    --arch traj --goal-mode vonly --cond-mode global --feat-norm --head-scale 0.01 --amax 3.5 \
    --expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 "$@" \
    > "$LOGS/train_lip_ac90g_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
log "round G: vonly tokens [A_t, grad_t, raw z_t, V(z_t,zg)] + iter-emb; seeds 0-3"
ac_train v3g_s0 0 --seed 0 &
ac_train v3g_s1 1 --seed 1 &
ac_train v3g_s2 2 --seed 2 &
ac_train v3g_s3 3 --seed 3 &
wait
log "round G trained"

ARMS="v3g_s0 v3g_s1 v3g_s2 v3g_s3"
for s in 42 44; do
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac90g_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac90g_${arm}_h25_s${s}" "$gpu" "$s" 25 50 solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

best="schedamax"; best_score=168; BEST_PATH="$ACT/lip_ac90_schedamax.pt"
tot=0; n=0
for arm in $ARMS; do
  s=$(sum2 "lipac90g_${arm}_h25_s42" "lipac90g_${arm}_h25_s44")
  log "arm ${arm}: s42+s44 = ${s}"
  if awk "BEGIN{exit !($s >= 0)}"; then
    tot=$(awk "BEGIN{print $tot + $s}"); n=$((n + 1))
  fi
  if awk "BEGIN{exit !($s > $best_score)}"; then
    best_score=$s; best=$arm; BEST_PATH="$ACT/lip_ac90g_${arm}.pt"
  fi
done
[ "$n" -gt 0 ] && log "vonly recipe mean over $n seeds: $(awk "BEGIN{print $tot / $n}") (schedamax 3-seed mean: 163.3)"
for arm in v3v_s0 v3v_s1 v3v_s2 v3v_s3 v3z_s0 v3z_s1 v3z_s2 v3z_s3; do
  s=$(sum2 "lipac90v_${arm}_h25_s42" "lipac90v_${arm}_h25_s44")
  log "(ref) ${arm}: s42+s44 = ${s}"
done
log "round G winner: ${best} (s42+s44 = ${best_score}, incumbent schedamax = 168)"

if [ "$best" != "schedamax" ]; then
  gpu=0
  for s in 42 43 44; do
    run_eval "win90g_${best}_h25_s${s}" "$gpu" "$s" 25 50 \
      solver=lip "solver.actor_path=$BEST_PATH" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  run_eval "win90g_${best}_h50_s42" 3 42 50 100 solver=lip "solver.actor_path=$BEST_PATH" &
  wait
  m=$(awk "BEGIN{print ($(sc win90g_${best}_h25_s42) + $(sc win90g_${best}_h25_s43) + $(sc win90g_${best}_h25_s44)) / 3}")
  log "DONE. round-V winner ${best} h25 mean = ${m} (incumbent schedamax = 88.0; target >90)"
else
  log "DONE. no vonly arm beat schedamax's 168 on s42+s44"
fi
