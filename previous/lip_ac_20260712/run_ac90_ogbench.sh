#!/bin/bash
# Cube h25 ">90 push" (4×H100). Baselines on this pod: AC-warm 88 (s42) / 76 (s44),
# sequential LIP v1 88/78 (writeup), champion h25 mean 87.3. Two attack lines:
#
#  Round A — eval-time compute, no retraining: restarts R=8 (+ robust_m=4) with
#    argmin-V selection, on BOTH lip_ac_warm and lip_n50_v1. Also probes the
#    dissociation: restart selection is a *ranking* task, so the drifted tandem
#    teacher (CEM 68) should benefit less/differently than the sequential (82).
#  Round B — during-run schedules + statics (all warm-started, K8/8k/n50):
#    sched      critic-lr cosine 1e-3->1e-4 + expectile anneal 0.1->0.03 over the
#               live phase (converge the teacher instead of letting it wander;
#               smooth early / sharp late) + actor-lr cosine 3e-4->3e-5
#    amax35     plan clamp 2.5->3.5 sigma (expert tails: 4-9%/dim exceed 2.5,
#               dim0 reaches 3.5 — the 2.5 clamp forbids used expert actions)
#    md5        goal window max-delta 5 blocks = exactly h25's 25 primitives
#    schedamax  sched + amax 3.5 combined
#  Selection on h25 s42+s44 (s44 = the binding draw, best-ever 78); winner gets
#  the best Round-A eval config stacked, confirmed on s42/43/44 h25 + s42 h50.
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

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90.log"; }
die() { log "FATAL: $*"; exit 1; }
for f in "$CACHE1" "$CACHE5" "$TD_WIN" "$ACT/lip_ac_warm.pt" "$ACT/lip_n50_v1.pt"; do
  [ -e "$f" ] || die "missing prerequisite: $f"
done

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
sum2() { # name1 name2 -> sum or -1 if any missing/FAIL
  local x y; x=$(sc "$1"); y=$(sc "$2")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && { echo -1; return; }
  awk "BEGIN{print $x + $y}"
}

# ---------------------------------------------------------------- round A: eval-time
log "round A: plain s44 baselines + restart configs on existing actors"
run_eval "lipac_warm_h25_s44" 0 44 25 50 solver=lip "solver.actor_path=$ACT/lip_ac_warm.pt" &
run_eval "anchor_lip_h25_s44" 1 44 25 50 solver=lip "solver.actor_path=$ACT/lip_n50_v1.pt" &
run_eval "rstA_warm_r8m0_h25_s42" 2 42 25 50 solver=lip "solver.actor_path=$ACT/lip_ac_warm.pt" solver.restarts=8 &
run_eval "rstA_warm_r8m4_h25_s42" 3 42 25 50 solver=lip "solver.actor_path=$ACT/lip_ac_warm.pt" solver.restarts=8 solver.robust_m=4 &
wait
run_eval "rstA_warm_r8m0_h25_s44" 0 44 25 50 solver=lip "solver.actor_path=$ACT/lip_ac_warm.pt" solver.restarts=8 &
run_eval "rstA_warm_r8m4_h25_s44" 1 44 25 50 solver=lip "solver.actor_path=$ACT/lip_ac_warm.pt" solver.restarts=8 solver.robust_m=4 &
run_eval "rstA_seq_r8m0_h25_s42" 2 42 25 50 solver=lip "solver.actor_path=$ACT/lip_n50_v1.pt" solver.restarts=8 &
run_eval "rstA_seq_r8m4_h25_s42" 3 42 25 50 solver=lip "solver.actor_path=$ACT/lip_n50_v1.pt" solver.restarts=8 solver.robust_m=4 &
wait
run_eval "rstA_seq_r8m0_h25_s44" 0 44 25 50 solver=lip "solver.actor_path=$ACT/lip_n50_v1.pt" solver.restarts=8 &
run_eval "rstA_seq_r8m4_h25_s44" 1 44 25 50 solver=lip "solver.actor_path=$ACT/lip_n50_v1.pt" solver.restarts=8 solver.robust_m=4 &
wait

plain_warm=$(sum2 "lipac_warm_h25_s42" "lipac_warm_h25_s44")
r8m0_warm=$(sum2 "rstA_warm_r8m0_h25_s42" "rstA_warm_r8m0_h25_s44")
r8m4_warm=$(sum2 "rstA_warm_r8m4_h25_s42" "rstA_warm_r8m4_h25_s44")
log "round A warm sums: plain=$plain_warm r8m0=$r8m0_warm r8m4=$r8m4_warm"
log "round A seq sums: plain=$(sum2 anchor_lip_h25_s42 anchor_lip_h25_s44) r8m0=$(sum2 rstA_seq_r8m0_h25_s42 rstA_seq_r8m0_h25_s44) r8m4=$(sum2 rstA_seq_r8m4_h25_s42 rstA_seq_r8m4_h25_s44)"
EVAL_EXTRA=(); EVAL_TAG=""
if awk "BEGIN{exit !($r8m4_warm > $plain_warm && $r8m4_warm >= $r8m0_warm)}"; then
  EVAL_EXTRA=(solver.restarts=8 solver.robust_m=4); EVAL_TAG="_r8m4"
elif awk "BEGIN{exit !($r8m0_warm > $plain_warm)}"; then
  EVAL_EXTRA=(solver.restarts=8); EVAL_TAG="_r8m0"
fi
log "chosen eval config: ${EVAL_TAG:-plain}"

# ---------------------------------------------------------------- round B: train
ac_train() { # arm gpu extra...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac90_${arm}.pt" outv="$MET/lip_ac90_${arm}_value.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM" \
    --out "$out" --out-value "$outv" --init-value "$TD_WIN" \
    --horizon 5 --iters 8 --steps 8000 --actor-lr 3e-4 --n-step 50 "$@" \
    > "$LOGS/train_lip_ac90_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
SCHED=(--expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4 --actor-lr-final 3e-5)
log "round B: training sched / amax35 / md5 / schedamax"
ac_train sched     0 "${SCHED[@]}" &
ac_train amax35    1 --expectile 0.03 --amax 3.5 &
ac_train md5       2 --expectile 0.03 --max-delta 5 &
ac_train schedamax 3 "${SCHED[@]}" --amax 3.5 &
wait
log "round B trained"

ARMS="sched amax35 md5 schedamax"
for s in 42 44; do
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac90_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac90_${arm}_h25_s${s}" "$gpu" "$s" 25 50 solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

# ---------------------------------------------------------------- pick + confirm
best="warm"; best_score=$plain_warm
for arm in $ARMS; do
  s=$(sum2 "lipac90_${arm}_h25_s42" "lipac90_${arm}_h25_s44")
  log "arm ${arm}: s42+s44 = ${s}"
  if awk "BEGIN{exit !($s > $best_score)}"; then best_score=$s; best=$arm; fi
done
BEST_PATH="$ACT/lip_ac_warm.pt"
[ "$best" != "warm" ] && BEST_PATH="$ACT/lip_ac90_${best}.pt"
log "round B winner: ${best} (s42+s44 = ${best_score}, baseline warm = ${plain_warm})"

log "confirm: winner ${best}${EVAL_TAG} on h25 x 3 seeds + h50 s42"
gpu=0
for s in 42 43 44; do
  run_eval "win90_${best}${EVAL_TAG}_h25_s${s}" "$gpu" "$s" 25 50 \
    solver=lip "solver.actor_path=$BEST_PATH" "${EVAL_EXTRA[@]}" &
  gpu=$(( (gpu + 1) % 4 ))
done
run_eval "win90_${best}${EVAL_TAG}_h50_s42" 3 42 50 100 \
  solver=lip "solver.actor_path=$BEST_PATH" "${EVAL_EXTRA[@]}" &
wait

m=$(awk "BEGIN{print ($(sc win90_${best}${EVAL_TAG}_h25_s42) + $(sc win90_${best}${EVAL_TAG}_h25_s43) + $(sc win90_${best}${EVAL_TAG}_h25_s44)) / 3}")
log "DONE. h25 mean of win90_${best}${EVAL_TAG} = ${m} (target >90; sequential champion 87.3)"
