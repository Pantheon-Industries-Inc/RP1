#!/bin/bash
# Round S3: MLP-champion ablations — the two questions from the f_theta review.
#   zg0: --drop-zg   'vonly-ized' MLP: input [A, gradV, E, z0]; raw goal embedding
#        removed, goal reaches the actor only via the teacher. Tests whether the
#        champion's raw-z_g shortcut is load-bearing (and whether goal-agnostic
#        conditioning costs anything on the WINNING architecture).
#   ng:  --no-gate   pure residual A + dA on the MLP (default init -> the gate is
#        its only unroll damping; v3's gate ablation cost 5 pts, MLP never tested).
# 2 seeds each; paired refs: schedamax s0 = 168, s1 = 154 (mw/gate/zg all standard).
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

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90s3.log"; }

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

BASE="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm $WM --init-value $TD_WIN
      --horizon 5 --iters 8 --steps 8000 --n-step 50 --amax 3.5
      --expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4
      --actor-lr 3e-4 --actor-lr-final 3e-5"

s3_train() { # arm gpu extra...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac90s3_${arm}.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" $BASE \
    --out "$out" --out-value "$MET/lip_ac90s3_${arm}_value.pt" "$@" \
    > "$LOGS/train_lip_ac90s3_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
log "round S3: MLP ablations — drop-zg / no-gate x seeds 0/1 vs schedamax refs"
s3_train zg0_s0 0 --drop-zg --seed 0 &
s3_train zg0_s1 1 --drop-zg --seed 1 &
s3_train ng_s0  2 --no-gate --seed 0 &
s3_train ng_s1  3 --no-gate --seed 1 &
wait
log "round S2 trained"

ARMS="zg0_s0 zg0_s1 ng_s0 ng_s1"
for arm in $ARMS; do
  ef=$(grep -E "^step" "$LOGS/train_lip_ac90s3_${arm}.log" 2>/dev/null | tail -1 \
       | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "arm ${arm}: final E_final = ${ef:-NA}"
done
log "(ref) schedamax final E_final = 1.458 (champion)"

for s in 42 44; do
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac90s3_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac90s3_${arm}_h25_s${s}" "$gpu" "$s" 25 50 solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

for arm in $ARMS; do
  x=$(sc "lipac90s3_${arm}_h25_s42"); y=$(sc "lipac90s3_${arm}_h25_s44")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
  log "arm ${arm}: s42+s44 = $(awk "BEGIN{print $x + $y}")"
done
log "(ref) schedamax: s0 = 168, s1 = 154 (round B)"
log "DONE. round S2 complete — mw0/mw03 vs H's mw=0.1 on paired seeds"
