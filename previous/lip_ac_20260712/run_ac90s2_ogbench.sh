#!/bin/bash
# Round S2: actor-loss ablation on v3.2 no-gate (round H base) — does the loss need
# the intermediate iterates' terminal values, or only the last iteration's?
#   L = V(z_H(A^K)) + mw * mean_k V(z_H(A^k))     (H default mw = 0.1)
# Arms: mw = 0 (final-only) and mw = 0.3 (stronger anytime shaping), 2 seeds each.
# Paired references: round H seeds 0/1 (mw = 0.1, identical otherwise).
# Theory: mw>0 = dense per-iterate signal + anytime shaping; mw=0 = pure deployment
# objective (lip_select=last), avoids premature-greediness pressure on early iterates.
# v1-era h50 evidence: mw0 = 68 < champion 72; mw3 = 72 (tie). Re-test on v3.2.
# All four arms get E_final logging AND evals (focused ablation -> eval everything).
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

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90s2.log"; }

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
      --horizon 5 --iters 8 --steps 8000 --n-step 50 --arch traj --goal-mode vonly
      --cond-mode global --feat-norm --head-scale 0.01 --no-gate --amax 3.5
      --expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4
      --actor-lr 3e-4 --actor-lr-final 3e-5"

s2_train() { # arm gpu extra...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac90s2_${arm}.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" $BASE \
    --out "$out" --out-value "$MET/lip_ac90s2_${arm}_value.pt" "$@" \
    > "$LOGS/train_lip_ac90s2_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
log "round S2: actor-loss ablation — mw0/mw03 x seeds 0/1 vs H (mw=0.1) refs"
s2_train mw0_s0  0 --mean-weight 0.0 --seed 0 &
s2_train mw0_s1  1 --mean-weight 0.0 --seed 1 &
s2_train mw03_s0 2 --mean-weight 0.3 --seed 0 &
s2_train mw03_s1 3 --mean-weight 0.3 --seed 1 &
wait
log "round S2 trained"

ARMS="mw0_s0 mw0_s1 mw03_s0 mw03_s1"
for arm in $ARMS; do
  ef=$(grep -E "^step" "$LOGS/train_lip_ac90s2_${arm}.log" 2>/dev/null | tail -1 \
       | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "arm ${arm}: final E_final = ${ef:-NA}"
done
for r in 0 1; do
  ef=$(grep -E "^step" "$LOGS/train_lip_ac90h_v3h_s${r}.log" 2>/dev/null | tail -1 \
       | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "(ref) H mw01_s${r}: final E_final = ${ef:-NA}"
done

for s in 42 44; do
  gpu=0
  for arm in $ARMS; do
    p="$ACT/lip_ac90s2_${arm}.pt"
    [ -f "$p" ] || continue
    run_eval "lipac90s2_${arm}_h25_s${s}" "$gpu" "$s" 25 50 solver=lip "solver.actor_path=$p" &
    gpu=$(( (gpu + 1) % 4 ))
  done
  wait
done

for arm in $ARMS; do
  x=$(sc "lipac90s2_${arm}_h25_s42"); y=$(sc "lipac90s2_${arm}_h25_s44")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
  log "arm ${arm}: s42+s44 = $(awk "BEGIN{print $x + $y}")"
done
for r in 0 1; do
  x=$(sc "lipac90h_v3h_s${r}_h25_s42"); y=$(sc "lipac90h_v3h_s${r}_h25_s44")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
  log "(ref) H mw01_s${r}: s42+s44 = $(awk "BEGIN{print $x + $y}")"
done
log "DONE. round S2 complete — mw0/mw03 vs H's mw=0.1 on paired seeds"
