#!/bin/bash
# Round S1: sweep around round H (v3.2 no-gate) on OPTIMIZATION axes.
# Diagnosis: the v3 family UNDERFITS (round V: E_final 3.5-7.5 vs v1's 1.46), so the
# sweep metric is E_final -- continuous, near-noiseless -- NOT n=50 eval success
# (+-5pt binomial noise). Only the top-2 arms by E_final get evals (s42+s44).
# Arms (seed 0 each, base = H config: no-gate, global-cond, feat-norm, head-scale 0.01):
#   hotlr  actor-lr 6e-4 -> 6e-5 (2x hotter)
#   12k    steps 12000 (schedules stretch automatically)
#   w384   width 384, layers 3 (capacity)
#   preln  pre-LN transformer blocks (norm_first stability variant)
# Refs: v1 champion train curve ends ~1.46; eval bar = schedamax 168 (s42+s44).
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

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/driver_ac90s1.log"; }

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
      --horizon 5 --iters 8 --n-step 50 --arch traj --goal-mode vonly
      --cond-mode global --feat-norm --head-scale 0.01 --no-gate --amax 3.5
      --expectile 0.1 --expectile-final 0.03 --critic-lr-final 1e-4 --seed 0"

s1_train() { # arm gpu extra...
  local arm=$1 gpu=$2; shift 2
  local out="$ACT/lip_ac90s1_${arm}.pt"
  [ -f "$out" ] && { log "train ${arm}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu $PY "$PLAN/train_lip_ac.py" $BASE \
    --out "$out" --out-value "$MET/lip_ac90s1_${arm}_value.pt" "$@" \
    > "$LOGS/train_lip_ac90s1_${arm}.log" 2>&1 \
    || log "train ${arm} FAILED"
}
log "round S1: sweep around H — hotlr / 12k / w384 / preln (metric: E_final)"
s1_train hotlr 0 --steps 8000  --actor-lr 6e-4 --actor-lr-final 6e-5 &
s1_train 12k   1 --steps 12000 --actor-lr 3e-4 --actor-lr-final 3e-5 &
s1_train w384  2 --steps 8000  --actor-lr 3e-4 --actor-lr-final 3e-5 --width 384 --layers 3 &
s1_train preln 3 --steps 8000  --actor-lr 3e-4 --actor-lr-final 3e-5 --pre-ln &
wait
log "round S1 trained"

ARMS="hotlr 12k w384 preln"
rank=""
for arm in $ARMS; do
  ef=$(grep -E "^step" "$LOGS/train_lip_ac90s1_${arm}.log" 2>/dev/null | tail -1 \
       | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  [ -z "${ef:-}" ] && { log "arm ${arm}: no E_final (train failed?)"; continue; }
  log "arm ${arm}: final E_final = ${ef} (v1 champion ref: 1.46; H/G refs in their logs)"
  rank="${rank}${ef} ${arm}\n"
done
TOP2=$(printf "$rank" | sort -n | head -2 | awk '{print $2}')
log "S1 top-2 by E_final: $(echo $TOP2 | tr '\n' ' ')"

gpu=0
for arm in $TOP2; do
  for s in 42 44; do
    run_eval "lipac90s1_${arm}_h25_s${s}" "$gpu" "$s" 25 50 \
      solver=lip "solver.actor_path=$ACT/lip_ac90s1_${arm}.pt" &
    gpu=$(( (gpu + 1) % 4 ))
  done
done
wait

for arm in $TOP2; do
  x=$(sc "lipac90s1_${arm}_h25_s42"); y=$(sc "lipac90s1_${arm}_h25_s44")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && continue
  log "arm ${arm}: s42+s44 = $(awk "BEGIN{print $x + $y}") (schedamax bar: 168)"
done
log "DONE. round S1 complete — compare vs H arms (lipac90h_*) and schedamax 168"
