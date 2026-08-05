#!/bin/bash
# Reacher planner baselines on a converted WM base: CEM, MPPI, gradient descent
# (AdamW), TD-value+CEM, plus env floors (random / nomove; WM-independent, run
# under the lejepa invocation only). Same eval cells as the LIP cards:
# {h25 (offset 25, budget 50), h50 (offset 50, budget 100)} x seeds {42,43,44}.
# Usage: run_reacher_baselines.sh <wm:{lejepa,pldm,dinowmnp}> <gpu>
# Idempotent via the shared per-WM summary CSV.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18

WM=$1
GPU=$2
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
PY=python3
CKPT="${WM}_reacher"
TD=$MET/td_reacher_${WM}_e0.1_n50.pt
SUM=$RES/summary_reacher_${WM}.csv
DRV=$LOGS/driver_reacher_base_${WM}.log
EVAL_TIMEOUT=10800

mkdir -p "$LOGS" "$RES"
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)][$WM] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name seed offset budget extra...
  local name=$1 seed=$2 offset=$3 budget=$4; shift 4
  if grep -q "^${name}," "$SUM"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$GPU timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name reacher \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" \
    "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED (see $LOGS/eval_${name}.log)"
    echo "${name},FAIL" >> "$SUM"; return 1
  fi
  echo "${name},${sr}" >> "$SUM"
  log "eval ${name}: ${sr}"
}

card(){ # tag extra...
  local tag=$1; shift
  for seed in 42 43 44; do
    run_eval "card_${tag}_h25_s${seed}" "$seed" 25 50 "$@"
    run_eval "card_${tag}_h50_s${seed}" "$seed" 50 100 "$@"
  done
  log "card ${tag} complete"
}

# WM-independent floors: attach to the lejepa run only
if [ "$WM" = "lejepa" ]; then
  card "floor_random" policy=random
  card "floor_nomove" policy=nomove
fi

card "${WM}_cem"  policy="$CKPT" solver=cem  solver.batch_size=10
card "${WM}_mppi" policy="$CKPT" solver=mppi solver.batch_size=10
card "${WM}_gd"   policy="$CKPT" solver=adam solver.batch_size=10
if [ -f "$TD" ]; then
  card "${WM}_tdcem" policy="$CKPT" solver=cem solver.batch_size=10 "+metric=$TD"
else
  log "skip tdcem (no TD at $TD yet)"
fi

log "=== ${WM} BASELINES FINAL"
sort "$SUM" | grep -v FAIL | tee -a "$DRV" | tail -40
touch "$RES/reacher_baselines_${WM}.DONE"
log "BASELINES ALL DONE"
