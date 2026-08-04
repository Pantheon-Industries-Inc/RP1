#!/bin/bash
# Reacher LIPv4 sweep: canonical triple-perfect recipe with an env-specific
# amax sweep (amax substitutes for the gate in v4 and is env-specific:
# tworoom 2.2, cube 3.5). Arms {1.7, 2.2, 2.8} at seed 0 -> card each ->
# winner by card sum -> seeds 1,2 -> cards.
# Usage: run_reacher_lip4.sh <wm:{lejepa,pldm,dinowmnp}> <gpu> [iters] [batch]
#   iters/batch: DINO token WMs OOM at K8/B128 — run e.g. `dinowmnp 0 4 16`.
# Idempotent: trained actors and CSV-cached eval rows are skipped on rerun.
# Result on the first campaign pod (2026-07-23): amax 2.2 won on BOTH
# lejepa (63.7) and pldm (85.0).
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
ITERS=${3:-8}
BATCH=${4:-128}
SUFF=""
[ "$ITERS" != "8" ] && SUFF="k${ITERS}"
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
CKPT="${WM}_reacher"
H5=/workspace/caches/reacher_random_train.h5
CACHE1=/workspace/caches/reacher_${WM}_fs1.pt
CACHE5=/workspace/caches/reacher_${WM}_fs5.pt
TD=$MET/td_reacher_${WM}_e0.1_n50.pt
SUM=$RES/summary_reacher_${WM}.csv
DRV=$LOGS/driver_reacher_lip4_${WM}.log
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=7200
AMAX_ARMS=${AMAX_ARMS:-"1.7 2.2 2.8"}   # override to "2.2" to skip the sweep (2.2 won on both WMs)

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)][$WM] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name seed offset budget extra...
  local name=$1 seed=$2 offset=$3 budget=$4; shift 4
  if grep -q "^${name}," "$SUM"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$GPU timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name reacher policy="$CKPT" \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
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

card(){ # tag actor
  local tag=$1 actor=$2
  for seed in 42 43 44; do
    run_eval "card_${tag}_h25_s${seed}" "$seed" 25 50 \
      solver=lip "solver.actor_path=$actor"
    run_eval "card_${tag}_h50_s${seed}" "$seed" 50 100 \
      solver=lip "solver.actor_path=$actor"
  done
  log "card ${tag} complete"
}

card_sum(){ # tag -> sum over 6 cells, or -1 if incomplete
  local tag=$1 total=0 v
  for seed in 42 43 44; do
    for h in h25 h50; do
      v=$(sc "card_${tag}_${h}_s${seed}")
      { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { echo "-1"; return; }
      total=$(awk "BEGIN{print $total + $v}")
    done
  done
  echo "$total"
}

train_arm(){ # amax seed
  local amax=$1 seed=$2
  local suff="a${amax/./}${SUFF}"
  local out="$ACT/lip4_reacher_${WM}_${suff}_s${seed}.pt"
  [ -f "$out" ] && { log "train ${suff} s${seed}: cached"; return 0; }
  [ -f "$TD" ] || die "missing TD warm-start $TD (run phase 1)"
  log "train ${suff} s${seed}: start (v4 amax${amax} md12 + canonical schedules)"
  CUDA_VISIBLE_DEVICES=$GPU timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$CKPT" \
    --init-value "$TD" \
    --arch v4 --amax "$amax" --max-delta 12 --iters "$ITERS" --batch "$BATCH" \
    --horizon 5 --steps 8000 \
    --n-step 50 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --seed "$seed" \
    --out "$out" --out-value "$MET/lip4_reacher_${WM}_${suff}_s${seed}_value.pt" \
    > "$LOGS/train_lip4_${WM}_${suff}_s${seed}.log" 2>&1 \
    || die "train ${suff} s${seed} failed"
  local ef; ef=$(grep -E "^step" "$LOGS/train_lip4_${WM}_${suff}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train ${suff} s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

# ---------------------------------------------------- wave 1: amax arms, seed 0
for amax in $AMAX_ARMS; do
  suff="a${amax/./}${SUFF}"
  train_arm "$amax" 0
  card "${WM}_${suff}-s0" "$ACT/lip4_reacher_${WM}_${suff}_s0.pt"
  log "arm ${suff} s0 card sum: $(card_sum "${WM}_${suff}-s0") / 600"
done

# ---------------------------------------------------- winner -> seeds 1,2
best_amax=""; best_sum=-1
for amax in $AMAX_ARMS; do
  suff="a${amax/./}${SUFF}"
  cs=$(card_sum "${WM}_${suff}-s0")
  log "arm ${suff}: $cs"
  if awk "BEGIN{exit !($cs > $best_sum)}"; then best_sum=$cs; best_amax=$amax; fi
done
[ -n "$best_amax" ] || die "no completed arm"
log "WINNER arm: amax $best_amax (card sum $best_sum)"

suff="a${best_amax/./}${SUFF}"
for seed in 1 2; do
  train_arm "$best_amax" "$seed"
  card "${WM}_${suff}-s${seed}" "$ACT/lip4_reacher_${WM}_${suff}_s${seed}.pt"
  log "winner s${seed} card sum: $(card_sum "${WM}_${suff}-s${seed}") / 600"
done

log "=== ${WM} LIP4 FINAL"
sort "$SUM" | grep -v FAIL | tee -a "$DRV" | tail -40
touch "$RES/reacher_lip4_${WM}.DONE"
log "LIP4 ALL DONE"
