#!/bin/bash
# Reacher Dyna round-1 full re-card on the winning fine-tuned WM (5050).
# Runs the full 6-cell card ({h25,h50} x seeds {42,43,44}, n=50) for:
#   - each of the 3 stage-1 LIP actors, planning THROUGH the fine-tuned WM
#   - CEM on the fine-tuned WM (same-WM planner-independent baseline)
# Compare vs: stage-1 LIP on the ORIGINAL WM = 63.8 (3-seed 6-cell mean),
# CEM on original WM = 78.7. Success = LIP re-card climbs toward/past CEM.
# Usage: reacher_r1_recard.sh <arm=5050> <gpu=0>
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=18 MKL_NUM_THREADS=18

ARM=${1:-5050}
GPU=${2:-0}
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
PY=python3
WMDIR=/workspace/swm_home/checkpoints/dyna_reacher_r1_${ARM}_final
SUM=$RES/summary_reacher_r1_${ARM}.csv
DRV=$LOGS/driver_reacher_r1_recard_${ARM}.log
EVAL_TIMEOUT=7200

mkdir -p "$LOGS" "$RES"; touch "$SUM"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][recard-${ARM}] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name offset budget extra...
  local name=$1 seed=$2 offset=$3 budget=$4; shift 4
  grep -q "^${name}," "$SUM" && { log "eval ${name}: cached ($(sc "$name"))"; return 0; }
  CUDA_VISIBLE_DEVICES=$GPU timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name reacher policy="$WMDIR" \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  echo "${name},${sr:-FAIL}" >> "$SUM"
  log "eval ${name}: ${sr:-FAIL}"
}

card(){ # tag extra...
  local tag=$1; shift
  for seed in 42 43 44; do
    run_eval "card_${tag}_h25_s${seed}" "$seed" 25 50 "$@"
    run_eval "card_${tag}_h50_s${seed}" "$seed" 50 100 "$@"
  done
  # 6-cell mean
  local tot=0 v n=0
  for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "card_${tag}_${h}_s${seed}"); [ "$v" = "FAIL" ] && continue
    tot=$(awk "BEGIN{print $tot+$v}"); n=$((n+1))
  done; done
  [ "$n" -gt 0 ] && log "card ${tag} 6-cell mean: $(awk "BEGIN{printf \"%.1f\", $tot/$n}") (n=$n)"
}

for s in 0 1 2; do
  card "r1${ARM}_lip_s${s}" solver=lip "solver.actor_path=/workspace/actors/lip4_reacher_lejepa_a22_s${s}.pt"
done
card "r1${ARM}_cem" solver=cem
log "RECARD_${ARM}_DONE"
