#!/bin/bash
# TwoRoom pretrained-bases campaign, phase 2: LIPv4 (canonical triple-perfect
# recipe) per base + full 12-cell cards. One WM per GPU, seeds sequential:
#   train s0 -> card s0 -> train s1 -> card s1 -> train s2 -> card s2
# Usage: run_bases_lip4.sh <wm:{lejepa,pldm,dinowm,dinowmnp}> <gpu>
# Idempotent: trained actors and CSV-cached eval rows are skipped on rerun.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18

WM=$1
GPU=$2
ITERS=${3:-8}   # LIP refinement iterations K (DINO runs capped K per directive)
SUFF=""
[ "$ITERS" != "8" ] && SUFF="k${ITERS}"
CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
CKPT="${WM}_tworoom"
H5=/workspace/caches/tworoom_play.h5
CACHE1=/workspace/caches/tworoom_${WM}_fs1.pt
CACHE5=/workspace/caches/tworoom_${WM}_fs5.pt
TD=$MET/td_${WM}_e0.1_n50.pt
SUM=$RES/summary_bases_${WM}${SUFF}.csv
DRV=$LOGS/driver_bases_${WM}${SUFF}.log
TRAIN_TIMEOUT=28800
EVAL_TIMEOUT=5400

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)][$WM] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name seed offset budget surface(std|hard) actor
  local name=$1 seed=$2 offset=$3 budget=$4 surface=$5 actor=$6
  if grep -q "^${name}," "$SUM"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$GPU timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name tworoom_lewm policy="$CKPT" \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" \
    solver=lip "solver.actor_path=$actor" \
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
  for surface in std hard; do
    for seed in 42 43 44; do
      run_eval "card_${tag}_${surface}_h25_s${seed}" "$seed" 25 50 "$surface" "$actor"
      run_eval "card_${tag}_${surface}_h50_s${seed}" "$seed" 50 100 "$surface" "$actor"
    done
  done
  log "card ${tag} complete"
}

card_sum(){ # tag
  local tag=$1 total=0 v
  for surface in std hard; do
    for seed in 42 43 44; do
      for h in h25 h50; do
        v=$(sc "card_${tag}_${surface}_${h}_s${seed}")
        { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { echo "-1"; return; }
        total=$(awk "BEGIN{print $total + $v}")
      done
    done
  done
  echo "$total"
}

train_seed(){ # seed
  local seed=$1
  local out="$ACT/trm_${WM}_v4${SUFF}_s${seed}.pt"
  [ -f "$out" ] && { log "train s${seed}: cached"; return 0; }
  [ -f "$TD" ] || die "missing TD warm-start $TD (run phase 1)"
  log "train s${seed}: start (canonical v4 amax2.2 md12 + schedules)"
  CUDA_VISIBLE_DEVICES=$GPU timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$CKPT" \
    --init-value "$TD" \
    --arch v4 --amax 2.2 --max-delta 12 --iters "$ITERS" --horizon 5 --steps 8000 \
    --n-step 50 --expectile 0.1 --expectile-final 0.03 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 \
    --seed "$seed" \
    --out "$out" --out-value "$MET/trm_${WM}_v4${SUFF}_s${seed}_value.pt" \
    > "$LOGS/train_${WM}_v4${SUFF}_s${seed}.log" 2>&1 \
    || die "train s${seed} failed (see $LOGS/train_${WM}_v4${SUFF}_s${seed}.log)"
  local ef; ef=$(grep -E "^step" "$LOGS/train_${WM}_v4${SUFF}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

for seed in 0 1 2; do
  train_seed "$seed"
  card "${WM}v4${SUFF}-s${seed}" "$ACT/trm_${WM}_v4${SUFF}_s${seed}.pt"
  log "seed ${seed} card sum: $(card_sum "${WM}v4${SUFF}-s${seed}") / 1200"
done

log "=== ${WM} FINAL"
sort "$SUM" | tee -a "$DRV"
touch "$RES/bases_${WM}${SUFF}.DONE"
log "ALL DONE"
