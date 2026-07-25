#!/bin/bash
# LIPv4 a20 family extension: train seeds 1,2 of the amax-2.0 arm + full cards.
# Idempotent; appends to summary_tworoom.csv and driver_tworoom_lip4.log.
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
H5=/workspace/caches/tworoom_play.h5
CACHE1=/workspace/caches/tworoom_lewm_fs1.pt
CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TD=$MET/td_e0.1_n50.pt
SUM=$RES/summary_tworoom.csv
DRV=$LOGS/driver_tworoom_lip4.log
NWORK=12
EVAL_THREADS=18
QUEUE=$RES/.evalq4x
QLOCK=$RES/.evalq4x.lock

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  grep -q "^${name}," "$SUM" && { log "eval ${name}: cached ($(sc "$name"))"; return 0; }
  local hard=(); [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$EVAL_THREADS MKL_NUM_THREADS=$EVAL_THREADS \
    timeout 7200 $PY "$PLAN/eval_wm.py" --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1
  local sr; sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}
qadd(){ echo "$1" >> "$QUEUE"; }
qpop(){ ( flock 9; head -n1 "$QUEUE" 2>/dev/null; sed -i '1d' "$QUEUE" 2>/dev/null ) 9>"$QLOCK"; }
worker(){ local wid=$1 gpu line name seed offset budget surface extra; gpu=$((wid % 4))
  while :; do line=$(qpop); [ -z "$line" ] && break
    IFS='|' read -r name seed offset budget surface extra <<<"$line"
    # shellcheck disable=SC2086
    run_eval "$name" "$gpu" "$seed" "$offset" "$budget" "$surface" $extra
  done; }
qrun(){ local i pids=(); for ((i=0;i<NWORK;i++)); do worker "$i" & pids+=($!); done; wait "${pids[@]}"; }
qcard(){ local tag=$1 actor=$2
  for surface in std hard; do for seed in 42 43 44; do
    qadd "card_${tag}_${surface}_h25_s${seed}|${seed}|25|50|${surface}|solver=lip solver.actor_path=${actor}"
    qadd "card_${tag}_${surface}_h50_s${seed}|${seed}|50|100|${surface}|solver=lip solver.actor_path=${actor}"
  done; done; }
card_sum(){ local tag=$1 total=0 v
  for surface in std hard; do for seed in 42 43 44; do for h in h25 h50; do
    v=$(sc "card_${tag}_${surface}_${h}_s${seed}")
    { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && { echo "-1"; return; }
    total=$(awk "BEGIN{print $total + $v}")
  done; done; done; echo "$total"; }

BASE_V4="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm lewm_tworoom
  --init-value $TD --horizon 5 --iters 8 --steps 8000 --n-step 50
  --expectile 0.1 --critic-lr 1e-3 --actor-lr 3e-4
  --critic-lr-final 1e-4 --expectile-final 0.03 --actor-lr-final 3e-5
  --arch v4 --amax 2.0"

train_s(){ local gpu=$1 seed=$2
  local out="$ACT/trm_v4_a20_s${seed}.pt"
  [ -f "$out" ] && { log "train v4_a20_s${seed}: cached"; return 0; }
  log "train v4_a20_s${seed}: start gpu${gpu}"
  echo "CMD: train_lip_ac.py $BASE_V4 --seed $seed" > "$LOGS/train_trm_v4_a20_s${seed}.log"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 timeout 10800 $PY "$PLAN/train_lip_ac.py" \
    $BASE_V4 --seed "$seed" \
    --out "$out" --out-value "$MET/trm_v4_a20_s${seed}_value.pt" \
    >> "$LOGS/train_trm_v4_a20_s${seed}.log" 2>&1 \
    || { log "train v4_a20_s${seed}: FAILED"; return 1; }
  local ef; ef=$(grep -E "^step" "$LOGS/train_trm_v4_a20_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train v4_a20_s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

rm -f "$QUEUE"; touch "$QUEUE"
log "=== lip4 a20 extension: seeds 1,2"
train_s 0 1 & train_s 1 2 &
wait
for s in 1 2; do
  ck="$ACT/trm_v4_a20_s${s}.pt"
  [ -f "$ck" ] && qcard "v4a20-s${s}" "$ck"
done
qrun
log "=== a20 family sums"
for s in 0 1 2; do log "card v4a20-s${s}: $(card_sum "v4a20-s${s}")"; done
perfect=""
for s in 0 1 2; do [ "$(card_sum "v4a20-s${s}")" = "1200" ] && perfect="$perfect v4a20-s${s}"; done
log "A20 PERFECT CARDS:${perfect:- none}"
touch "$RES/tworoom_lip4_a20ext.DONE"
log "A20 EXT DONE"
