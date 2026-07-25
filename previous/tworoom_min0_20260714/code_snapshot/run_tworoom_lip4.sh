#!/bin/bash
# TwoRoom LIPv4 campaign (2026-07-15, pod a): make the gate-free min0 (--arch v4,
# kind lip4) hit a PERFECT 12-cell card. Base recipe = the sched winner (tandem
# warm from td_e0.1_n50 + cube-champion schedules), gate removed; sweep the
# clipping (amax) + small-init (head-scale).
# Wave 1 packs 8 trainings (2/GPU): 4 arms seed0 + speculative seeds 1,2 for the
# two strongest candidates (a25, hs). Then FULL CARDS for every ckpt (evals ~1min).
# Fallback: if only a20/a30 s0 is perfect, train its seeds + card.
# Idempotent via /workspace/results/summary_tworoom.csv.
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
TRAIN_TIMEOUT=10800
EVAL_TIMEOUT=7200
NWORK=${NWORK:-12}
EVAL_THREADS=${EVAL_THREADS:-18}
QUEUE=$RES/.evalq4
QLOCK=$RES/.evalq4.lock

mkdir -p "$LOGS" "$RES" "$MET" "$ACT"
touch "$SUM"

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed offset budget surface(std|hard) extra...
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  if grep -q "^${name}," "$SUM"; then
    log "eval ${name}: cached ($(sc "$name"))"; return 0
  fi
  local hard=()
  [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$EVAL_THREADS MKL_NUM_THREADS=$EVAL_THREADS \
    timeout $EVAL_TIMEOUT $PY "$PLAN/eval_wm.py" \
    --config-name tworoom_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" \
    "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
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

qadd(){ echo "$1" >> "$QUEUE"; }
qpop(){ ( flock 9; head -n1 "$QUEUE" 2>/dev/null; sed -i '1d' "$QUEUE" 2>/dev/null ) 9>"$QLOCK"; }
worker(){
  local wid=$1 gpu line name seed offset budget surface extra
  gpu=$((wid % 4))
  while :; do
    line=$(qpop); [ -z "$line" ] && break
    IFS='|' read -r name seed offset budget surface extra <<<"$line"
    # shellcheck disable=SC2086
    run_eval "$name" "$gpu" "$seed" "$offset" "$budget" "$surface" $extra
  done
}
qrun(){
  local i pids=()
  for ((i=0; i<NWORK; i++)); do worker "$i" & pids+=($!); done
  wait "${pids[@]}"
}
qcard(){ # tag actor
  local tag=$1 actor=$2
  for surface in std hard; do
    for seed in 42 43 44; do
      qadd "card_${tag}_${surface}_h25_s${seed}|${seed}|25|50|${surface}|solver=lip solver.actor_path=${actor}"
      qadd "card_${tag}_${surface}_h50_s${seed}|${seed}|50|100|${surface}|solver=lip solver.actor_path=${actor}"
    done
  done
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

# ------------------------------------------------------------------ arms
# base = sched winner recipe with --arch v4 (min0 + no gate is implied by v4)
BASE_V4="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm lewm_tworoom
  --init-value $TD --horizon 5 --iters 8 --steps 8000 --n-step 50
  --expectile 0.1 --critic-lr 1e-3 --actor-lr 3e-4
  --critic-lr-final 1e-4 --expectile-final 0.03 --actor-lr-final 3e-5
  --arch v4"
ARMS="a20 a25 a30 hs"

flags(){ case $1 in
  a20) echo "--amax 2.0";;
  a25) echo "";;
  a30) echo "--amax 3.0";;
  hs)  echo "--head-scale 0.01";;
  *) return 1;;
esac; }

train_arm(){ # gpu arm seed
  local gpu=$1 arm=$2 seed=$3
  local out="$ACT/trm_v4_${arm}_s${seed}.pt" fl
  fl=$(flags "$arm") || { log "unknown arm $arm"; return 1; }
  [ -f "$out" ] && { log "train v4_${arm}_s${seed}: cached"; return 0; }
  log "train v4_${arm}_s${seed}: start gpu${gpu} [$fl]"
  echo "CMD: train_lip_ac.py $BASE_V4 $fl --seed $seed" > "$LOGS/train_trm_v4_${arm}_s${seed}.log"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 timeout $TRAIN_TIMEOUT $PY "$PLAN/train_lip_ac.py" \
    $BASE_V4 $fl --seed "$seed" \
    --out "$out" --out-value "$MET/trm_v4_${arm}_s${seed}_value.pt" \
    >> "$LOGS/train_trm_v4_${arm}_s${seed}.log" 2>&1 \
    || { log "train v4_${arm}_s${seed}: FAILED (see log)"; return 1; }
  local ef; ef=$(grep -E "^step" "$LOGS/train_trm_v4_${arm}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train v4_${arm}_s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

rm -f "$QUEUE"; touch "$QUEUE"

# ------------------------------------------------------------------ wave 1: 8 trainings, 2/GPU
log "=== lip4 wave 1: 8 trainings (4 arms s0 + speculative a25/hs seeds 1,2)"
train_arm 0 a20 0 & train_arm 1 a25 0 & train_arm 2 a30 0 & train_arm 3 hs 0 &
train_arm 0 a25 1 & train_arm 1 a25 2 & train_arm 2 hs 1  & train_arm 3 hs 2  &
wait
log "wave 1 trainings done"

# ------------------------------------------------------------------ cards for everything trained
log "=== lip4 cards"
for arm in $ARMS; do
  for s in 0 1 2; do
    ck="$ACT/trm_v4_${arm}_s${s}.pt"
    [ -f "$ck" ] && qcard "v4${arm}-s${s}" "$ck"
  done
done
qrun
log "cards drained"

report(){ # -> logs all sums; echoes "arm seed sum" lines
  for arm in $ARMS; do
    for s in 0 1 2; do
      [ -f "$ACT/trm_v4_${arm}_s${s}.pt" ] || continue
      echo "$arm $s $(card_sum "v4${arm}-s${s}")"
    done
  done
}
log "=== lip4 card sums"
report | while read -r arm s cs; do log "card v4${arm}-s${s}: $cs"; done

# ------------------------------------------------------------------ fallback:
# if no perfect card yet and some non-speculative arm's s0 is the best, seed it
best_line=$(report | sort -k3 -rn | head -1)
best_arm=$(echo "$best_line" | awk '{print $1}')
perfect_any=$(report | awk '$3 == 1200' | wc -l)
if [ "$perfect_any" -eq 0 ]; then
  log "no perfect card in wave 1 — fallback: seeds 1,2 for best arm $best_arm"
  train_arm 0 "$best_arm" 1 & train_arm 1 "$best_arm" 2 &
  wait
  for s in 1 2; do
    ck="$ACT/trm_v4_${best_arm}_s${s}.pt"
    [ -f "$ck" ] && qcard "v4${best_arm}-s${s}" "$ck"
  done
  qrun
  log "=== fallback card sums"
  report | while read -r arm s cs; do log "card v4${arm}-s${s}: $cs"; done
fi

log "=== LIP4 FINAL"
perfect=""
while read -r arm s cs; do
  [ "$cs" = "1200" ] && perfect="$perfect v4${arm}-s${s}"
done < <(report)
log "PERFECT CARDS:${perfect:- none}"
touch "$RES/tworoom_lip4.DONE"
log "LIP4 ALL DONE"
