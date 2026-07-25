#!/bin/bash
# LIPv4 round C: plain-deploy triple-perfect hunt, wall-trap-targeted.
# Base = ax22 (amax 2.2, the round-B family best 1200/1198/1200). Variants:
#   md12 = --max-delta 12 (HER goals to 60 steps -> more cross-door supervision)
#   pc05 = --p-cross 0.5  (more cross-episode goals -> diverse routing pairs)
#   k12  = --iters 12     (more refinement per replan -> escape wall pinning)
# each x seeds 0/1/2; plus plain ax22 seeds 3/4/5 (family evidence).
# NO restarts anywhere. Idempotent via summary_tworoom.csv.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results; MET=/workspace/metrics; ACT=/workspace/actors
H5=/workspace/caches/tworoom_play.h5
CACHE1=/workspace/caches/tworoom_lewm_fs1.pt
CACHE5=/workspace/caches/tworoom_lewm_fs5.pt
TD=$MET/td_e0.1_n50.pt
SUM=$RES/summary_tworoom.csv; DRV=$LOGS/driver_tworoom_lip4.log
QUEUE=$RES/.evalq4c; QLOCK=$RES/.evalq4c.lock

log(){ echo "[$(date +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep "^${1}," "$SUM" | tail -1 | cut -d, -f2; }

run_eval(){ local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5 surface=$6; shift 6
  grep -q "^${name}," "$SUM" && { log "eval ${name}: cached ($(sc "$name"))"; return 0; }
  local hard=(); [ "$surface" = "hard" ] && hard=("+eval.cross_wall=true")
  mkdir -p "$RES/videos_${name}"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=18 MKL_NUM_THREADS=18 timeout 7200 python3 "$PLAN/eval_wm.py" \
    --config-name tworoom_lewm seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "+video_dir=$RES/videos_${name}" "${hard[@]}" "$@" \
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
qrun(){ local i pids=(); for ((i=0;i<12;i++)); do worker "$i" & pids+=($!); done; wait "${pids[@]}"; }

BASE="--cache $CACHE5 --cache-td $CACHE1 --h5 $H5 --wm lewm_tworoom
  --init-value $TD --horizon 5 --steps 8000 --n-step 50
  --expectile 0.1 --critic-lr 1e-3 --actor-lr 3e-4
  --critic-lr-final 1e-4 --expectile-final 0.03 --actor-lr-final 3e-5
  --arch v4 --amax 2.2"

flags(){ case $1 in
  md12) echo "--iters 8 --max-delta 12";;
  pc05) echo "--iters 8 --p-cross 0.5";;
  k12)  echo "--iters 12";;
  ax22) echo "--iters 8";;
  *) return 1;;
esac; }

train_arm(){ local gpu=$1 arm=$2 seed=$3
  local out="$ACT/trm_v4c_${arm}_s${seed}.pt" fl; fl=$(flags "$arm") || { log "unknown arm $arm"; return 1; }
  [ -f "$out" ] && { log "train v4c_${arm}_s${seed}: cached"; return 0; }
  log "train v4c_${arm}_s${seed}: start gpu${gpu} [$fl]"
  echo "CMD: train_lip_ac.py $BASE $fl --seed $seed" > "$LOGS/train_trm_v4c_${arm}_s${seed}.log"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 timeout 10800 python3 "$PLAN/train_lip_ac.py" \
    $BASE $fl --seed "$seed" --out "$out" --out-value "$MET/trm_v4c_${arm}_s${seed}_value.pt" \
    >> "$LOGS/train_trm_v4c_${arm}_s${seed}.log" 2>&1 \
    || { log "train v4c_${arm}_s${seed}: FAILED"; return 1; }
  local ef; ef=$(grep -E "^step" "$LOGS/train_trm_v4c_${arm}_s${seed}.log" | tail -1 \
    | grep -oE "E_final [0-9.]+" | awk '{print $2}')
  log "train v4c_${arm}_s${seed}: done, E_final ${ef:-NA} (indicator only)"
}

rm -f "$QUEUE"; touch "$QUEUE"
log "=== lip4 round C: md12/pc05/k12 x s0-2 + ax22 x s3-5 (plain deploy only)"
i=0
for arm in md12 pc05 k12; do
  for s in 0 1 2; do
    train_arm $((i % 4)) "$arm" "$s" &
    i=$((i+1))
  done
done
for s in 3 4 5; do
  train_arm $((i % 4)) "ax22" "$s" &
  i=$((i+1))
done
wait
log "round C trainings done"

for arm in md12 pc05 k12 ax22; do
  for s in 0 1 2 3 4 5; do
    ck="$ACT/trm_v4c_${arm}_s${s}.pt"
    [ -f "$ck" ] || continue
    for surface in std hard; do for es in 42 43 44; do
      qadd "card_v4c${arm}-s${s}_${surface}_h25_s${es}|${es}|25|50|${surface}|solver=lip solver.actor_path=${ck}"
      qadd "card_v4c${arm}-s${s}_${surface}_h50_s${es}|${es}|50|100|${surface}|solver=lip solver.actor_path=${ck}"
    done; done
  done
done
qrun
log "=== round C card sums"
for arm in md12 pc05 k12 ax22; do
  fam=""
  for s in 0 1 2 3 4 5; do
    [ -f "$ACT/trm_v4c_${arm}_s${s}.pt" ] || continue
    total=0
    for surface in std hard; do for es in 42 43 44; do for h in h25 h50; do
      v=$(sc "card_v4c${arm}-s${s}_${surface}_${h}_s${es}"); total=$(awk "BEGIN{print $total + ${v:-0}}")
    done; done; done
    fam="$fam $total"
    log "card v4c${arm}-s${s}: $total"
  done
  log "family v4c${arm}:$fam"
done
touch "$RES/tworoom_lip4c.DONE"
log "LIP4C DONE"
