#!/bin/bash
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/code/stable-worldmodel TQDM_DISABLE=1 MUJOCO_GL=osmesa
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results; ACT=/workspace/actors
SUM=$RES/summary_tworoom.csv; DRV=$LOGS/driver_tworoom_lip4.log
QUEUE=$RES/.evalqr8; QLOCK=$RES/.evalqr8.lock
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
qpop(){ ( flock 9; head -n1 "$QUEUE" 2>/dev/null; sed -i "1d" "$QUEUE" 2>/dev/null ) 9>"$QLOCK"; }
worker(){ local wid=$1 gpu line name seed offset budget surface extra; gpu=$((wid % 4))
  while :; do line=$(qpop); [ -z "$line" ] && break
    IFS="|" read -r name seed offset budget surface extra <<<"$line"
    run_eval "$name" "$gpu" "$seed" "$offset" "$budget" "$surface" $extra
  done; }
rm -f "$QUEUE"; touch "$QUEUE"
log "=== R8 cards for v4a20 s0/s1/s2"
for s in 0 1 2; do
  for surface in std hard; do for es in 42 43 44; do
    qadd "card_v4a20r8-s${s}_${surface}_h25_s${es}|${es}|25|50|${surface}|solver=lip solver.actor_path=$ACT/trm_v4_a20_s${s}.pt solver.restarts=8"
    qadd "card_v4a20r8-s${s}_${surface}_h50_s${es}|${es}|50|100|${surface}|solver=lip solver.actor_path=$ACT/trm_v4_a20_s${s}.pt solver.restarts=8"
  done; done
done
pids=(); for ((i=0;i<12;i++)); do worker "$i" & pids+=($!); done; wait "${pids[@]}"
log "=== R8 card sums"
for s in 0 1 2; do
  total=0; bad=0
  for surface in std hard; do for es in 42 43 44; do for h in h25 h50; do
    v=$(sc "card_v4a20r8-s${s}_${surface}_${h}_s${es}"); { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && bad=1
    total=$(awk "BEGIN{print $total + ${v:-0}}")
  done; done; done
  log "card v4a20r8-s${s}: $total (bad=$bad)"
done
touch "$RES/tworoom_r8.DONE"
log "R8 CARDS DONE"
