#!/bin/bash
# Scores round: (a) latent+CEM full card (10 missing cells), (b) random-policy
# card (12 cells, the floor), (c) restarts=8 probes on v4a20-s2's two blemish
# cells. Idempotent; appends to summary_tworoom.csv, logs to driver_tworoom_lip4.log.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
ACT=/workspace/actors
PY=python3
SUM=$RES/summary_tworoom.csv
DRV=$LOGS/driver_tworoom_lip4.log
NWORK=12
EVAL_THREADS=18
QUEUE=$RES/.evalqsc
QLOCK=$RES/.evalqsc.lock

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

rm -f "$QUEUE"; touch "$QUEUE"
log "=== scores round: latent card + random card + v4a20-s2 R8 probes"

for surface in std hard; do
  for seed in 42 43 44; do
    qadd "latent_${surface}_h25_s${seed}|${seed}|25|50|${surface}|"
    qadd "latent_${surface}_h50_s${seed}|${seed}|50|100|${surface}|"
    qadd "random_${surface}_h25_s${seed}|${seed}|25|50|${surface}|policy=random"
    qadd "random_${surface}_h50_s${seed}|${seed}|50|100|${surface}|policy=random"
  done
done
qadd "r8probe_v4a20s2_std_h50_s44|44|50|100|std|solver=lip solver.actor_path=$ACT/trm_v4_a20_s2.pt solver.restarts=8"
qadd "r8probe_v4a20s2_hard_h50_s42|42|50|100|hard|solver=lip solver.actor_path=$ACT/trm_v4_a20_s2.pt solver.restarts=8"
qrun

log "=== scores round results"
for surface in std hard; do
  for seed in 42 43 44; do
    for h in h25 h50; do
      log "latent_${surface}_${h}_s${seed}: $(sc latent_${surface}_${h}_s${seed}) | random: $(sc random_${surface}_${h}_s${seed})"
    done
  done
done
log "R8 probes: std_h50_s44 $(sc r8probe_v4a20s2_std_h50_s44) | hard_h50_s42 $(sc r8probe_v4a20s2_hard_h50_s42)"
touch "$RES/tworoom_scores.DONE"
log "SCORES ROUND DONE"
