#!/bin/bash
# Targeted fixes for the three s42-h25 failure mechanisms (see failure matrix):
#   (a) long boundary-hugging pushes beat local refinement (LIP-core 14,15,48)
#       -> inference diversity on the tandem winner: R16 noise 0.5 / 1.0,
#          R16 + robust_m 4 selection, and LIP-seeded MPPI (n_steps 3, 100
#          samples ~ 500 rollouts, still ~18x under CEM)
#   (b) tandem aggression overshoots precision/rotation tasks (31,32,39; all
#       tandem arms fail regardless of p-imag; sequential LIP2 at amax 2.5
#       passes) -> retrain winner arm at --amax 2.5
#   (c) CEM sampling noise misses precision tasks -> already solved by min0's
#       gradient refinement; the MPPI hybrid in (a) must keep that advantage.
# Gate: waits for the tandem driver's T4 (12 final_ rows), then postpones its
# T5 (CEM diagnostic) and takes the GPUs; T5 is re-queued at the end.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
# evals are CPU-bound (pymunk+render); uncapped torch OMP pools (258 threads/proc)
# thrash 64 cores at load 137 — cap threads and pack evals densely instead
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
WM=lewm_pusht_official
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
CACHE1=/workspace/caches/pusht_official_fs1.pt
CACHE5=/workspace/caches/pusht_official_fs5.pt
WINNER=$ACT/tnd_old_pi.pt
EVAL_TIMEOUT=14400

mkdir -p /workspace/locks
exec 8>/workspace/locks/fixes.lock
flock -n 8 || { echo "another fixes driver running; exiting"; exit 1; }
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] [fixes] $*" | tee -a "$LOGS/min0_driver.log"; }

run_eval() { # name gpu seed offset budget, then extra hydra args
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached ($(grep "^${name}," "$RES/summary.csv" | tail -1 | cut -d, -f2))"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name pusht_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1 8>&-
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then
    log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1
  fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}
sr_of() { grep "^${1}," "$RES/summary.csv" | tail -1 | cut -d, -f2; }
combined() {
  local x y; x=$(sr_of "$1"); y=$(sr_of "$2")
  { [ -z "$x" ] || [ "$x" = "FAIL" ] || [ -z "$y" ] || [ "$y" = "FAIL" ]; } && return 1
  awk "BEGIN{print $x + $y}"
}

# ---------------------------------------------------------------- gate on T4
log "waiting for tandem T4 (12 final_ rows) before taking GPUs"
until [ "$(grep -cE '^final_' "$RES/summary.csv")" -ge 12 ]; do sleep 120; done
log "T4 complete; postponing tandem T5 and starting fixes"
pkill -f "run_pusht_tandem" 2>/dev/null; sleep 2
pkill -f "eval_wm.*finalcem" 2>/dev/null; sleep 3

[ -f "$WINNER" ] || { log "FATAL: winner ckpt missing"; exit 1; }

# ---------------------------------------------------------------- FB train (GPU3)
a25_train() {
  local out="$ACT/tnd_old_pi_a25.pt" outv="$MET/tnd_old_pi_a25_value.pt"
  [ -f "$out" ] && { log "a25 train: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=3 $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm $WM \
    --out "$out" --out-value "$outv" \
    --horizon 5 --iters 8 --steps 8000 \
    --init-value "$MET/td2_e0.01_n1.pt" --n-step 1 \
    --expectile 0.1 --expectile-final 0.01 \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 2.5 \
    --drop-z0 --drop-zg --p-imag 0.5 --no-gate --seed 0 \
    > "$LOGS/train_tnd_old_pi_a25.log" 2>&1 8>&- \
    || log "a25 train FAILED"
}
log "FB: training old_pi at amax 2.5 (GPU3)"
a25_train &
A25_PID=$!

# ---------------------------------------------------------------- FA solver cfgs (GPU0-2)
fa_cfg() { # tag -> hydra args
  case "$1" in
    r16a)   echo "solver.restarts=16 solver.restart_noise=0.5" ;;
    r16b)   echo "solver.restarts=16 solver.restart_noise=1.0" ;;
    r16rob) echo "solver.restarts=16 solver.restart_noise=0.5 solver.robust_m=4" ;;
    mppi)   echo "solver.restarts=8 solver.restart_noise=0.5 solver.n_steps=3 solver.num_samples=100" ;;
  esac
}
log "FA: solver-diversity configs on the winner (dense: all 8 concurrent, GPUs 0-2)"
i=0
for tag in r16a r16b r16rob mppi; do
  for cell in "h25 25 50" "h50 50 100"; do
    set -- $cell; cname=$1; coff=$2; cbud=$3
    run_eval "fx_${tag}_${cname}_s42" "$(( i % 3 ))" 42 "$coff" "$cbud" \
      solver=lip "solver.actor_path=$WINNER" $(fa_cfg "$tag") &
    i=$(( i + 1 ))
  done
done
wait
wait $A25_PID 2>/dev/null

# FB evals (R8 baseline mode), both cells concurrent
run_eval "fx_a25_h25_s42" 0 42 25 50 solver=lip \
  "solver.actor_path=$ACT/tnd_old_pi_a25.pt" \
  solver.restarts=8 solver.restart_noise=0.5 &
run_eval "fx_a25_h50_s42" 1 42 50 100 solver=lip \
  "solver.actor_path=$ACT/tnd_old_pi_a25.pt" \
  solver.restarts=8 solver.restart_noise=0.5 &
wait

# ---------------------------------------------------------------- FC compose
log "FC ranking (baseline old_pi R8 = 138):"
BEST_TAG=""; BEST_C=""
for tag in r16a r16b r16rob mppi; do
  c=$(combined "fx_${tag}_h25_s42" "fx_${tag}_h50_s42") || continue
  log "  ${tag}: ${c}"
  if [ -z "$BEST_C" ] || awk "BEGIN{exit !($c > $BEST_C)}"; then BEST_C=$c; BEST_TAG=$tag; fi
done
A25_C=$(combined "fx_a25_h25_s42" "fx_a25_h50_s42" || echo "")
log "  a25 (R8): ${A25_C:-n/a}"

# combo: a25 actor x best solver cfg (skip if either leg missing)
if [ -n "$BEST_TAG" ] && [ -n "$A25_C" ] && [ -f "$ACT/tnd_old_pi_a25.pt" ]; then
  run_eval "fx_combo_h25_s42" 0 42 25 50 solver=lip \
    "solver.actor_path=$ACT/tnd_old_pi_a25.pt" $(fa_cfg "$BEST_TAG") &
  run_eval "fx_combo_h50_s42" 1 42 50 100 solver=lip \
    "solver.actor_path=$ACT/tnd_old_pi_a25.pt" $(fa_cfg "$BEST_TAG") &
  wait
fi
COMBO_C=$(combined "fx_combo_h25_s42" "fx_combo_h50_s42" || echo "")

# global winner among {base 138 R8, FA best, a25, combo}
GW="base"; GC=138; GCK=$WINNER; GARGS="solver.restarts=8 solver.restart_noise=0.5"
if [ -n "$BEST_C" ] && awk "BEGIN{exit !($BEST_C > $GC)}"; then
  GW=$BEST_TAG; GC=$BEST_C; GCK=$WINNER; GARGS=$(fa_cfg "$BEST_TAG"); fi
if [ -n "$A25_C" ] && awk "BEGIN{exit !($A25_C > $GC)}"; then
  GW="a25"; GC=$A25_C; GCK=$ACT/tnd_old_pi_a25.pt
  GARGS="solver.restarts=8 solver.restart_noise=0.5"; fi
if [ -n "$COMBO_C" ] && awk "BEGIN{exit !($COMBO_C > $GC)}"; then
  GW="combo"; GC=$COMBO_C; GCK=$ACT/tnd_old_pi_a25.pt; GARGS=$(fa_cfg "$BEST_TAG"); fi
echo "${GW},${GC},${GCK},${GARGS}" > "$RES/fixes_winner.txt"
log "fixes winner: ${GW} (${GC}); 3-draw if beats base"

if [ "$GW" != "base" ]; then
  i=0
  for cell in "h25 25 50" "h50 50 100"; do
    set -- $cell; cname=$1; coff=$2; cbud=$3
    for seed in 42 43 44; do
      run_eval "final_fx_${GW}_${cname}_s${seed}" "$(( i % 4 ))" "$seed" "$coff" "$cbud" \
        solver=lip "solver.actor_path=$GCK" $GARGS &
      i=$(( i + 1 ))
    done
  done
  wait
  log "fixes 3-draw done:"
  grep -E "^final_fx" "$RES/summary.csv" | while read -r l; do log "  $l"; done
fi

# ---------------------------------------------------------------- resume T5
log "re-queueing tandem driver for T5 (idempotent)"
(setsid nohup bash /workspace/min0/run_pusht_tandem_v2.sh \
  > /workspace/logs/driver_tandem_t5.log 2>&1 < /dev/null &)
log "FIXES ALL DONE"
