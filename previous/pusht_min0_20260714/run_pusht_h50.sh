#!/bin/bash
# h50 root-cause measurement (the cells killed by the tandem pivot).
#
# Hypothesis: PushT h50 is value-limited — aliasing forced n=1 backups, so the
# 50-step field is ~50 chained bootstraps (mush), unlike ogbench where clean
# states allowed n=50 and long-horizon LIP worked. De-aliased states (win3) or
# imagination-matched single-frame at long n should repair the field; CEM h50
# (planner-independent, no actor confound) is the clean readout vs old = 62.
# If confirmed -> wire refinement-gradient (single-frame) + restart-selection
# (win3 long-range) value split at eval; no MPPI anywhere.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
PY=python3
WM=lewm_pusht_official
H5=/workspace/swm_home/datasets/pusht_expert_train.h5
CACHE1=/workspace/caches/pusht_official_fs1.pt
EVAL_TIMEOUT=14400

mkdir -p /workspace/locks
exec 7>/workspace/locks/h50.lock
flock -n 7 || { echo "h50 driver already running"; exit 1; }

log() { echo "[$(date +%H:%M:%S)] [h50] $*" | tee -a "$LOGS/min0_driver.log"; }
run_eval() {
  local name=$1 gpu=$2 seed=$3 offset=$4 budget=$5; shift 5
  grep -q "^${name}," "$RES/summary.csv" && { log "eval ${name}: cached"; return 0; }
  CUDA_VISIBLE_DEVICES=$gpu timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" \
    --config-name pusht_lewm \
    seed="$seed" eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    solver.batch_size=10 output.filename="${name}.txt" "$@" \
    > "$LOGS/eval_${name}.log" 2>&1 7>&-
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}

# gate: wait until the fixes driver releases its lock
log "waiting for fixes driver to finish"
exec 6>/workspace/locks/fixes.lock
flock 6
log "fixes done; starting h50 measurement"

# quick trains: imagination-matched single-frame at long n (3 min each)
vi_train() {
  local tau=$1 n=$2 gpu=$3 out=$4; shift 4
  [ -f "$out" ] && return 0
  CUDA_VISIBLE_DEVICES=$gpu $PY /workspace/min0/train_metric_velimag.py \
    --cache "$CACHE1" --h5 "$H5" --wm $WM --expectile "$tau" --n-step "$n" \
    --steps 6000 --out "$out" "$@" > "$LOGS/train_$(basename "$out" .pt).log" 2>&1 7>&- \
    || log "train $(basename "$out") FAILED"
}
vi_train 0.01 5  0 "$MET/im_e0.01_n5.pt"  --no-vel &
vi_train 0.01 15 1 "$MET/im_e0.01_n15.pt" --no-vel &
vi_train 0.03 15 2 "$MET/im_e0.03_n15.pt" --no-vel &
vi_train 0.01 25 3 "$MET/im_e0.01_n25.pt" --no-vel &
wait
vi_train 0.03 5 0 "$MET/im_e0.03_n5.pt" --no-vel &
# mixed-n TD (per-pair n from a set): short n keeps the local field crisp,
# long n supervises the far field directly
vi_train 0.01 1 1 "$MET/im_e0.01_nmix1x5x15.pt" --no-vel --n-mix 1,5,15 &
vi_train 0.01 1 2 "$MET/im_e0.01_nmix1x5x25.pt" --no-vel --n-mix 1,5,25 &
wait
log "im long-n + mixed-n metrics trained"

# CEM h50 s42 readout (baseline: old teacher = 62); wn3 cells are DIAGNOSTIC
# only (is the far field repairable) — deployment stays single-frame LIPv4
IM_METRICS="im_e0.01_n1 im_e0.01_n5 im_e0.01_n15 im_e0.01_n25 im_e0.03_n5 im_e0.03_n15 im_e0.01_nmix1x5x15 im_e0.01_nmix1x5x25"
DIAG_METRICS="wn3_e0.01_n1 wn3_e0.03_n3 wn3_e0.03_n5 wn3_e0.1_n15"
i=0
for m in $IM_METRICS $DIAG_METRICS; do
  [ -f "$MET/${m}.pt" ] || { log "missing $m"; continue; }
  run_eval "cem_${m}_h50_s42" "$(( i % 4 ))" 42 50 100 "+metric=$MET/${m}.pt" &
  i=$(( i + 1 ))
done
wait
log "h50 readout (old teacher = 62):"
for m in $IM_METRICS $DIAG_METRICS; do
  log "  ${m}: $(grep "^cem_${m}_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)"
done

# best single-frame long-n critic -> ONE standard LIPv4 tandem arm if it
# beats the old teacher's far field
BEST=""; BESTV=62
for m in $IM_METRICS; do
  v=$(grep "^cem_${m}_h50_s42," "$RES/summary.csv" | tail -1 | cut -d, -f2)
  { [ -z "$v" ] || [ "$v" = "FAIL" ]; } && continue
  if awk "BEGIN{exit !($v > $BESTV)}"; then BESTV=$v; BEST=$m; fi
done
if [ -z "$BEST" ]; then
  log "no single-frame long-n critic beats old teacher h50=62; TD-side fix insufficient"
  log "H50 MEASUREMENT DONE"; exit 0
fi
TAU=$(echo "$BEST" | sed -E 's/.*_e([0-9.]+)_n.*/\1/')
if echo "$BEST" | grep -q nmix; then
  NARGS="--n-step 1 --n-mix $(echo "$BEST" | sed -E 's/.*_nmix//; s/x/,/g')"
else
  NARGS="--n-step $(echo "$BEST" | sed -E 's/.*_n([0-9]+)$/\1/')"
fi
log "best far-field critic: ${BEST} (h50 ${BESTV}) -> LIPv4 tandem arm (tau=${TAU} ${NARGS})"
OUT=/workspace/actors/tnd_h50_pi.pt
if [ ! -f "$OUT" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/pusht_official_fs5.pt --cache-td "$CACHE1" \
    --h5 "$H5" --wm $WM --out "$OUT" --out-value "$MET/tnd_h50_pi_value.pt" \
    --horizon 5 --iters 8 --steps 8000 --init-value "$MET/${BEST}.pt" \
    $NARGS --expectile 0.1 --expectile-final "$TAU" \
    --critic-lr 1e-3 --critic-lr-final 1e-4 \
    --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5 \
    --drop-z0 --drop-zg --p-imag 0.5 --no-gate --seed 0 \
    > "$LOGS/train_tnd_h50_pi.log" 2>&1 7>&- || log "h50 tandem train FAILED"
fi
run_eval "tnd_h50_pi_R8_h25_s42" 0 42 25 50 solver=lip "solver.actor_path=$OUT" \
  solver.restarts=8 solver.restart_noise=0.5 &
run_eval "tnd_h50_pi_R8_h50_s42" 1 42 50 100 solver=lip "solver.actor_path=$OUT" \
  solver.restarts=8 solver.restart_noise=0.5 &
wait
log "LIPv4-on-repaired-field: h25 $(grep '^tnd_h50_pi_R8_h25_s42,' "$RES/summary.csv" | tail -1 | cut -d, -f2) h50 $(grep '^tnd_h50_pi_R8_h50_s42,' "$RES/summary.csv" | tail -1 | cut -d, -f2) (old_pi R8 = 88/50)"
log "H50 MEASUREMENT DONE"
