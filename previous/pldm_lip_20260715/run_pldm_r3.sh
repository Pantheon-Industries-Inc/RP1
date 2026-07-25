#!/bin/bash
# Round 3: settle flat35 (152, 1 seed) vs k12 (146, 2 seeds) + test flat35xK12 composition.
# Also: confirm cells for current leader k12 (s43 h25 + h50 x3, both seeds) run first (cheap).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=osmesa
export TQDM_DISABLE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
MET=/workspace/metrics
ACT=/workspace/actors
PY=python3
WM_PLDM=/workspace/ckpts/PLDM_OgBench_lewm
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
CACHE1=/workspace/caches/cube_pldm_fs1.pt
CACHE5=/workspace/caches/cube_pldm_fs5.pt
EVAL_TIMEOUT=14400
touch "$RES/summary.csv"

log() { echo "[$(date +%H:%M:%S)] PLDM-R3: $*" | tee -a "$LOGS/pipeline_pldm.log"; }

while pgrep -f "run_pldm_r2.sh" > /dev/null; do sleep 30; done

GPUQ=/tmp/gpuq.r3.$$
mkfifo "$GPUQ"; exec 9<>"$GPUQ"; rm -f "$GPUQ"
for g in 0 1 2 3; do echo "$g" >&9; done
run_pool() {
  local pids=()
  for job in "$@"; do
    local name="${job%%|*}" cmd="${job#*|}"
    read -r -u 9 GPU
    (
      export CUDA_VISIBLE_DEVICES=$GPU
      log "start [$name] on gpu$GPU"
      eval "$cmd" > "$LOGS/${name}.log" 2>&1
      rc=$?
      [ $rc -eq 0 ] && log "done  [$name]" || log "FAIL  [$name] rc=$rc"
      echo "$GPU" >&9
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}

do_eval() {
  local name=$1 seed=$2 offset=$3 budget=$4; shift 4
  if grep -q "^${name}," "$RES/summary.csv"; then
    log "eval ${name}: cached"; return 0
  fi
  timeout "$EVAL_TIMEOUT" $PY "$PLAN/eval_wm.py" --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps="$offset" eval.eval_budget="$budget" \
    output.filename="${name}.txt" "$@" > "$LOGS/ev_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/ev_${name}.log" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$")
  if [ -z "${sr:-}" ]; then echo "${name},FAIL" >> "$RES/summary.csv"; return 1; fi
  echo "${name},${sr}" >> "$RES/summary.csv"
  log "eval ${name}: ${sr}"
}

train_arm() { # name extra-args...   (all warm from t003n50)
  local name=$1; shift
  [ -f "$ACT/pldm_${name}.pt" ] && { log "train ${name}: cached"; return 0; }
  $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM_PLDM" \
    --init-value "$MET/cf_pldm_t003n50.pt" \
    --horizon 5 --iters 8 --steps 8000 --n-step 50 --batch 128 \
    --drop-z0 --drop-zg --no-gate \
    --out "$ACT/pldm_${name}.pt" --out-value "$MET/pldm_${name}_value.pt" "$@"
}

# ---- phase A: confirm cells for current leader k12 (both seeds), cheap ----
log "phase A: k12 confirm cells"
CONF=()
for s in s0 s1; do
  CONF+=("cf_k12_${s}_h25_s43|do_eval lip_k12_${s}_h25_s43 43 25 50 policy=$WM_PLDM solver=lip solver.actor_path=$ACT/pldm_k12_${s}.pt")
  for d in 42 43 44; do
    CONF+=("cf_k12_${s}_h50_s${d}|do_eval lip_k12_${s}_h50_s${d} $d 50 100 policy=$WM_PLDM solver=lip solver.actor_path=$ACT/pldm_k12_${s}.pt")
  done
done
run_pool "${CONF[@]}"

# ---- phase B: round-3 trainings ----
FLAT35="--expectile 0.03 --critic-lr 1e-3 --actor-lr 3e-4 --amax 3.5"
SMX="--expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5"
log "phase B: round-3 training (6 arms)"
run_pool \
  "tr_flat35_s1|train_arm flat35_s1 $FLAT35 --seed 1" \
  "tr_flat35_s2|train_arm flat35_s2 $FLAT35 --seed 2" \
  "tr_flat35k12_s0|train_arm flat35k12_s0 $FLAT35 --iters 12 --seed 0" \
  "tr_k12_s2|train_arm k12_s2 $SMX --iters 12 --seed 2" \
  "tr_flat35k12_s1|train_arm flat35k12_s1 $FLAT35 --iters 12 --seed 1" \
  "tr_flat35_s3|train_arm flat35_s3 $FLAT35 --seed 3"
log "round-3 training complete"

# ---- phase C: selection evals for new arms ----
EVJOBS=()
for arm in flat35_s1 flat35_s2 flat35_s3 flat35k12_s0 flat35k12_s1 k12_s2; do
  [ -f "$ACT/pldm_${arm}.pt" ] || { log "skip evals for missing arm $arm"; continue; }
  for seed in 42 44; do
    EVJOBS+=("ev_lip_${arm}_s${seed}|do_eval lip_${arm}_h25_s${seed} $seed 25 50 policy=$WM_PLDM solver=lip solver.actor_path=$ACT/pldm_${arm}.pt")
  done
done
run_pool "${EVJOBS[@]}"

log "ROUND 3 COMPLETE. summary:"
sort "$RES/summary.csv" | tee -a "$LOGS/pipeline_pldm.log"
