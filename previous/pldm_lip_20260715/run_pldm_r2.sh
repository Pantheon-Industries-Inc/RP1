#!/bin/bash
# Round 2: compose round-1 winners + seed replicas. Waits for pipeline_pldm.sh to finish.
# R1 standings (s42+s44 h25): flat 144, k12 142, amax25 138, smx_s1 138 | smx_s0 128 (seed spread 10)
# Arms: flat seed replicas, flat+K12 composition (x3 seeds), k12 replica (sched vs flat disambig),
#       flat+amax3.5 (completes amax x schedule square), flat warm-started from t01n50.
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

log() { echo "[$(date +%H:%M:%S)] PLDM-R2: $*" | tee -a "$LOGS/pipeline_pldm.log"; }

# wait for round-1 pipeline to exit (frees GPUs)
while pgrep -f "pipeline_pldm.sh" > /dev/null; do sleep 60; done
log "round 1 pipeline finished; starting round 2"

GPUQ=/tmp/gpuq.r2.$$
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

train_arm() { # name init-value extra-args...
  local name=$1 iv=$2; shift 2
  [ -f "$ACT/pldm_${name}.pt" ] && { log "train ${name}: cached"; return 0; }
  $PY "$PLAN/train_lip_ac.py" \
    --cache "$CACHE5" --cache-td "$CACHE1" --h5 "$H5" --wm "$WM_PLDM" \
    --init-value "$iv" \
    --horizon 5 --iters 8 --steps 8000 --n-step 50 --batch 128 \
    --drop-z0 --drop-zg --no-gate \
    --out "$ACT/pldm_${name}.pt" --out-value "$MET/pldm_${name}_value.pt" "$@"
}

FLAT="--expectile 0.03 --critic-lr 1e-3 --actor-lr 3e-4 --amax 2.5"
SMX="--expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5 --amax 3.5"
TD003=$MET/cf_pldm_t003n50.pt
TD01=$MET/cf_pldm_t01n50.pt

log "round-2 training (8 arms)"
run_pool \
  "tr_flat_s1|train_arm flat_s1 $TD003 $FLAT --seed 1" \
  "tr_flat_s2|train_arm flat_s2 $TD003 $FLAT --seed 2" \
  "tr_flatk12_s0|train_arm flatk12_s0 $TD003 $FLAT --iters 12 --seed 0" \
  "tr_flatk12_s1|train_arm flatk12_s1 $TD003 $FLAT --iters 12 --seed 1" \
  "tr_flatk12_s2|train_arm flatk12_s2 $TD003 $FLAT --iters 12 --seed 2" \
  "tr_k12_s1|train_arm k12_s1 $TD003 $SMX --iters 12 --seed 1" \
  "tr_flat35_s0|train_arm flat35_s0 $TD003 --expectile 0.03 --critic-lr 1e-3 --actor-lr 3e-4 --amax 3.5 --seed 0" \
  "tr_flatt01_s0|train_arm flatt01_s0 $TD01 $FLAT --seed 0"
log "round-2 training complete"

EVJOBS=()
for arm in flat_s1 flat_s2 flatk12_s0 flatk12_s1 flatk12_s2 k12_s1 flat35_s0 flatt01_s0; do
  [ -f "$ACT/pldm_${arm}.pt" ] || { log "skip evals for missing arm $arm"; continue; }
  for seed in 42 44; do
    EVJOBS+=("ev_lip_${arm}_s${seed}|do_eval lip_${arm}_h25_s${seed} $seed 25 50 policy=$WM_PLDM solver=lip solver.actor_path=$ACT/pldm_${arm}.pt")
  done
done
run_pool "${EVJOBS[@]}"

log "ROUND 2 COMPLETE. summary:"
sort "$RES/summary.csv" | tee -a "$LOGS/pipeline_pldm.log"
