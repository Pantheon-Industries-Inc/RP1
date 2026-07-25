#!/bin/bash
# PLDM h75 program: baselines + TD diagnostics + round-1 h75-native tandem sweep.
# h75-native lessons from LeWM (writeup §8.4): value tau 0.03->0.1 (+6), goal window 100->150 primitives (+6).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home MUJOCO_GL=osmesa TQDM_DISABLE=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PLAN=/workspace/code/stable-worldmodel/scripts/plan
LOGS=/workspace/logs; RES=/workspace/results; MET=/workspace/metrics; ACT=/workspace/actors
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
WM=/workspace/ckpts/PLDM_OgBench_lewm
log() { echo "[$(date +%H:%M:%S)] PLDM-H75: $*" | tee -a "$LOGS/pipeline_pldm.log"; }

GPUQ=/tmp/gpuq.h75.$$
mkfifo "$GPUQ"; exec 9<>"$GPUQ"; rm -f "$GPUQ"
for g in 0 1 2 3; do echo "$g" >&9; done
run_pool() {
  local pids=()
  for job in "$@"; do
    local name="${job%%|*}" cmd="${job#*|}"
    read -r -u 9 GPU
    ( export CUDA_VISIBLE_DEVICES=$GPU
      log "start [$name] on gpu$GPU"
      eval "$cmd" > "$LOGS/${name}.log" 2>&1
      rc=$?; [ $rc -eq 0 ] && log "done  [$name]" || log "FAIL  [$name] rc=$rc"
      echo "$GPU" >&9 ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}
do_eval() { # name seed extra...
  local name=$1 seed=$2; shift 2
  grep -q "^${name}," "$RES/summary.csv" && { log "eval ${name}: cached"; return 0; }
  timeout 14400 python3 "$PLAN/eval_wm.py" --config-name cube \
    seed="$seed" eval.dataset_name="$H5" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=75 eval.eval_budget=150 \
    output.filename="${name}.txt" "$@" > "$LOGS/ev_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/ev_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { echo "${name},FAIL" >> "$RES/summary.csv"; return 1; }
  echo "${name},${sr}" >> "$RES/summary.csv"; log "eval ${name}: ${sr}"
}
train_arm() { # name extra...
  local name=$1; shift
  [ -f "$ACT/pldm_${name}.pt" ] && { log "train ${name}: cached"; return 0; }
  python3 "$PLAN/train_lip_ac.py" \
    --cache /workspace/caches/cube_pldm_fs5.pt --cache-td /workspace/caches/cube_pldm_fs1.pt \
    --h5 "$H5" --wm "$WM" \
    --horizon 5 --iters 12 --steps 8000 --n-step 50 --batch 128 --amax 3.5 \
    --drop-z0 --drop-zg --no-gate \
    --out "$ACT/pldm_${name}.pt" --out-value "$MET/pldm_${name}_value.pt" "$@"
}

log "phase A: h75 baselines + diagnostics"
run_pool \
  "ev_pldmcem_h75_s42|do_eval pldmcem_h75_s42 42 policy=$WM solver=cem" \
  "ev_pldmcem_h75_s43|do_eval pldmcem_h75_s43 43 policy=$WM solver=cem" \
  "ev_pldmcem_h75_s44|do_eval pldmcem_h75_s44 44 policy=$WM solver=cem" \
  "ev_tdcem_t003_h75_s42|do_eval tdcem_pldm_t003_h75_s42 42 policy=$WM solver=cem +metric=$MET/cf_pldm_t003n50.pt"
run_pool \
  "ev_tdcem_t01_h75_s42|do_eval tdcem_pldm_t01_h75_s42 42 policy=$WM solver=cem +metric=$MET/cf_pldm_t01n50.pt" \
  "ev_lip_k12_s0_h75_s42|do_eval lip_k12_s0_h75_s42 42 policy=$WM solver=lip solver.actor_path=$ACT/pldm_k12_s0.pt" \
  "ev_lip_k12_s0_h75_s44|do_eval lip_k12_s0_h75_s44 44 policy=$WM solver=lip solver.actor_path=$ACT/pldm_k12_s0.pt" \
  "ev_lip_k12_s1_h75_s42|do_eval lip_k12_s1_h75_s42 42 policy=$WM solver=lip solver.actor_path=$ACT/pldm_k12_s1.pt"

log "phase B: round-1 h75-native trainings (4 arms)"
SCHED="--expectile 0.1 --expectile-final 0.03 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5"
T01="--expectile 0.1 --critic-lr 1e-3 --critic-lr-final 1e-4 --actor-lr 3e-4 --actor-lr-final 3e-5"
run_pool \
  "tr_h75md20_s0|train_arm h75md20_s0 $SCHED --max-delta 20 --init-value $MET/cf_pldm_t003n50.pt --seed 0" \
  "tr_h75md30_s0|train_arm h75md30_s0 $SCHED --max-delta 30 --init-value $MET/cf_pldm_t003n50.pt --seed 0" \
  "tr_h75md20t01_s0|train_arm h75md20t01_s0 $T01 --max-delta 20 --init-value $MET/cf_pldm_t01n50.pt --seed 0" \
  "tr_h75md30t01_s0|train_arm h75md30t01_s0 $T01 --max-delta 30 --init-value $MET/cf_pldm_t01n50.pt --seed 0"

log "phase C: round-1 selection (s42+s44 h75)"
EV=()
for arm in h75md20_s0 h75md30_s0 h75md20t01_s0 h75md30t01_s0; do
  [ -f "$ACT/pldm_${arm}.pt" ] || { log "skip $arm"; continue; }
  for s in 42 44; do
    EV+=("ev_lip_${arm}_s${s}|do_eval lip_${arm}_h75_s${s} $s policy=$WM solver=lip solver.actor_path=$ACT/pldm_${arm}.pt")
  done
done
run_pool "${EV[@]}"
log "H75 ROUND 1 COMPLETE"
sort "$RES/summary.csv" | grep h75 | tee -a "$LOGS/pipeline_pldm.log"
