#!/bin/bash
# Pod B parameterized eval ladder (2x H100).
# Usage: run_evals_b.sh TAG WM_DIR [-a actor1,actor2,...] [-m td_metric.pt]
#                       [-s 42,43,44] [-p probe_dir]
#   TAG     names the summary csv + logs (e.g. r1_wm1)
#   WM_DIR  exported model dir (config.json = ARCH subtree + one .pt)
#   -a      comma-list of LIP actor .pt paths -> lip evals per actor x seed
#   -m      TD metric .pt -> adds TD+CEM evals per seed
#   -s      eval seeds (default 42,43,44)
#   -p      LIP_PROBE_DIR for the FIRST lip eval (A/B imagined-vs-reached dump)
# Results: /workspace/results/summary_TAG.csv (idempotent by row; re-run freely)
set -u
TAG=$1; WM=$2; shift 2
ACTORS=""; METRIC=""; SEEDS="42,43,44"; PROBE=""
while getopts "a:m:s:p:" o; do case $o in
  a) ACTORS=$OPTARG;; m) METRIC=$OPTARG;; s) SEEDS=$OPTARG;; p) PROBE=$OPTARG;;
esac; done

export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PLAN=/workspace/code/stable-worldmodel/scripts/plan
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
LOGS=/workspace/logs; RES=/workspace/results
SUM=$RES/summary_${TAG}.csv
DRV=$LOGS/driver_${TAG}.log
EVAL_TIMEOUT=14400
mkdir -p "$LOGS" "$RES" /workspace/swm_home
touch "$SUM"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

run_eval(){ # name gpu seed extra-hydra-args...
  local name=$1 gpu=$2 seed=$3
  shift 3
  local c
  c=$(sc "$name")
  if [ -n "$c" ] && [ "$c" != "FAIL" ]; then log "eval ${name}: cached (${c})"; return 0; fi
  CUDA_VISIBLE_DEVICES=$gpu timeout $EVAL_TIMEOUT python3 "$PLAN/eval_wm.py" \
    --config-name cube seed="$seed" eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    output.filename="${name}.txt" "$@" > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

# Build the job list: "name|seed|extra args" (probe attaches to first lip job)
JOBS=()
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
for s in "${SEED_ARR[@]}"; do
  JOBS+=("cem_${TAG}_e${s}|${s}|solver=cem")
done
if [ -n "$METRIC" ]; then
  for s in "${SEED_ARR[@]}"; do
    JOBS+=("cemtd_${TAG}_e${s}|${s}|solver=cem +metric=$METRIC")
  done
fi
first_lip=1
if [ -n "$ACTORS" ]; then
  IFS=',' read -ra ACT_ARR <<< "$ACTORS"
  ai=0
  for actor in "${ACT_ARR[@]}"; do
    for s in "${SEED_ARR[@]}"; do
      if [ "$first_lip" = "1" ] && [ -n "$PROBE" ]; then
        JOBS+=("lip_${TAG}_a${ai}_e${s}|${s}|PROBE solver=lip solver.actor_path=$actor")
        first_lip=0
      else
        JOBS+=("lip_${TAG}_a${ai}_e${s}|${s}|solver=lip solver.actor_path=$actor")
      fi
    done
    ai=$((ai+1))
  done
fi

log "=== ladder ${TAG}: ${#JOBS[@]} jobs on 2 GPUs (wm=$WM) ==="

worker(){ # gpu parity
  local gpu=$1 parity=$2 i=0
  local job name seed extra
  for job in "${JOBS[@]}"; do
    if [ $((i % 2)) = "$parity" ]; then
      name="${job%%|*}"
      local r="${job#*|}"; seed="${r%%|*}"; extra="${r#*|}"
      if [ "${extra:0:6}" = "PROBE " ]; then
        # shellcheck disable=SC2086
        LIP_PROBE_DIR=$PROBE run_eval "$name" "$gpu" "$seed" policy="$WM" ${extra#PROBE }
      else
        # shellcheck disable=SC2086
        run_eval "$name" "$gpu" "$seed" policy="$WM" $extra
      fi
    fi
    i=$((i+1))
  done
}
worker 0 0 &
worker 1 1 &
wait
log "LADDER_${TAG}_DONE"
