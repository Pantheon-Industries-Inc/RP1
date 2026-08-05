#!/bin/bash
# Dyna: collect ON-POLICY data with the winning LIP actors.
#
# The shipped reacher_collect_r1.sh cannot be used as-is: it hardcodes the old
# actor paths (lip4_reacher_*_a22_s*) and sets MUJOCO_GL=osmesa, which this
# campaign banned (osmesa renders differ from the dataset's by 2.58/255 and put
# the lejepa encoder out of domain). This is the same idea with the current
# artifacts and EGL.
#
# WHY DYNA SHOULD HELP HELD SPECIFICALLY: the value cannot see settling (AUC
# 0.505-0.63) partly because the random-play dataset barely contains it -- only
# 0.4% of at-goal states are still at goal 25 steps later. A goal-reaching policy
# generates exactly the behaviour that is missing, so fine-tuning the WM on it
# should improve the imagined dynamics near the goal, which is where held-at-end
# is decided.
#
# SPLIT DISCIPLINE: collection draws tasks from episodes 0:8000 ONLY. The eval
# pool 8000:10000 is never touched, on either the collection or the expert side.
set -u
GPU_LIST=${1:-"0 1 2"}
CALLS=${2:-6}
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")
PLAN=/workspace/swm_cem/scripts/plan
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
WM=/workspace/swm_home/checkpoints/lejepa_reacher
OUT=/workspace/dyna_data; mkdir -p "$OUT"
L=/workspace/logs/dynacol; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dynacol] $*"; }

# the 48.9 recipe's three actors
ACTORS="l1e3ua18m01_s0 l1e3ua18m01_s1 l1e3ua18m01_s2"

i=0
for tag in $ACTORS; do
  A=/workspace/actors/lip4_g1k_${tag}.pt
  [ -f "$A" ] || { log "MISSING $A"; continue; }
  gpu=$(echo $GPU_LIST | cut -d" " -f$(( (i % 3) + 1 )))
  (
    for c in $(seq 0 $((CALLS - 1))); do
      rec="$OUT/onp_${tag}_c${c}.lance"
      [ -d "$rec" ] && { log "  ${tag} c${c}: exists"; continue; }
      seed=$(( 900 + i * 100 + c ))
      SWM_RECORD_PATH=$rec MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
      CUDA_VISIBLE_DEVICES=$gpu timeout 7200 python3 "$PLAN/eval_wm.py" \
        --config-name reacher policy="$WM" \
        eval.dataset_name="$CANON" dataset.stats="$CANON" \
        +eval.ep_range=0:8000 seed=$seed \
        eval.goal_offset_steps=25 eval.eval_budget=50 \
        solver=lip solver.actor_path="$A" solver.rollout_compat=false \
        solver.batch_size=10 output.filename=col_${tag}_c${c}.txt \
        > "$L/col_${tag}_c${c}.log" 2>&1
      kept=$(grep -oE "kept=[0-9]+" "$L/col_${tag}_c${c}.log" | tail -1)
      log "  ${tag} c${c} (seed ${seed}): ${kept:-no-record-line}"
    done
  ) &
  i=$((i + 1))
done
wait

log "collected lances:"
tot=0
for d in "$OUT"/onp_*.lance; do
  [ -d "$d" ] || continue
  r=$(python3 -c "
import lance,sys
try: print(lance.dataset(sys.argv[1]).count_rows())
except Exception: print(0)
" "$d")
  tot=$((tot + r))
  log "  $(basename $d): ${r} rows"
done
log "TOTAL on-policy rows: ${tot}"
log "DYNACOL_DONE"
