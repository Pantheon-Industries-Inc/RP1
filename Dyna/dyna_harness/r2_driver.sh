#!/bin/bash
# Dyna round-2 driver (loop pod). Phases strictly serialized so eval-path work
# never overlaps training (known SIGABRT/SIGTERM combo):
#  P1 validation draws s45/s46 for the 3 r1w actors on WM1 (selection protocol)
#  P2 round-2 collection: r1w actors on WM1, seeds 2000+, 60 calls/actor
#  P3 (training, parallel): r2 fine-tune (WM1 + expert(+)r1(+)r2 @50/50, 1 epoch,
#     gate both) on gpu0  ||  r1 fine-tune SEED REPLICA (same r1 data, seed 3073)
#     on gpu1
#  P4 gates: r2 arm (divergence vs 31.8/36.1 + canary) + replica quick-gate
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
RES=/workspace/results
PY=python3
WM1=/workspace/models/dyna_r1_5050b
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
SUM=$RES/summary_r1w.csv
DRV=$LOGS/driver_r2.log
mkdir -p "$LOGS" "$RES" /workspace/dyna_data

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
sc(){ grep -h "^${1}," "$SUM" 2>/dev/null | tail -1 | cut -d, -f2; }

run_eval(){ # name seed extra...
  local name=$1 seed=$2
  shift 2
  local c
  c=$(sc "$name")
  if [ -n "$c" ] && [ "$c" != "FAIL" ]; then log "eval ${name}: cached (${c})"; return 0; fi
  CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY "$PLAN/eval_wm.py" \
    --config-name cube seed="$seed" eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
    output.filename="${name}.txt" "$@" > "$LOGS/eval_${name}.log" 2>&1
  local sr
  sr=$(grep -oE "success_rate[^0-9]*[0-9.]+" "$LOGS/eval_${name}.log" | tail -1 | grep -oE "[0-9.]+$")
  [ -z "${sr:-}" ] && { log "eval ${name}: FAILED"; echo "${name},FAIL" >> "$SUM"; return 1; }
  echo "${name},${sr}" >> "$SUM"; log "eval ${name}: ${sr}"
}

cd "$CODE"
log "=== r2 driver start (pid $$) ==="

# ---------------- P1: validation draws (selection protocol), sequential
for a in 0 1 2; do
  for s in 45 46; do
    run_eval "lip_r1w_a${a}_v${s}" "$s" policy="$WM1" solver=lip \
      "solver.actor_path=/workspace/actors/lip4_r1w_s${a}.pt"
  done
done
log "P1 validation draws done"

# ---------------- P2: round-2 collection (sequential, quiet pod)
collect_call(){ # actor call_idx
  local a=$1 i=$2
  local seed=$((2000 + a * 100 + i))
  local rec="/workspace/dyna_data/onpolicy_r2_a${a}.lance"
  SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=0 timeout 7200 $PY "$PLAN/eval_wm.py" \
    --config-name cube seed=$seed eval.dataset_name="$EXPERT" ++bf16=true \
    eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
    policy="$WM1" solver=lip "solver.actor_path=/workspace/actors/lip4_r1w_s${a}.pt" \
    output.filename="r2col_a${a}_c${i}.txt" > "$LOGS/r2col_a${a}_c${i}.log" 2>&1
  local kept
  kept=$(grep -oE "kept=[0-9]+" "$LOGS/r2col_a${a}_c${i}.log" | tail -1)
  log "r2 collect a${a}/c${i}: rc=$? ${kept:-no-record}"
}
if [ ! -f /workspace/dyna_data/R2_COLLECT_DONE ]; then
  for i in $(seq 0 59); do
    for a in 0 1 2; do
      grep -q "kept=" "$LOGS/r2col_a${a}_c${i}.log" 2>/dev/null || collect_call "$a" "$i"
    done
  done
  for a in 0 1 2; do
    $PY - "/workspace/dyna_data/onpolicy_r2_a${a}.lance" << 'PYEOF' >> "$DRV" 2>&1
import sys, lance, numpy as np
ds = lance.dataset(sys.argv[1])
epi = np.asarray(ds.take(list(range(ds.count_rows())), columns=["episode_idx"]).to_pydict()["episode_idx"])
print(f"{sys.argv[1]}: rows={ds.count_rows()} episodes={len(set(epi.tolist()))}")
PYEOF
  done
  touch /workspace/dyna_data/R2_COLLECT_DONE
fi
log "P2 collection done"
log "R2_READY_FOR_TRAINING — hand back to orchestrator for P3 (dataset build + fine-tunes)"
