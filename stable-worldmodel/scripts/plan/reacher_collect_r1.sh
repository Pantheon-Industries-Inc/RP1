#!/bin/bash
# Reacher Dyna round-1 on-policy collection: roll the stage-1 LIP actors in
# the real sim via the eval path (goals set from the eval h5) and record
# (pixels, action, qpos, qvel) per step through SWM_RECORD_PATH — the exact
# mechanism validated on OGBench. 3 actors x N calls x 50 envs, one lance per
# actor (concurrent writers must not share a lance). Seeds 2000+ disjoint
# from eval (42/43/44). Sequential on one GPU (parallel 50-env evals segfault).
# Usage: reacher_collect_r1.sh <wm> <gpu> [calls_per_actor=14]
#   (14 calls x ~40 kept x 3 actors ≈ 1,700 episodes ≈ OGBench r1 scale)
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

WM=$1
GPU=$2
CALLS=${3:-14}
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
OUT=/workspace/dyna_data
DRV=$LOGS/driver_reacher_collect_r1_${WM}.log
PY=python3

mkdir -p "$OUT" "$LOGS"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][${WM}-r1] $*" | tee -a "$DRV"; }

collect_call(){ # actor_seed call_idx
  local a=$1 i=$2
  local seed=$((2000 + a * 100 + i))
  local rec="$OUT/reacher_onpolicy_r1_${WM}_a${a}.lance"
  local tag="rcol_${WM}_a${a}_c${i}"
  grep -q "kept=" "$LOGS/collect_${tag}.log" 2>/dev/null && { log "call a${a}/c${i}: cached"; return 0; }
  SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=$GPU timeout 7200 $PY \
    "$PLAN/eval_wm.py" --config-name reacher seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    +eval.ep_range=${COLLECT_EP_RANGE:-0:8000} \
    policy="${WM}_reacher" solver=lip \
    "solver.actor_path=/workspace/actors/lip4_reacher_${WM}_a22_s${a}.pt" \
    output.filename="${tag}.txt" > "$LOGS/collect_${tag}.log" 2>&1
  local rc=$?
  local kept
  kept=$(grep -oE "kept=[0-9]+" "$LOGS/collect_${tag}.log" | tail -1)
  log "call a${a}/c${i} (seed $seed): rc=$rc ${kept:-no-record-line}"
  return $rc
}

log "=== reacher r1 collection start (${WM}, 3 actors x ${CALLS} calls x 50 envs)"
for i in $(seq 0 $((CALLS - 1))); do
  for a in 0 1 2; do
    collect_call "$a" "$i" || log "call a${a}/c${i} nonzero (will retry on rerun)"
  done
done

for a in 0 1 2; do
  $PY - "$OUT/reacher_onpolicy_r1_${WM}_a${a}.lance" << 'PYEOF' >> "$DRV" 2>&1
import sys, lance, numpy as np
ds = lance.dataset(sys.argv[1])
epi = np.asarray(ds.take(list(range(ds.count_rows())), columns=["episode_idx"]).to_pydict()["episode_idx"])
print(f"{sys.argv[1]}: rows={ds.count_rows()} episodes={len(set(epi.tolist()))}")
PYEOF
done
log "REACHER_COLLECT_R1_DONE"
