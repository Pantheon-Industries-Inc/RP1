#!/bin/bash
# DINO on-policy collection: 2 stage-1 LIP actors (s0,s1; s2 was not trained),
# k4-suffixed names. More calls per actor to reach ~1500 episodes. Reuses the
# eval-path SWM_RECORD_PATH recorder. GPU0.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1 MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${1:-0}
CALLS=${2:-22}
CODE=/workspace/stable-worldmodel
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
OUT=/workspace/dyna_data
DRV=$LOGS/driver_reacher_collect_r1_dinowmnp.log
PY=python3
mkdir -p "$OUT" "$LOGS"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dino-r1] $*" | tee -a "$DRV"; }

collect_call(){ # actor call_idx
  local a=$1 i=$2
  local seed=$((2000 + a * 100 + i))
  local rec="$OUT/reacher_onpolicy_r1_dinowmnp_a${a}.lance"
  local tag="rcol_dino_a${a}_c${i}"
  grep -q "kept=" "$LOGS/collect_${tag}.log" 2>/dev/null && { log "call a${a}/c${i}: cached"; return 0; }
  SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=$GPU timeout 7200 $PY \
    "$PLAN/eval_wm.py" --config-name reacher seed=$seed \
    eval.goal_offset_steps=25 eval.eval_budget=50 \
    policy=dinowmnp_reacher solver=lip \
    "solver.actor_path=/workspace/actors/lip4_reacher_dinowmnp_a22k4_s${a}.pt" \
    output.filename="${tag}.txt" > "$LOGS/collect_${tag}.log" 2>&1
  local kept; kept=$(grep -oE "kept=[0-9]+" "$LOGS/collect_${tag}.log" | tail -1)
  log "call a${a}/c${i} (seed $seed): rc=$? ${kept:-no-record}"
}

log "=== DINO r1 collection start (2 actors x ${CALLS} calls x 50 envs)"
for i in $(seq 0 $((CALLS - 1))); do
  for a in 0 1; do collect_call "$a" "$i"; done
done
for a in 0 1; do
  $PY - "$OUT/reacher_onpolicy_r1_dinowmnp_a${a}.lance" << 'PYEOF' >> "$DRV" 2>&1
import sys, lance, numpy as np
ds = lance.dataset(sys.argv[1])
epi = np.asarray(ds.take(list(range(ds.count_rows())), columns=["episode_idx"]).to_pydict()["episode_idx"])
print(f"{sys.argv[1]}: rows={ds.count_rows()} episodes={len(set(epi.tolist()))}")
PYEOF
done
log "REACHER_COLLECT_R1_DONE"
