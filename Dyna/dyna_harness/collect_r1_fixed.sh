#!/bin/bash
# ############################################################################
# # DATA SPLIT WARNING -- see ../DATA_SPLIT_POLICY.md
# # Dyna on-policy collection MUST come from episodes disjoint from the ones
# # eval draws its tasks from. It currently does NOT: collection and eval both
# # draw from the same 10k-episode expert lance, so the fine-tuned WM has seen
# # the eval states. Required split: collect 0-7999, eval 8000-9999
# # (episode_split.COLLECT / .EVAL). Numbers from this script are upper bounds.
# ############################################################################
# Dyna on-policy collection -- FIXED parallelism.
#
# Replaces collect_r1.sh. Four defects addressed (see PARALLELIZATION_ANALYSIS.md
# sections 4.1b and 5.2):
#
#   1. MUJOCO_EGL_DEVICE_ID was never set, so CUDA_VISIBLE_DEVICES spread CUDA
#      across GPUs while ALL EGL render contexts piled onto physical GPU 0.
#      (Verified: mujoco/egl/__init__.py:create_initialized_egl_device_display
#      iterates eglQueryDevicesEXT() in a fixed order and takes the first device
#      that initializes; CUDA_VISIBLE_DEVICES does not filter that list --
#      measured n_egl_devices=5 regardless of its value.)  -> now set per worker.
#
#   2. `PAR=3 -> PAR=1` was LATCHED on the first failure and never reset, so one
#      abort in batch 0 made the remaining 7 batches sequential.
#      -> replaced with bounded per-job retry; no global degradation.
#
#   3. Both the retry path and the sequential path hardcoded gpu 0, stranding
#      GPUs 1-3.  -> every job runs on its worker's own GPU.
#
#   4. `wait $P0; wait $P1; wait $P2` was a lockstep barrier per batch: batch i+1
#      could not start until the slowest actor finished batch i.
#      -> one independent worker per actor, no barrier. Total time is now
#      max-over-actors instead of sum-over-batches-of-max.
#
# The one-actor-per-worker shape is REQUIRED, not incidental: concurrent writers
# must not share a lance, and each actor writes its own onpolicy_r1_a${a}.lance.
# To use more than 3 workers you must shard per call and merge (see SHARD note).
#
# Usage: ./collect_r1_fixed.sh [n_calls_per_actor] [max_retries]
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TQDM_DISABLE=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
CODE=/workspace/code/stable-worldmodel
LOGS=/workspace/logs
DRV=$LOGS/driver_collect_r1.log
WM=/workspace/models/v2WM
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
OUT=/workspace/dyna_data
NCALLS=${1:-8}
MAXTRY=${2:-3}
mkdir -p "$OUT" "$LOGS"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }

collect_call(){ # actor gpu call_idx
  local a=$1 gpu=$2 i=$3
  local seed=$((1000 + a * 100 + i))
  local rec="$OUT/onpolicy_r1_a${a}.lance"
  local tag="col_a${a}_c${i}"
  # FIX 1: MUJOCO_EGL_DEVICE_ID pins the EGL context to the same physical GPU
  # that CUDA_VISIBLE_DEVICES pins compute to. Without it every worker renders
  # on GPU 0 no matter what CUDA_VISIBLE_DEVICES says.
  SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
    timeout 7200 python3 \
    "$CODE/scripts/plan/eval_wm.py" --config-name cube seed=$seed \
    eval.dataset_name="$EXPERT" ++bf16=true eval.img_size=224 \
    eval.goal_offset_steps=25 eval.eval_budget=50 eval.num_eval=50 \
    policy="$WM" solver=lip "solver.actor_path=/workspace/actors/lip4_v2_s${a}.pt" \
    output.filename="${tag}.txt" > "$LOGS/collect_${tag}.log" 2>&1
  local rc=$?
  local kept
  kept=$(grep -oE "kept=[0-9]+" "$LOGS/collect_${tag}.log" | tail -1)
  log "call a${a}/c${i} (seed $seed, gpu$gpu): rc=$rc ${kept:-no-record-line}"
  return $rc
}

done_already(){ # actor call_idx  -- uniform idempotency guard (the original only
                # applied this on the retry/sequential paths, so a restart redid
                # all parallel work)
  grep -q "kept=" "$LOGS/collect_col_a${1}_c${2}.log" 2>/dev/null
}

# One worker per actor. Its GPU is fixed for the whole chain, so its lance has a
# single writer and its EGL contexts have a single device.
worker(){ # actor gpu
  local a=$1 gpu=$2 fails=0
  for i in $(seq 0 $((NCALLS - 1))); do
    if done_already "$a" "$i"; then
      log "call a${a}/c${i}: cached"
      continue
    fi
    local try=1
    until collect_call "$a" "$gpu" "$i"; do
      # FIX 2/3: bounded per-JOB retry on this worker's OWN gpu. No global latch,
      # no fallback to gpu 0.
      if [ "$try" -ge "$MAXTRY" ]; then
        log "call a${a}/c${i}: GIVING UP after ${try} attempts"
        fails=$((fails + 1))
        break
      fi
      try=$((try + 1))
      log "call a${a}/c${i}: retry ${try}/${MAXTRY} on gpu${gpu}"
      sleep 10
    done
  done
  log "worker a${a} (gpu${gpu}) finished, ${fails} unrecoverable call(s)"
  return $fails
}

cd "$CODE"
log "=== collection start: 3 actors x ${NCALLS} calls x 50 envs, 1 worker/actor, no barrier ==="
T0=$SECONDS
# FIX 4: no per-batch barrier -- three independent chains.
worker 0 0 & W0=$!
worker 1 1 & W1=$!
worker 2 2 & W2=$!
FAILED=0
wait $W0 || FAILED=$((FAILED + $?))
wait $W1 || FAILED=$((FAILED + $?))
wait $W2 || FAILED=$((FAILED + $?))
log "all workers done in $((SECONDS - T0))s; ${FAILED} unrecoverable call(s) total"

# SHARD note: GPU 3 is idle here because the 3 lances have 3 writers. To use all
# four, write per-call lances (onpolicy_r1_a${a}_c${i}.lance) into a 24-job work
# queue over 4 workers and concat afterwards -- build_dyna_mix.py's stream_source
# already reads a list of lances, so the merge is nearly free.

for a in 0 1 2; do
  python3 - "$OUT/onpolicy_r1_a${a}.lance" << 'PYEOF' >> "$DRV" 2>&1
import sys, lance, numpy as np
ds = lance.dataset(sys.argv[1])
epi = np.asarray(ds.take(list(range(ds.count_rows())), columns=["episode_idx"]).to_pydict()["episode_idx"])
print(f"{sys.argv[1]}: rows={ds.count_rows()} episodes={len(set(epi.tolist()))}")
PYEOF
done
[ "$FAILED" = 0 ] && log "COLLECT_R1_DONE" || log "COLLECT_R1_INCOMPLETE (${FAILED} calls failed)"
