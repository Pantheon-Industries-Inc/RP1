#!/bin/bash
# Dyna round-1 on-policy collection (pod A, must be otherwise QUIET).
# 3 actors x 8 calls x 50 envs (= up to 400 eps/actor before the <25-step drop).
# Each call: eval-path rollout with SWM_RECORD_PATH -> per-actor lance
# (concurrent writers must not share one lance). Parallelism: 3 -> auto-degrade
# to 1 on abort.
#
# ############################################################################
# # DATA SPLIT WARNING -- see ../DATA_SPLIT_POLICY.md
# #
# # Collection seeds are 1000+ while eval uses 42/43/44. That is SEED
# # disjointness, which guarantees NOTHING: both draw start states from the
# # SAME 10k-episode expert lance, so this collection covers eval episodes and
# # the WM fine-tuned on it has seen the states it is later scored on.
# #
# # Required: restrict collection to episodes 0-7999 and eval to 8000-9999
# # (episode_split.COLLECT / .EVAL). eval_wm.py has no episode-range filter
# # yet, so this script CANNOT yet honour the split -- results from it are
# # upper bounds, not generalization estimates. Measure the leak with:
# #   python episode_split.py overlap --collect <lances> --eval-dataset <lance>
# ############################################################################
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
mkdir -p "$OUT" "$LOGS"

log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }

collect_call(){ # actor_seed gpu call_idx
  local a=$1 gpu=$2 i=$3
  local seed=$((1000 + a * 100 + i))
  local rec="$OUT/onpolicy_r1_a${a}.lance"
  local tag="col_a${a}_c${i}"
  SWM_RECORD_PATH=$rec CUDA_VISIBLE_DEVICES=$gpu timeout 7200 python3 \
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

cd "$CODE"
PAR=3
log "=== collection r1 start (3 actors x 8 calls x 50 envs, PAR=$PAR) ==="
for i in $(seq 0 7); do
  if [ "$PAR" = 3 ]; then
    collect_call 0 0 "$i" &
    P0=$!
    collect_call 1 1 "$i" &
    P1=$!
    collect_call 2 2 "$i" &
    P2=$!
    R=0
    wait $P0 || R=1
    wait $P1 || R=1
    wait $P2 || R=1
    if [ "$R" != 0 ]; then
      log "parallel batch $i had a failure -> degrading to sequential + retrying"
      PAR=1
      for a in 0 1 2; do
        grep -q "kept=" "$LOGS/collect_col_a${a}_c${i}.log" 2>/dev/null || collect_call "$a" 0 "$i"
      done
    fi
  else
    for a in 0 1 2; do
      grep -q "kept=" "$LOGS/collect_col_a${a}_c${i}.log" 2>/dev/null || collect_call "$a" 0 "$i"
    done
  fi
done

for a in 0 1 2; do
  python3 - "$OUT/onpolicy_r1_a${a}.lance" << 'PYEOF' >> "$DRV" 2>&1
import sys, lance, numpy as np
ds = lance.dataset(sys.argv[1])
epi = np.asarray(ds.take(list(range(ds.count_rows())), columns=["episode_idx"]).to_pydict()["episode_idx"])
print(f"{sys.argv[1]}: rows={ds.count_rows()} episodes={len(set(epi.tolist()))}")
PYEOF
done
log "COLLECT_R1_DONE"
