#!/bin/bash
# One canonical LIPv4 cell on the full-token DINO-WM, to measure per-cell cost
# before sizing the amax x lr grid.
#
# Recipe is the canonical cube one (from dyna_repair_downstream.sh), warm-started
# from the TD sweep winner. Two things differ from the v2WM runs by necessity:
#   * latents are 75,264-d instead of 192-d, so the fs5 cache that train_lip_ac
#     pushes onto the GPU is 98.8 GB of the 143 GB card -- watch for OOM;
#   * no --action-stats-pin: the rebuilt expert_actions.h5 IS the expert set and
#     its stats match the hardcoded pin to ~0.5%, so its own stats are correct.
#
# Runs on GPU 1: LIP training does no rendering, so it safely co-resides with the
# EGL eval on GPU 0 (renders must never overlap each other, training may).
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export TQDM_DISABLE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CODE=/workspace/code/stable-worldmodel; P=$CODE/scripts/plan
L=/workspace/logs
mkdir -p /workspace/actors "$L"; cd "$CODE"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_lip_timing.log"; }

AMAX=1.6; SEED=0
OUT=/workspace/actors/lip4_dinofull_a${AMAX}_s${SEED}.pt
log "=== LIPv4 timing cell: amax $AMAX, actor-lr 3e-4, warm-start e0.1_lr3e-4 ==="
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=1 timeout 43200 python3 "$P/train_lip_ac.py" \
  --cache /workspace/caches/dinofull_tr8000_fs5.pt \
  --cache-td /workspace/caches/dinofull_tr8000.pt \
  --h5 /workspace/datasets/expert_actions.h5 \
  --wm /workspace/ckpts/dinowm_noprop_cube \
  --init-value /workspace/metrics/dinofull_e0.1_lr0.0003.pt \
  --horizon 5 --iters 8 --steps 6000 --n-step 50 --amax "$AMAX" \
  --expectile 0.1 --expectile-final 0.03 \
  --critic-lr 1e-3 --critic-lr-final 1e-4 \
  --actor-lr 3e-4 --actor-lr-final 3e-5 \
  --arch v4 --seed "$SEED" \
  --out "$OUT" --out-value "${OUT%.pt}_value.pt" \
  > "$L/lip_timing.log" 2>&1
rc=$?
t1=$(date +%s)
log "  exit $rc after $(( (t1-t0)/60 )) min"
if [ -f "$OUT" ]; then
  log "  actor written: $(ls -la "$OUT" | awk '{print $5}') bytes"
else
  log "  NO ACTOR -- last lines:"
  tail -12 "$L/lip_timing.log" | tee -a "$L/driver_lip_timing.log"
fi
log "LIP_TIMING_DONE"
