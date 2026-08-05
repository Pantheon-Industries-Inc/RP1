#!/bin/bash
# Reacher DINO (dinowmnp) stage-1: convert -> capped cache -> TD -> LIP.
# DINO flat-token latents are 75264-d (196 patches x 384): full 2M rows would
# be ~566GB, and LIP training OOMs at K8/B128 through the explicit-attention
# predictor, so phase1 caps to 200k rows and LIP runs K4 B16 (validated).
# GPU0. Data (reacher_random.lance + train h5) already exist from stage-0.
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TQDM_DISABLE=1 MUJOCO_GL=osmesa OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
PLAN=/workspace/stable-worldmodel/scripts/plan
LOGS=/workspace/logs
DRV=$LOGS/driver_dino_stage1.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][dino-s1] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

if [ ! -f /workspace/pretrained/reacher/dinowm_noprop_weights.ckpt ]; then
  log "extract tarball"
  tar -I zstd --no-same-owner --no-same-permissions -xf /workspace/dinowm_noprop.tar.zst -C /workspace/pretrained/ || die "extract failed"
fi

if [ ! -d /workspace/swm_home/checkpoints/dinowmnp_reacher ]; then
  log "convert dinowmnp"
  CUDA_VISIBLE_DEVICES=0 python3 "$PLAN/convert_reacher_bases.py" --only dinowmnp \
    --dataset dmc/reacher_random.lance --h5 /workspace/caches/reacher_random_train.h5 \
    > "$LOGS/convert_dino.log" 2>&1 || die "convert failed (see convert_dino.log)"
  grep -E "backbone vs|ratio|wrote" "$LOGS/convert_dino.log" | tee -a "$DRV"
fi
log "convert ok"

log "phase1 (capped cache + TD)"
bash "$PLAN/reacher_phase1.sh" dinowmnp 0 || die "phase1 failed"

log "LIP amax 2.2 x3 seeds (K4 B16)"
AMAX_ARMS=2.2 bash "$PLAN/run_reacher_lip4.sh" dinowmnp 0 4 16 || log "LIP driver returned nonzero"

log "baselines (CEM/MPPI/GD)"
bash "$PLAN/run_reacher_baselines.sh" dinowmnp 0 || log "baselines returned nonzero"
log "DINO_STAGE1_DONE"
