#!/bin/bash
# TwoRoom pretrained-bases campaign, phase 1: per-WM latent caches + TD warm-starts.
# One WM per GPU; each chain is idempotent. Usage: tworoom_bases_phase1.sh <wm> <gpu>
# wm in {lejepa,pldm,dinowm,dinowmnp} -> checkpoint "<wm>_tworoom".
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

WM=$1
GPU=$2
CODE=/workspace/code/stable-worldmodel
TRM=$CODE/scripts/trm
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
CACHES=/workspace/caches
MET=/workspace/metrics
PY=python3
CKPT="${WM}_tworoom"
CACHE1=$CACHES/tworoom_${WM}_fs1.pt
CACHE5=$CACHES/tworoom_${WM}_fs5.pt
TD=$MET/td_${WM}_e0.1_n50.pt

mkdir -p "$LOGS" "$CACHES" "$MET"
log() { echo "[$(date +%H:%M:%S)][$WM] $*" | tee -a "$LOGS/bases_phase1_${WM}.log"; }
die() { log "FATAL: $*"; exit 1; }

if [ ! -f "$CACHE1" ]; then
  log "fs1 cache"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$TRM/cache_latents.py" --wm "$CKPT" \
    --dataset tworoom_play.lance --out "$CACHE1" --state-key state --batch-size 256 \
    > "$LOGS/cache_${WM}_fs1.log" 2>&1 || die "fs1 cache failed"
fi
log "fs1 ok"

if [ ! -f "$CACHE5" ]; then
  log "fs5 subsample"
  $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" --frameskip 5 \
    > "$LOGS/cache_${WM}_fs5.log" 2>&1 || die "fs5 subsample failed"
fi
log "fs5 ok"

if [ ! -f "$TD" ]; then
  log "TD warm-start (tau 0.1, n-step 50, 6k steps)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$PLAN/train_metric.py" --cache "$CACHE1" \
    --learner td --head quasimetric --expectile 0.1 --n-step 50 \
    --steps 6000 --out "$TD" > "$LOGS/td_${WM}.log" 2>&1 || die "TD warm-start failed"
fi
log "TD ok"
log "PHASE1 DONE"
