#!/bin/bash
# Reacher LIP campaign, phase 1: h5 + per-WM latent caches + TD warm-starts.
# One WM per GPU; each chain is idempotent. Usage: reacher_phase1.sh <wm> <gpu>
# wm in {lejepa,pldm,dinowmnp} -> checkpoint "<wm>_reacher".
# Expects the repo at /workspace/stable-worldmodel (symlink to /workspace/code/
# stable-worldmodel on Dyna-lineage pods).
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/stable-worldmodel
export HF_HOME=/root/hf
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

WM=$1
GPU=$2
CODE=/workspace/stable-worldmodel
TRM=$CODE/scripts/trm
PLAN=$CODE/scripts/plan
LOGS=/workspace/logs
CACHES=/workspace/caches
MET=/workspace/metrics
PY=python3
CKPT="${WM}_reacher"
LANCE=dmc/reacher_random.lance
H5EVAL=/workspace/swm_home/datasets/dmc/reacher_random.h5
H5TRAIN=/workspace/caches/reacher_random_train.h5
CACHE1=$CACHES/reacher_${WM}_fs1.pt
# dinowmnp flat-token latents are 75264-d; full 2M rows = ~566GB. Cap to the
# first 1000 episodes (200k rows, ~56GB) — h5 ep_offset is id-indexed so the
# episode-id subset stays consistent for train_lip_ac.
MAXROWS_ARGS=""
[ "$WM" = "dinowmnp" ] && MAXROWS_ARGS="--max-rows 200000"
CACHE5=$CACHES/reacher_${WM}_fs5.pt
TD=$MET/td_reacher_${WM}_e0.1_n50.pt

mkdir -p "$LOGS" "$CACHES" "$MET"
log() { echo "[$(date +%H:%M:%S)][$WM] $*" | tee -a "$LOGS/reacher_phase1_${WM}.log"; }
die() { log "FATAL: $*"; exit 1; }

# h5 build: single-writer guard via flock; all WM chains need it
(
  flock 9
  if [ ! -f "$H5EVAL" ] || [ ! -f "$H5TRAIN" ]; then
    log "building train+eval h5 from lance"
    $PY "$CODE/scripts/data/build_reacher_h5.py" --dataset "$LANCE" \
      --train-out "$H5TRAIN" --eval-out "$H5EVAL" --eval-episodes 1024 \
      > "$LOGS/reacher_h5.log" 2>&1 || die "h5 build failed"
    log "h5 built"
  fi
) 9>/workspace/caches/.h5.lock
[ -f "$H5EVAL" ] || die "h5 missing after build"

if [ ! -f "$CACHE1" ]; then
  log "fs1 cache"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$TRM/cache_latents.py" --wm "$CKPT" \
    --dataset "$LANCE" --out "$CACHE1" --batch-size 256 $MAXROWS_ARGS \
    > "$LOGS/cache_reacher_${WM}_fs1.log" 2>&1 || die "fs1 cache failed"
fi
log "fs1 ok"

if [ ! -f "$CACHE5" ]; then
  log "fs5 subsample"
  $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" --frameskip 5 \
    > "$LOGS/cache_reacher_${WM}_fs5.log" 2>&1 || die "fs5 subsample failed"
fi
log "fs5 ok"

if [ ! -f "$TD" ]; then
  log "TD warm-start (tau 0.1, n-step 50, 6k steps)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$PLAN/train_metric.py" --cache "$CACHE1" \
    --learner td --head quasimetric --expectile 0.1 --n-step 50 \
    --steps 6000 --out "$TD" > "$LOGS/td_reacher_${WM}.log" 2>&1 \
    || die "TD warm-start failed"
fi
log "TD ok"
log "PHASE1 DONE"
