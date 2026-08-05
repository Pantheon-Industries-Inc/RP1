#!/bin/bash
# Stage 2 on the H200 pod: extract h5, convert bases, apply patches, build
# caches + the three costs, then re-establish the REFERENCE ROW at n=6.
#
# The reference row is the priority: every claim in RESULTS_window_held_20260730
# is relative to it (LIP-window pooled 44.2 / Latent+CEM-window 42.7 /
# TD+CEM-window 11.7 held @h25). Reproducing those three numbers on rebuilt
# artifacts is also the end-to-end check that the rebuild is faithful.
set -u
export STABLEWM_HOME=/workspace/swm_home PYTHONPATH=/workspace/swm_cem TQDM_DISABLE=1
export HF_HOME=/root/hf TQDM_DISABLE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
L=/workspace/logs; mkdir -p "$L"
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][stage2] $*"; }
die(){ log "FATAL: $*"; exit 1; }
CANON=/workspace/datasets_canon/lewm-reacher/reacher.h5
PLAN=/workspace/swm_cem/scripts/plan
TRM=/workspace/swm_cem/scripts/trm

# ---------------------------------------------------------------- extract h5
if [ ! -f "$CANON" ]; then
  log "extracting reacher.h5 (93G)"
  tar -I zstd --no-same-owner -xf /workspace/_hfdl/reacher.tar.zst \
    -C /workspace/datasets_canon/lewm-reacher/ > "$L/extract.log" 2>&1 || die "extract failed"
  rm -f /workspace/_hfdl/reacher.tar.zst
fi
log "h5 ok: $(du -sh $CANON | cut -f1)"
export HDF5_PLUGIN_PATH=$(python3 -c "import hdf5plugin; print(hdf5plugin.PLUGIN_PATH)")

# ---------------------------------------------------------------- bases
if [ ! -f /workspace/pretrained/reacher/lejepa_reacher_weights.ckpt ] \
   && [ ! -d /workspace/swm_home/checkpoints/lejepa_reacher ]; then
  log "extracting base tarballs"
  mkdir -p /workspace/pretrained
  for t in lewm pldm dinowm_noprop; do
    [ -f /workspace/pretrained_tars/${t}.tar.zst ] && \
      tar -I zstd --no-same-owner -xf /workspace/pretrained_tars/${t}.tar.zst -C /workspace/pretrained/ 2>&1 | tail -1
  done
fi
if [ ! -d /workspace/swm_home/checkpoints/lejepa_reacher ]; then
  log "converting bases (transformers 4.49 => NO key renames)"
  cd /workspace/swm_cem
  CUDA_VISIBLE_DEVICES=0 python3 "$PLAN/convert_reacher_bases.py" \
    --dataset "$CANON" --h5 "$CANON" > "$L/convert.log" 2>&1 \
    || log "convert returned nonzero -- inspect convert.log"
  cd /workspace
fi
ls /workspace/swm_home/checkpoints/ 2>/dev/null | head -5
[ -d /workspace/swm_home/checkpoints/lejepa_reacher ] || die "no lejepa checkpoint after convert"
log "bases ok"

# ---------------------------------------------------------------- patches
cd /workspace
for p in patch_detseed.py patch_padcontext.py patch_vframes3.py make_l2window.py patch_actpen.py; do
  [ -f "/workspace/$p" ] || { log "skip $p (absent)"; continue; }
  python3 "/workspace/$p" >> "$L/patches.log" 2>&1 && log "applied $p" || log "$p: already applied or FAILED (see patches.log)"
done

# ---------------------------------------------------------------- caches
C1=/workspace/caches/canon_lejepa_fs1.pt
C5=/workspace/caches/canon_lejepa_fs5.pt
if [ ! -f "$C1" ]; then
  log "fs1 cache (2.01M rows, ~30 min)"
  CUDA_VISIBLE_DEVICES=0 python3 "$TRM/cache_latents.py" \
    --wm /workspace/swm_home/checkpoints/lejepa_reacher --dataset "$CANON" \
    --out "$C1" --state-key qpos --batch-size 512 > "$L/cache1.log" 2>&1 || die "fs1 cache failed"
fi
if [ ! -f "$C5" ]; then
  CUDA_VISIBLE_DEVICES=0 python3 "$TRM/subsample_cache.py" --in "$C1" --out "$C5" \
    --frameskip 5 > "$L/cache5.log" 2>&1 || die "fs5 subsample failed"
fi
log "caches ok: $(du -sh $C1 | cut -f1) / $(du -sh $C5 | cut -f1)"

# ---------------------------------------------------------------- costs
TD=/workspace/metrics/td_canon_lejepa_s0.pt
W3=/workspace/metrics/window3_lejepa.pt
[ -f "$TD" ] || CUDA_VISIBLE_DEVICES=0 python3 "$PLAN/train_metric.py" --cache "$C1" \
  --learner td --head quasimetric --expectile 0.1 --n-step 50 --steps 6000 --seed 0 \
  --out "$TD" > "$L/td.log" 2>&1 || log "TD failed"
[ -f "$W3" ] || { [ -f /workspace/train_window.py ] && CUDA_VISIBLE_DEVICES=0 python3 /workspace/train_window.py \
  --cache "$C1" --lag 5 --frames 3 --expectile 0.1 --n-step 50 --steps 6000 --seed 0 \
  --out "$W3" > "$L/w3.log" 2>&1 || log "window value skipped/failed"; }
log "costs: TD $([ -f $TD ] && echo ok || echo MISSING) | window3 $([ -f $W3 ] && echo ok || echo MISSING) | l2window3 $([ -f /workspace/metrics/l2window3.pt ] && echo ok || echo MISSING)"
log "STAGE2_DONE -- next: vframes3 smoke, then LIP s0-2 on the window critic, then the reference row"
