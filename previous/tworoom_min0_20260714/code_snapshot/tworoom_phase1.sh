#!/bin/bash
# TwoRoom min0 campaign, phase 1: data + caches on pod a (4xH100).
# Mirrors tworoom_lip_20260710/run_all.sh phase 0+1 exactly (seed 7 -> identical dataset).
set -u
export STABLEWM_HOME=/workspace/swm_home
export PYTHONPATH=/workspace/code/stable-worldmodel
export TQDM_DISABLE=1
export MUJOCO_GL=osmesa

CODE=/workspace/code/stable-worldmodel
TRM=$CODE/scripts/trm
LOGS=/workspace/logs
CACHES=/workspace/caches
PY=python3
PLAY=tworoom_play.lance
H5=$CACHES/tworoom_play.h5
CACHE1=$CACHES/tworoom_lewm_fs1.pt
CACHE5=$CACHES/tworoom_lewm_fs5.pt

mkdir -p "$LOGS" "$CACHES"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGS/tworoom_phase1.log"; }
die() { log "FATAL: $*"; exit 1; }

log "phase 0: WM smoke test"
CUDA_VISIBLE_DEVICES=0 $PY - <<'EOF' >> "$LOGS/tworoom_phase1.log" 2>&1 || die "WM smoke test failed"
import torch, stable_worldmodel as swm
wm = swm.wm.utils.load_pretrained('lewm_tworoom').to('cuda').eval()
x = torch.rand(2, 3, 3, 224, 224, device='cuda')
out = wm.encode({'pixels': x})
assert out['emb'].shape == (2, 3, 192), out['emb'].shape
print('smoke: WM encode OK', flush=True)
EOF
log "smoke OK"

if [ ! -d "/workspace/swm_home/datasets/$PLAY" ]; then
  log "phase 1a: collecting 1000 play episodes (seed 7)"
  $PY /workspace/code/collect_play.py --episodes 1000 --num-envs 32 --seed 7 \
    --out "/workspace/swm_home/datasets/$PLAY" > "$LOGS/tworoom_collect.log" 2>&1 \
    || die "data collection failed (see $LOGS/tworoom_collect.log)"
  log "collected 1000 episodes"
else
  log "phase 1a: dataset exists, skipping"
fi

if [ ! -f "$H5" ]; then
  log "phase 1b: building h5"
  $PY /workspace/code/build_h5.py --dataset "$PLAY" --out "$H5" \
    > "$LOGS/tworoom_build_h5.log" 2>&1 || die "h5 build failed"
  log "h5 built"
fi

if [ ! -f "$CACHE1" ]; then
  log "phase 1c: fs1 latent cache"
  CUDA_VISIBLE_DEVICES=0 $PY "$TRM/cache_latents.py" --wm lewm_tworoom --dataset "$PLAY" \
    --out "$CACHE1" --state-key state --batch-size 512 \
    > "$LOGS/tworoom_cache_fs1.log" 2>&1 || die "fs1 cache failed"
  log "fs1 cache built"
fi

if [ ! -f "$CACHE5" ]; then
  log "phase 1d: fs5 subsample"
  $PY "$TRM/subsample_cache.py" --in "$CACHE1" --out "$CACHE5" --frameskip 5 \
    > "$LOGS/tworoom_cache_fs5.log" 2>&1 || die "fs5 cache failed"
  log "fs5 cache built"
fi

$PY - <<EOF >> "$LOGS/tworoom_phase1.log" 2>&1
import torch
for p in ["$CACHE1", "$CACHE5"]:
    c = torch.load(p, map_location='cpu', weights_only=False)
    z = c['z'] if isinstance(c, dict) and 'z' in c else c
    try:
        print(p, {k: (tuple(v.shape) if hasattr(v, 'shape') else v) for k, v in c.items()} if isinstance(c, dict) else tuple(z.shape), flush=True)
    except Exception as e:
        print(p, 'inspect error', e, flush=True)
EOF
log "PHASE1 DONE"
