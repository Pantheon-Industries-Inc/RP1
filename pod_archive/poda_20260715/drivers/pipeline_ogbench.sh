#!/bin/bash
# End-to-end LIP-AC pipeline on a fresh pod: wait for dataset -> verify h5 ->
# build fs1/fs5 caches (4-GPU sharded) -> verify caches -> run_ac_ogbench.sh.
set -u
export PYTHONPATH=/workspace/code/stable-worldmodel
export MUJOCO_GL=osmesa
export TQDM_DISABLE=1
LOGS=/workspace/logs
H5=/workspace/datasets/lewm_cube_full/cube_single_expert.h5
WM=/workspace/ckpts/ogbench_cube_single_v2WM
mkdir -p "$LOGS" /workspace/caches

log() { echo "[$(date +%H:%M:%S)] PIPELINE: $*" | tee -a "$LOGS/pipeline.log"; }
die() { log "FATAL: $*"; exit 1; }

# ------------------------------------------------------- 1. dataset
log "waiting for dataset download/extract"
for i in $(seq 1 1080); do
  [ -f /workspace/datasets/lewm_cube_full/extract.done ] && break
  sleep 20
done
[ -f /workspace/datasets/lewm_cube_full/extract.done ] || die "dataset never finished (3h)"
log "dataset ready: $(du -sh $H5 | cut -f1)"

# ------------------------------------------------------- 2. verify h5 (add ep_offset if absent)
python3 - <<'PY' >> "$LOGS/pipeline.log" 2>&1 || die "h5 verification failed"
import h5py, hdf5plugin, numpy as np  # hdf5plugin registers the pixels compression filter
p = "/workspace/datasets/lewm_cube_full/cube_single_expert.h5"
with h5py.File(p, "r") as h:
    keys = set(h.keys())
    print("h5 keys:", sorted(keys))
    need = {"pixels", "action", "ep_len", "ep_offset", "qpos", "qvel"}
    assert not (need - keys), f"missing {need - keys}"
    ln = h["ep_len"][:]
    assert ln.sum() == 2010000 and (ln == 201).all(), (int(ln.sum()),)
    px = h["pixels"]
    assert px.dtype == np.uint8 and px.shape[1:] == (224, 224, 3), (px.dtype, px.shape)
    print("action:", h["action"].shape, "pixels:", px.shape, px.dtype)
    print("first/last frame readable:", float(px[0].mean()), float(px[-1].mean()))
    print("last action readable:", h["action"][-1])
print("H5 OK")
PY
log "h5 verified"

# ------------------------------------------------------- 3. caches (skip if built)
if [ ! -f /workspace/caches/caches.done ]; then
  log "building fs1 cache: 4 shards x 2500 episodes"
  for i in 0 1 2 3; do
    s=$((i * 2500)); e=$(((i + 1) * 2500))
    CUDA_VISIBLE_DEVICES=$i python3 /workspace/valscripts/cache_cube_full.py \
      --h5 "$H5" --wm "$WM" --stride 1 --ep-start "$s" --ep-end "$e" \
      --out /workspace/caches/fs1_shard$i.pt > "$LOGS/cache_shard$i.log" 2>&1 &
  done
  wait
  for i in 0 1 2 3; do
    [ -f /workspace/caches/fs1_shard$i.pt ] || die "shard $i missing, see $LOGS/cache_shard$i.log"
  done
  log "merging shards -> fs1 + derived fs5"
  python3 /workspace/valscripts/merge_caches.py \
    /workspace/caches/cube_full_fs1.pt /workspace/caches/cube_full_fs5.pt \
    /workspace/caches/fs1_shard0.pt /workspace/caches/fs1_shard1.pt \
    /workspace/caches/fs1_shard2.pt /workspace/caches/fs1_shard3.pt \
    > "$LOGS/cache_merge.log" 2>&1 || die "merge failed, see $LOGS/cache_merge.log"
  rm -f /workspace/caches/fs1_shard*.pt
  echo done > /workspace/caches/caches.done
fi

python3 - <<'PY' >> "$LOGS/pipeline.log" 2>&1 || die "cache verification failed"
from stable_worldmodel.trm import LatentCache
c1 = LatentCache.load("/workspace/caches/cube_full_fs1.pt")
c5 = LatentCache.load("/workspace/caches/cube_full_fs5.pt")
assert len(c1.z) == 2010000, len(c1.z)
assert len(c5.z) == 10000 * 41, len(c5.z)          # ceil(201/5) = 41 per episode
assert c1.latent_dim == c5.latent_dim == 192
eps5 = c5.episodes()
assert len(eps5) == 10000 and all(len(r) == 41 for r in eps5.values())
print("caches OK: fs1", c1.z.shape, "fs5", c5.z.shape)
PY
log "caches verified"

# ------------------------------------------------------- 4. experiment
log "launching run_ac_ogbench.sh"
bash /workspace/run_ac_ogbench.sh >> "$LOGS/pipeline.log" 2>&1
log "pipeline complete"
