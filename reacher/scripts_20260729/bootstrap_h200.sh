#!/bin/bash
# Bootstrap the 6xH200 pod from zero after losing volume 3cv5zzezm9.
# Bare box: 500G local disk, no packages, no data.
#
# Order is chosen so the two long poles (HF dataset download, pip) run first and
# in parallel with everything else. Each stage is idempotent and leaves a marker,
# so a re-run resumes rather than restarts.
set -u
L=/workspace/logs; mkdir -p "$L" /workspace/swm_home/checkpoints /workspace/caches \
  /workspace/metrics /workspace/actors /workspace/results /workspace/datasets_canon/lewm-reacher
log(){ echo "[$(date -u +%m%d-%H:%M:%S)][boot] $*"; }

# ---------------------------------------------------------------- 1. OS + EGL
if [ ! -f /workspace/.boot_apt ]; then
  log "apt: EGL client stack (the tworoom pod needed exactly these)"
  apt-get update -qq 2>/dev/null
  apt-get install -y -qq libegl1 libopengl0 libgles2 rsync zstd 2>&1 | tail -1
  ldconfig
  touch /workspace/.boot_apt
fi
log "apt ok"

# ---------------------------------------------------------------- 2. HF download (long pole, background)
if [ ! -f /workspace/.boot_h5 ]; then
  if [ ! -f /workspace/_hfdl/reacher.tar.zst ]; then
    log "HF download quentinll/lewm-reacher (23.75G tar.zst) -> background"
    pip install -q huggingface_hub hf_transfer 2>&1 | tail -1
    (export HF_HUB_ENABLE_HF_TRANSFER=1
     huggingface-cli download quentinll/lewm-reacher reacher.tar.zst \
       --repo-type dataset --local-dir /workspace/_hfdl > "$L/hfdl.log" 2>&1
     touch /workspace/.hfdl_done) &
  fi
fi

# ---------------------------------------------------------------- 3. python env
if [ ! -f /workspace/.boot_pip ]; then
  log "pip: torch cu124 + the pinned stack (transformers 4.49 = no key renames on convert)"
  pip install -q --upgrade pip 2>&1 | tail -1
  pip install -q torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
  pip install -q "transformers==4.49.0" "mujoco==3.10.0" "dm_control==1.0.43" \
    h5py hdf5plugin pylance "numpy<2" hydra-core loguru einops timm \
    gymnasium imageio imageio-ffmpeg opencv-python-headless pyarrow \
    scikit-learn matplotlib tqdm wandb 2>&1 | tail -2
  touch /workspace/.boot_pip
fi
log "pip ok: $(python3 -c 'import torch,transformers,mujoco; print(torch.__version__, transformers.__version__, mujoco.__version__)' 2>&1 | tail -1)"

# ---------------------------------------------------------------- 4. code
if [ -d /workspace/swm_cem/stable_worldmodel ]; then
  cd /workspace/swm_cem && pip install -q -e . 2>&1 | tail -1 || true
  log "code present + installed editable"
else
  log "WAITING for code rsync from laptop (/workspace/swm_cem)"
fi

# ---------------------------------------------------------------- 5. EGL smoke
if [ -f /workspace/.boot_pip ]; then
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python3 -c "
import mujoco
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>')
d=mujoco.MjData(m); mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,64,64); r.update_scene(d); px=r.render()
print('EGL_OK', px.shape, round(float(px.mean()),3))
" 2>&1 | grep -E "EGL_OK|Error" | head -2
fi

# ---------------------------------------------------------------- 6. extract h5 when the download lands
if [ ! -f /workspace/.boot_h5 ]; then
  for i in $(seq 1 240); do
    [ -f /workspace/.hfdl_done ] && break
    [ -f /workspace/_hfdl/reacher.tar.zst ] && ! pgrep -f huggingface-cli >/dev/null && break
    sleep 30
  done
  if [ -f /workspace/_hfdl/reacher.tar.zst ]; then
    log "extracting reacher.h5 (93G) -- needs ~93G free of the 500G"
    tar -I zstd --no-same-owner -xf /workspace/_hfdl/reacher.tar.zst \
      -C /workspace/datasets_canon/lewm-reacher/ >> "$L/extract.log" 2>&1 \
      && touch /workspace/.boot_h5 && log "h5 extracted: $(du -sh /workspace/datasets_canon/lewm-reacher/reacher.h5 | cut -f1)" \
      || log "FATAL extract failed"
    rm -f /workspace/_hfdl/reacher.tar.zst   # 23G back
  else
    log "FATAL: HF download did not complete"
  fi
fi
log "BOOTSTRAP_STAGE1_DONE"
