#!/bin/bash
# Bare-pod bootstrap for the DINO-WM TD/LIP campaign, 2026-07-31.
#
# ORDER MATTERS (this cost 12 min on the last pod, see [[pod-container-rebuild]]):
#   1. env extras FIRST -- they drag torch 2.4.1 -> 2.13 and numpy -> 2.x
#   2. THEN re-pin torch/torchvision/numpy
#   3. THEN force-reinstall the cu12 NVIDIA wheels, because torch 2.13 leaves
#      nvidia-*-cu13 wheels that shadow cuDNN (CUDNN_STATUS_NOT_INITIALIZED),
#      and plain uninstall deletes cu12's .so while pip still lists it.
#   Verify with a real GPU conv + cudnn.version() == 90100. `import torch`
#   succeeding proves nothing here.
#
# Dataset goes on LOCAL disk: /workspace is MFS at ~63 MB/s, and the encode
# reads all 20 GB of it. Caches go local too, for the same reason.
set -u
L=/workspace/logs; mkdir -p "$L" /root/datasets /root/caches /workspace/ckpts
DRV=$L/driver_bootstrap2.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$DRV"; }

log "=== P1 apt: EGL/GL ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > "$L/apt.log" 2>&1
apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0 fonts-dejavu-core \
  >> "$L/apt.log" 2>&1 || log "  apt warnings (continuing)"
ldconfig

log "=== P2 HF dataset -> local disk (20.5 GB), backgrounded ==="
pip install -q --break-system-packages huggingface_hub hf_transfer >> "$L/pip0.log" 2>&1
(
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<'PY' > "$L/hfdl.log" 2>&1
from huggingface_hub import snapshot_download
p = snapshot_download('galilai-group/ogb_cube_single', repo_type='dataset',
                      local_dir='/root/datasets/ogb_cube_single', max_workers=16)
print('downloaded', p)
PY
  touch /root/.hfdl_done
) &

log "=== P3 pip: env extras first (they will bump torch; fixed in P4) ==="
pip install -q --break-system-packages \
  "transformers==4.49.0" "lightning==2.6.5" "stable-pretraining==0.1.7" \
  "mujoco==3.10.0" "dm_control==1.0.43" "ogbench==1.2.1" "pylance==8.0.0" lancedb \
  hydra-core omegaconf h5py loguru einops timm pillow imageio imageio-ffmpeg \
  opencv-python-headless wandb pygame pymunk shapely minigrid gymnasium-robotics \
  "stable_baselines3>=2.0.0" ale-py typer rich tabulate hdf5plugin \
  matplotlib scikit-learn pandas tqdm zarr > "$L/pip1.log" 2>&1 \
  && log "  extras ok" || { log "  PIP1 FAILED"; tail -15 "$L/pip1.log" | tee -a "$DRV"; }

log "=== P4 re-pin torch 2.4.1+cu124 / numpy<2 ==="
pip install -q --break-system-packages "numpy<2" torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124 > "$L/pip2.log" 2>&1 \
  && log "  re-pin ok" || { log "  PIP2 FAILED"; tail -10 "$L/pip2.log" | tee -a "$DRV"; }

log "=== P5 purge cu13 shadows, restore cu12 NVIDIA wheels ==="
pip uninstall -y -q --break-system-packages \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
  nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx >> "$L/pip3.log" 2>&1
pip install -q --break-system-packages --force-reinstall --no-deps \
  nvidia-cudnn-cu12==9.1.0.70 nvidia-cublas-cu12==12.4.2.65 \
  nvidia-cuda-runtime-cu12==12.4.99 nvidia-cuda-nvrtc-cu12==12.4.99 \
  nvidia-cuda-cupti-cu12==12.4.99 nvidia-cufft-cu12==11.2.0.44 \
  nvidia-curand-cu12==10.3.5.119 nvidia-cusolver-cu12==11.6.0.99 \
  nvidia-cusparse-cu12==12.3.0.142 nvidia-nccl-cu12==2.20.5 \
  nvidia-nvjitlink-cu12==12.4.99 nvidia-nvtx-cu12==12.4.99 >> "$L/pip3.log" 2>&1 \
  && log "  cu12 restored" || log "  PIP3 had errors"

log "=== P6 smoke: real GPU conv + cudnn version ==="
python3 - <<'PY' 2>&1 | tee -a "$DRV"
import torch, numpy, transformers
print('torch', torch.__version__, '| numpy', numpy.__version__,
      '| transformers', transformers.__version__)
print('cudnn', torch.backends.cudnn.version(), '| cuda', torch.version.cuda,
      '| gpus', torch.cuda.device_count())
x = torch.randn(2, 3, 196, 196, device='cuda'); c = torch.nn.Conv2d(3, 16, 3).cuda()
print('CONV_OK', tuple(c(x).shape))
assert torch.backends.cudnn.version() == 90100, 'WRONG CUDNN -- cu13 still shadowing'
print('CUDNN_PIN_OK')
PY
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python3 -c "
import mujoco
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>')
d=mujoco.MjData(m); mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,64,64); r.update_scene(d); print('EGL_OK', r.render().shape)
" 2>&1 | grep -E "EGL_OK" | tee -a "$DRV"

log "=== P7 waiting on dataset download ==="
for i in $(seq 1 240); do [ -f /root/.hfdl_done ] && break; sleep 30; done
if [ -f /root/.hfdl_done ]; then
  du -sh /root/datasets/ogb_cube_single 2>/dev/null | tee -a "$DRV"
  log "  dataset ready"
else
  log "  DATASET STILL DOWNLOADING (check $L/hfdl.log)"
fi
log "BOOTSTRAP2_DONE"
