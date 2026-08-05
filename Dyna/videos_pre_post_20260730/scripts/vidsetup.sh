#!/bin/bash
# Minimal re-bootstrap of the LeWM/cube eval stack on a recycled container.
# Same pins as Dyna/dyna_harness/pod2_setup.sh (torch 2.4.1+cu124 already present),
# so rendered rollouts come from the identical stack that produced the cards.
set -u
L=/workspace/logs; mkdir -p "$L"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/driver_vidsetup.log"; }
log "=== P1 apt: EGL/GL ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > "$L/vidsetup_apt.log" 2>&1
apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0 >> "$L/vidsetup_apt.log" 2>&1 || log "apt warnings"
ldconfig
log "=== P2 pip: pinned stack ==="
pip install -q --break-system-packages \
  "transformers==4.49.0" "lightning==2.6.5" "stable-pretraining==0.1.7" \
  "mujoco==3.10.0" "dm_control==1.0.43" "ogbench==1.2.1" "pylance==8.0.0" lancedb \
  hydra-core omegaconf h5py hf_transfer huggingface_hub loguru einops timm pillow \
  imageio imageio-ffmpeg opencv-python-headless wandb > "$L/vidsetup_pip.log" 2>&1 \
  && log "pip ok" || { log "PIP FAILED"; tail -20 "$L/vidsetup_pip.log" | tee -a "$L/driver_vidsetup.log"; exit 1; }
log "=== P3 smoke ==="
python3 -c "
import torch, transformers, mujoco, hydra, lance, imageio
print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.device_count(), \"gpus\")
print(\"transformers\", transformers.__version__, \"| mujoco\", mujoco.__version__)
" 2>&1 | tee -a "$L/driver_vidsetup.log"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python3 -c "
import mujoco
m=mujoco.MjModel.from_xml_string(\"<mujoco><worldbody><body><geom size=\\\"0.1\\\"/></body></worldbody></mujoco>\")
d=mujoco.MjData(m); mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,64,64); r.update_scene(d); px=r.render()
print(\"EGL_OK\", px.shape, round(float(px.mean()),3))
" 2>&1 | grep -E "EGL_OK|Error|error" | head -3 | tee -a "$L/driver_vidsetup.log"
log "VIDSETUP_DONE"
