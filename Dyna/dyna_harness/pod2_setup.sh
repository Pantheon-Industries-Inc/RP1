#!/bin/bash
# Rebuild the PLDM lane on a fresh pod after the AP-JP-1 volume became
# unattachable (2026-07-30). LeWM needs nothing -- its cards are complete and
# committed. This installs the PINNED stack (torch 2.4.1+cu124 on sm_90; H200
# is Hopper so cu124 is native) so numbers stay comparable to every PLDM cell
# already reported, pulls the expert lance from HF, and rebuilds only what the
# PLDM LIP grid needs: expert_actions.h5, PLDM latent caches, TD teacher.
#
# Anchor (from the take-2 card, per-draw): PLDM latent+CEM draw 43 = 66.0.
# If the anchor misses by >2 pts the stack is not reproducing the old numbers
# and the grid must NOT be compared against the old references.
set -u -o pipefail
L=/workspace/logs; mkdir -p "$L"
DRV=$L/driver_setup.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

log "=== POD2 SETUP (pid $$) ==="

# ---------------------------------------------------------------- P1 apt/EGL
if [ ! -f /workspace/.apt_done ]; then
  log "P1: EGL/GL libs"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq > "$L/setup_apt.log" 2>&1
  apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0 libosmesa6 \
    >> "$L/setup_apt.log" 2>&1 || log "  apt had warnings (continuing)"
  touch /workspace/.apt_done
fi
log "P1: done"

# ------------------------------------------------------------- P2 python stack
if [ ! -f /workspace/.pip_done ]; then
  log "P2: pinned python stack (torch 2.4.1+cu124, transformers 4.49.0)"
  pip install -q --break-system-packages \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 \
    > "$L/setup_pip.log" 2>&1 || die "torch install failed"
  pip install -q --break-system-packages \
    "transformers==4.49.0" "lightning==2.6.5" "stable-pretraining==0.1.7" \
    "mujoco==3.10.0" "dm_control==1.0.43" "ogbench==1.2.1" "pylance==8.0.0" lancedb \
    hydra-core omegaconf h5py hf_transfer huggingface_hub loguru einops timm \
    imageio imageio-ffmpeg opencv-python-headless wandb \
    >> "$L/setup_pip.log" 2>&1 || die "deps install failed"
  touch /workspace/.pip_done
fi
python3 -c "
import torch
assert torch.cuda.is_available(), 'no CUDA'
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'sm90' , 'sm_90' in torch.cuda.get_arch_list(), torch.cuda.device_count(), 'gpus')
import transformers; print('transformers', transformers.__version__)
" 2>&1 | tee -a "$DRV" || die "stack smoke test failed"
log "P2: done"

# ------------------------------------------------------------------ P3 dataset
EXPERT=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance
if [ ! -d "$EXPERT" ]; then
  log "P3: expert lance from HF (galilai-group/ogb_cube_single, ~20.5GB)"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - > "$L/setup_hf.log" 2>&1 <<'PY' || die "HF download failed"
from huggingface_hub import snapshot_download
p = snapshot_download('galilai-group/ogb_cube_single', repo_type='dataset',
                      local_dir='/workspace/datasets/ogb_cube_single',
                      max_workers=16)
print('downloaded to', p)
PY
fi
[ -d "$EXPERT" ] || { ls -R /workspace/datasets/ogb_cube_single | head -20 | tee -a "$DRV"; die "lance not at expected path"; }
python3 - <<'PY' 2>&1 | tee -a "$DRV"
import lance
d = lance.dataset('/workspace/datasets/ogb_cube_single/ogb_cube_single.lance')
cols = [f.name for f in d.schema]
print('expert rows', d.count_rows(), '| has privileged_block_0_pos:',
      'privileged_block_0_pos' in cols, '| qpos:', 'qpos' in cols)
assert d.count_rows() == 2_010_000, 'unexpected row count'
PY
log "P3: dataset ready"
log "SETUP_DONE"
