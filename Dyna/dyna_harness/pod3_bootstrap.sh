#!/bin/bash
# Bootstrap a fresh pod that has the SAME volume + NVIDIA_DRIVER_CAPABILITIES
# set, so cube evals can run under egl (the in-domain renderer).
#
# Ordering is deliberate: EGL is verified FIRST, before any install, because
# the whole point of this pod is the graphics capability. If the flag did not
# take we learn it in seconds, not after a 20-minute torch download.
#
# STALE-MARKER HAZARD: /workspace/.pip_done etc. live on the VOLUME but the
# python stack lives in the CONTAINER (/usr/local). A new container therefore
# inherits markers claiming an install that is not there. Everything below is
# verified by IMPORT, never by marker.
#
# Volume assets expected to persist (nothing here re-downloads them):
#   datasets/ogb_cube_single/  caches/pldm_tr8000_fs{1,5}.pt  metrics/pldm_TD.pt
#   models/PLDM_OgBench_lewm (keys already inverted to the 4.49 layout)
#   actors/lip4_pldm_mw*_s0.pt (8 of 12 stage-A actors from the osmesa pod)
set -u
L=/workspace/logs; mkdir -p "$L"
DRV=$L/driver_bootstrap.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }
log "=== POD3 BOOTSTRAP (pid $$) ==="

# ------------------------------------------------------- P0 EGL capability gate
log "P0: EGL capability (the reason this pod exists)"
log "  NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
ICD=$(ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null | tr '\n' ' ')
log "  egl_vendor.d: ${ICD:-EMPTY}"
if ! ls /usr/share/glvnd/egl_vendor.d/*nvidia*.json >/dev/null 2>&1; then
  log "  no NVIDIA ICD json"
  die "NVIDIA EGL not mounted. Recreate the pod with NVIDIA_DRIVER_CAPABILITIES=all (or compute,utility,graphics). Do NOT apt-install libnvidia-gl-* -- it pulls libnvidia-compute at a different patch level than the running kernel module and breaks CUDA."
fi
ldconfig -p | grep -q libEGL_nvidia || die "libEGL_nvidia.so.0 absent despite ICD json"
log "P0: NVIDIA EGL present"

# --------------------------------------------------------- P1 apt render libs
export DEBIAN_FRONTEND=noninteractive
dpkg -s libegl1 >/dev/null 2>&1 || {
  log "P1: apt render libs"
  apt-get update -qq > "$L/bs_apt.log" 2>&1
  apt-get install -y -qq libegl1 libgl1 libglvnd0 libglib2.0-0 libosmesa6 \
    >> "$L/bs_apt.log" 2>&1 || log "  apt warnings (continuing)"
}
log "P1: done"

# ------------------------------------------- P2 python stack (import-verified)
need_install=0
python3 - <<'PY' >/dev/null 2>&1 || need_install=1
import torch, transformers, lance, lancedb, mujoco, dm_control, h5py, hydra
assert torch.cuda.is_available()
assert torch.__version__.startswith("2.4.1")
PY
if [ "$need_install" = 1 ]; then
  log "P2: installing pinned stack (fresh container -- markers on the volume are stale)"
  pip install -q --break-system-packages \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 \
    > "$L/bs_pip.log" 2>&1 || die "torch install failed"
  # lancedb, pygame, pymunk, shapely are NOT optional: swm imports lancedb at
  # module load, and the cube env pulls pygame/pymunk/shapely through its
  # gym registration. All four were missing on the bare image.
  pip install -q --break-system-packages \
    "transformers==4.49.0" "lightning==2.6.5" "stable-pretraining==0.1.7" \
    "mujoco==3.10.0" "dm_control==1.0.43" "ogbench==1.2.1" "pylance==8.0.0" \
    lancedb pygame pymunk shapely \
    hydra-core omegaconf h5py hf_transfer huggingface_hub loguru einops timm \
    imageio imageio-ffmpeg opencv-python-headless wandb \
    >> "$L/bs_pip.log" 2>&1 || die "deps install failed"
else
  log "P2: stack already present"
fi
python3 -c "
import torch, transformers
print('  torch', torch.__version__, '| cuda', torch.version.cuda,
      '| gpus', torch.cuda.device_count(), '| sm_90', 'sm_90' in torch.cuda.get_arch_list())
print('  transformers', transformers.__version__)
" 2>&1 | tee -a "$DRV" || die "stack verification failed"
PYTHONPATH=/workspace/code/stable-worldmodel python3 -c "import stable_worldmodel" 2>&1 \
  | grep -v UserWarning | grep -qE "Error|Traceback" && die "stable_worldmodel import broken"
log "P2: stack OK"

# ------------------------------------------------------ P3 volume assets
log "P3: volume assets"
for f in /workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
         /workspace/datasets/expert_actions.h5 \
         /workspace/caches/pldm_tr8000_fs1.pt /workspace/caches/pldm_tr8000_fs5.pt \
         /workspace/metrics/pldm_TD.pt /workspace/models/PLDM_OgBench_lewm/weights.pt \
         /workspace/code/stable-worldmodel/scripts/plan/eval_wm.py; do
  [ -e "$f" ] || die "volume asset missing: $f (is this the same volume?)"
done
log "  stage-A actors present: $(ls /workspace/actors/ 2>/dev/null | grep -cE 'lip4_pldm_mw.*_s0\.pt$')/12"
log "P3: assets OK"

# ------------------------------------------- P4 live EGL render smoke (10 tasks)
log "P4: EGL render smoke"
export PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 TQDM_DISABLE=1 OMP_NUM_THREADS=8
cd /workspace/code/stable-worldmodel
timeout 1800 python3 scripts/plan/eval_wm.py --config-name cube seed=43 \
  eval.dataset_name=/workspace/datasets/ogb_cube_single/ogb_cube_single.lance \
  ++bf16=true eval.img_size=224 eval.goal_offset_steps=25 eval.eval_budget=50 \
  eval.num_eval=10 "+eval.ep_range=8000:10000" \
  policy=/workspace/models/PLDM_OgBench_lewm solver=cem \
  output.filename=bs_egl_smoke.txt > "$L/bs_egl_smoke.log" 2>&1
grep -q "ep_range 8000:10000" "$L/bs_egl_smoke.log" \
  || { tail -12 "$L/bs_egl_smoke.log"; die "EGL smoke failed -- see $L/bs_egl_smoke.log"; }
log "P4: EGL renders. $(grep -oE "success_rate.: [0-9.]+" "$L/bs_egl_smoke.log" | tail -1) (n=10, not a card number)"
log "BOOTSTRAP_DONE -- launch: nohup bash /workspace/pldm_grid_egl.sh > /workspace/logs/nohup_grid_egl.log 2>&1 &"
