#!/bin/bash
# Install NVIDIA GL/EGL USERSPACE ONLY, at the exact running driver version, so
# cube evals can use egl on a container the runtime mounted compute-only.
#
# WHY NOT apt: libnvidia-gl-580's only candidate is 580.173.02 and it pulls
# libnvidia-compute-580 with it, against a 580.126.09 kernel module -> the
# classic "Driver/library version mismatch", which kills CUDA.
# WHY THIS WORKS: the .run installer for the EXACT running version installs the
# same-version userspace (libEGL_nvidia, libGLX_nvidia, the vendor ICD json,
# libglvnd). Nothing about the kernel module changes -- --no-kernel-module
# skips it entirely -- so there is no version skew to create.
#
# Running CUDA jobs are safe in principle (already-mapped inodes survive a
# library replacement), but this script still WAITS for training to drain
# before touching anything: a careless kill cost this campaign a training wave
# earlier tonight, and 35 minutes of patience is cheaper than redoing it.
set -u
L=/workspace/logs; mkdir -p "$L"; DRV=$L/driver_nvgl.log
log(){ echo "[$(date -u +%m%d-%H:%M:%S)] $*" | tee -a "$DRV"; }
die(){ log "FATAL: $*"; exit 1; }

V=$(grep -oE "[0-9]+\.[0-9]+\.[0-9]+" /proc/driver/nvidia/version 2>/dev/null | head -1)
[ -n "$V" ] || die "cannot read running driver version"
RUN=/workspace/NVIDIA-Linux-x86_64-${V}.run
log "=== NVIDIA GL userspace install, version $V (pid $$) ==="

# ------------------------------------------------------------------ download
if [ ! -s "$RUN" ]; then
  log "P1: downloading exact-version installer"
  curl -fL --retry 3 -o "$RUN" \
    "https://us.download.nvidia.com/XFree86/Linux-x86_64/${V}/NVIDIA-Linux-x86_64-${V}.run" \
    > "$L/nvgl_dl.log" 2>&1 || die "download failed for $V"
fi
chmod +x "$RUN"
log "P1: installer ready ($(du -h "$RUN" | cut -f1))"

# --------------------------------------------------- wait for CUDA jobs to drain
if [ "${WAIT_FOR_TRAINS:-1}" = 1 ]; then
  while pgrep -f "train_lip_a[c]|train_metri[c]|lewm_exper[t]" >/dev/null 2>&1; do
    log "  waiting: $(pgrep -c -f 'train_lip_a[c]') train procs still running"
    sleep 120
  done
fi
log "P2: no CUDA training running"

# ------------------------------------------------------------------- install
# --no-kernel-module        : never touch the host's module
# --install-libglvnd        : provide the GLVND dispatch + vendor ICD
# --skip-depmod/--no-*-check: containers have no X, no nouveau, no initramfs
log "P3: installing userspace libraries"
sh "$RUN" --silent --no-kernel-module --install-libglvnd \
  --no-drm --no-nvidia-modprobe --no-rebuild-initramfs \
  --skip-depmod --no-x-check --no-nouveau-check --no-questions \
  > "$L/nvgl_install.log" 2>&1 || {
    tail -25 "$L/nvgl_install.log"
    die "installer failed -- see $L/nvgl_install.log (fallback: recreate the pod with NVIDIA_DRIVER_CAPABILITIES=all)"; }
ldconfig
log "P3: install returned OK"

# -------------------------------------------------------------------- verify
log "P4: verification"
log "  ICD: $(ls /usr/share/glvnd/egl_vendor.d/ 2>/dev/null | tr '\n' ' ')"
ldconfig -p | grep -q libEGL_nvidia || die "libEGL_nvidia still absent after install"
# CUDA must still work -- a broken libcuda here is the whole risk being managed
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA BROKEN after install'
x = torch.randn(64, 64, device='cuda') @ torch.randn(64, 64, device='cuda')
print('  CUDA OK:', torch.cuda.device_count(), 'gpus, matmul', tuple(x.shape))
" 2>&1 | tee -a "$DRV" | grep -q "CUDA OK" || die "CUDA broken after install -- recreate the pod"
PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 python3 -c "
import OpenGL.EGL as egl, ctypes
d = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
maj, minr = egl.EGLint(), egl.EGLint()
ok = egl.eglInitialize(d, maj, minr)
print('  eglInitialize:', bool(ok), 'version', maj.value, minr.value)
assert ok, 'eglInitialize failed'
" 2>&1 | tee -a "$DRV" | grep -q "eglInitialize: True" || die "EGL still not initializing"
log "NVGL_DONE -- egl is available"
