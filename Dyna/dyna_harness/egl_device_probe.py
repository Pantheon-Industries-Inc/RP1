"""Probe: does MUJOCO_EGL_DEVICE_ID (not CUDA_VISIBLE_DEVICES) decide which
physical GPU MuJoCo's EGL contexts land on?

Hypothesis under test (mujoco/egl/__init__.py:create_initialized_egl_device_display):
with MUJOCO_EGL_DEVICE_ID unset, every process iterates eglQueryDevicesEXT() in
the SAME order and returns the FIRST device that initializes -- so N concurrent
processes all pile their contexts onto the same physical GPU regardless of
CUDA_VISIBLE_DEVICES. That is the suspected cause of the 3-parallel-eval SIGABRT.

Usage:  python3 egl_device_probe.py <n_contexts> <hold_seconds>
Prints one JSON line, then holds the contexts open so an external nvidia-smi can
attribute GPU memory. Exit code 0 = all contexts created, 1 = failed partway.
"""
import json
import os
import sys
import time

n_ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 50
hold = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0

rec = {
    "pid": os.getpid(),
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
    "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
}

try:
    from mujoco.egl import egl_ext as EGL

    # does EGL device enumeration honour CUDA_VISIBLE_DEVICES at all?
    rec["n_egl_devices"] = len(EGL.eglQueryDevicesEXT())
except Exception as e:  # noqa: BLE001
    rec["n_egl_devices"] = f"{type(e).__name__}: {e}"

ctxs = []
try:
    from mujoco.egl import GLContext

    for i in range(n_ctx):
        c = GLContext(224, 224)
        c.make_current()
        ctxs.append(c)
    rec["contexts_created"] = len(ctxs)
    rec["ok"] = True
except Exception as e:  # noqa: BLE001
    rec["contexts_created"] = len(ctxs)
    rec["ok"] = False
    rec["error"] = f"{type(e).__name__}: {e}"

print(json.dumps(rec), flush=True)
time.sleep(hold)
sys.exit(0 if rec.get("ok") else 1)
