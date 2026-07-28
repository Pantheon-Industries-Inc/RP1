#!/usr/bin/env python3
"""Regression fixture: confirm the archived world models reproduce the campaign's
latents in the CURRENT environment.

The reference in ``ref_latents.npz`` was produced on the campaign machine
(Linux / H100 / torch 2.4.1 / transformers 4.57.6). Agreement to ~1e-4 is
float32 accumulation-order noise across BLAS/hardware; anything materially
larger means the environment is no longer reproducing the campaign.

    STABLEWM_HOME=$(pwd) python3 verify_reproduction.py

Exits non-zero on failure. Needs no dataset and no GPU -- seconds to run.
"""
import sys, pathlib
import numpy as np
import torch
import transformers
import stable_worldmodel as swm

TOL = 5e-3  # ~10x the observed cross-version drift (worst was 4.2e-4)
here = pathlib.Path(__file__).parent
ref = np.load(here / "ref_latents.npz")

g = torch.Generator().manual_seed(1234)
px = torch.rand(2, 3, 3, 224, 224, generator=g)
pro = torch.rand(2, 3, 2, generator=g) * 200.0

print(f"torch {torch.__version__} | transformers {transformers.__version__}")
print("reference: torch 2.4.1 / transformers 4.57.6 / H100\n")

bad = 0
for name in ["lejepa_tworoom", "pldm_tworoom", "dinowm_tworoom"]:
    m = swm.wm.utils.load_pretrained(name).eval()
    info = {"pixels": px}
    if getattr(m, "wants_proprio", False):
        info["proprio"] = pro
    with torch.no_grad():
        z = m.encode(info)["emb"].float().numpy()
    d = float(np.abs(z - ref[name]).max())
    ok = d < TOL
    bad += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:18s} max|delta| = {d:.3e}")

print()
if bad:
    print(f"{bad}/3 models do NOT reproduce the reference -- environment has drifted.")
    sys.exit(1)
print("All 3 world models reproduce the campaign latents.")
