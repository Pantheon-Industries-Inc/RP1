"""What does the refiner CHANGE about a plan, and does it predict the loss?

Probe 3. E1 showed eight learned iterations take a CEM plan worth 79.6 down to
68.9. This reads the per-replan probe dumps and decomposes (refined - init)
three ways, then crosses each with whether the episode was won:

  * WHERE IN THE HORIZON  -- per-block L2 of the delta, early vs late. E16 put
    the fabrication late in the plan; if the delta concentrates there, the fix
    is a trusted prefix / horizon-discounted energy.
  * TEMPORAL FREQUENCY    -- DCT power of the delta, low vs high modes. A real
    action sequence is temporally smooth; a per-step-independent gradient is
    not. If the delta is high-frequency, projecting the update onto low modes
    is a targeted fix that leaves the objective alone.
  * MAGNITUDE             -- how far the plan travels, and how much of the
    clip box it uses.

Env: D (job dir holding probes_*/), OUT (default $D/plan_delta).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

D = Path(os.environ["D"])
OUT = Path(os.environ.get("OUT", str(D / "plan_delta")))
OUT.mkdir(parents=True, exist_ok=True)


def dct_power(x: np.ndarray) -> np.ndarray:
    """Power per temporal DCT mode of (..., H, A), summed over the action axis."""
    h = x.shape[-2]
    k = np.arange(h)
    basis = np.cos(np.pi * (k[:, None] + 0.5) * k[None, :] / h)  # (H, modes)
    coeff = np.einsum("...ha,hm->...ma", x, basis) * (2.0 / h)
    return (coeff**2).sum(-1)


report: dict[str, object] = {}
for probe_dir in sorted(D.glob("probes_*")):
    files = sorted(probe_dir.glob("*.pt"))
    if not files:
        continue
    d_all, a_all, init_all = [], [], []
    for f in files:
        b = torch.load(f, map_location="cpu", weights_only=False)
        if "A_init" not in b or "A" not in b:
            continue
        a, a0 = b["A"].float().numpy(), b["A_init"].float().numpy()
        a_all.append(a)
        init_all.append(a0)
        d_all.append(a - a0)
    if not d_all:
        continue
    delta = np.concatenate(d_all)           # (N, H, A)
    plan = np.concatenate(a_all)
    init = np.concatenate(init_all)
    h = delta.shape[1]
    per_block = np.linalg.norm(delta, axis=-1)          # (N, H)
    half = h // 2
    power = dct_power(delta)                            # (N, modes)
    lo = power[:, : max(1, h // 4)].sum(-1)
    hi = power[:, h // 2 :].sum(-1)
    entry = {
        "replans": int(delta.shape[0]),
        "horizon": int(h),
        "delta_norm_mean": round(float(np.linalg.norm(delta, axis=(1, 2)).mean()), 4),
        "init_norm_mean": round(float(np.linalg.norm(init, axis=(1, 2)).mean()), 4),
        "plan_norm_mean": round(float(np.linalg.norm(plan, axis=(1, 2)).mean()), 4),
        "delta_per_block": [round(float(v), 4) for v in per_block.mean(0)],
        "delta_first_half": round(float(per_block[:, :half].mean()), 4),
        "delta_second_half": round(float(per_block[:, half:].mean()), 4),
        "late_over_early": round(float(per_block[:, half:].mean() / max(per_block[:, :half].mean(), 1e-9)), 3),
        "dct_low_power": round(float(lo.mean()), 5),
        "dct_high_power": round(float(hi.mean()), 5),
        "high_over_low": round(float(hi.mean() / max(lo.mean(), 1e-9)), 3),
        "frac_at_clip": round(float((np.abs(plan) > 0.99 * np.abs(plan).max()).mean()), 4),
    }
    report[probe_dir.name] = entry
    print(f"[plan-delta] {probe_dir.name}: {json.dumps(entry)}", flush=True)

(OUT / "plan_delta.json").write_text(json.dumps(report, indent=1))
print("[plan-delta] DONE", flush=True)
