"""Extract the hyperparameters that were ACTUALLY used, from the artifacts.

The report's tables were written from what the drivers were meant to pass. That
is not the same as what ran -- a driver can be edited after a run, a flag can be
silently ignored, a default can differ from the documented one. Every trainer
here records its effective configuration into the checkpoint it writes, so the
checkpoints are the ground truth and this dumps them.

Covers exactly the artifacts behind the reported numbers:
  LIP    actors/lip4_leak6_{base}_s{0..5}.pt      (the @0.1 table row)
  PWM    actors/pwmabl_{base}_s1000_terminal_s{0..2}.pt
  value  metrics/window3_{base}_e005_g098.pt      (shared by every Value+X arm
                                                   and both actors' init)
Writes JSON to stdout and a markdown table, and flags any field that disagrees
across seeds or across bases where it should not.
"""

import glob
import json
import os
import sys

import torch

SCALARS = (int, float, str, bool, type(None))


def meta(path):
    """Scalar-valued metadata only -- never state dicts or tensors."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(b, dict):
        return {"_type": type(b).__name__}
    out = {}
    for k, v in b.items():
        if isinstance(v, SCALARS):
            out[k] = v
        elif isinstance(v, dict) and k in ("arch", "cfg", "config", "hypers"):
            out[k] = {kk: vv for kk, vv in v.items() if isinstance(vv, SCALARS)}
        elif hasattr(v, "shape"):
            out[k] = f"<tensor {tuple(v.shape)}>"
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


def group(paths, label):
    recs = {}
    for p in sorted(paths):
        if os.path.exists(p):
            recs[os.path.basename(p)] = meta(p)
    if not recs:
        print(f"## {label}\n\n  (no artifacts found)\n")
        return {}
    keys = sorted({k for r in recs.values() for k in r})
    print(f"## {label}   ({len(recs)} artifacts)\n")
    const, vary = {}, {}
    for k in keys:
        vals = {json.dumps(r.get(k), sort_keys=True, default=str) for r in recs.values()}
        (const if len(vals) == 1 else vary)[k] = vals
    print("**identical across all artifacts**\n")
    for k in sorted(const):
        v = json.loads(list(const[k])[0]) if const[k] else None
        if k in ("sd", "state_dict"):
            continue
        print(f"  - `{k}` = `{v}`")
    if vary:
        print("\n**varies** (expected: seed, per-base knobs, paths)\n")
        for k in sorted(vary):
            print(f"  - `{k}`:")
            for name, r in recs.items():
                print(f"      {name:52} {r.get(k)}")
    print()
    return recs


print("# Effective hyperparameters, read back from the artifacts\n")
print("Source of truth: the configuration each trainer recorded into the")
print("checkpoint it wrote, not the driver scripts.\n")

all_recs = {}
for base in ("lejepa", "pldm"):
    all_recs[f"LIP {base}"] = group(
        glob.glob(f"/workspace/actors/lip4_leak6_{base}_s?.pt"),
        f"LIP actors — {base} (reported @0.1 row)")
for base in ("lejepa", "pldm"):
    all_recs[f"PWM {base}"] = group(
        glob.glob(f"/workspace/actors/pwmabl_{base}_s1000_terminal_s?.pt"),
        f"PWM actors — {base} (terminal, 1000 steps)")
all_recs["value"] = group(
    [f"/workspace/metrics/window3_{b}_e005_g098.pt" for b in ("lejepa", "pldm")],
    "Shared window value (Value+CEM, Value+Adam, and both actors' init)")

with open("/workspace/results/hypers_effective.json", "w") as f:
    json.dump(all_recs, f, indent=2, default=str)
print("wrote /workspace/results/hypers_effective.json", file=sys.stderr)
