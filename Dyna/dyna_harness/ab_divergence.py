"""A/B imagined-vs-reached divergence from LIP_PROBE_DIR dumps.

For consecutive replan rounds t -> t+1 (same env i):
  imagined value  E_t[i]              (value of the plan's imagined terminal)
  reached value   V(z0_{t+1}[i], zg_t[i])  (value at the latent actually
                                            reached after executing the block)
  divergence[i]   = reached - imagined     (>0 = WM was optimistic)

Reports mean/median over all consecutive-round env pairs. The value is the
actor's tandem critic (the exact V the planner maximized at deploy time).

Usage: ab_divergence.py PROBE_DIR VALUE_PT
"""
import sys
from pathlib import Path

import torch

probe_dir, value_pt = Path(sys.argv[1]), sys.argv[2]

from stable_worldmodel.trm import load_metric  # exactly what LIPSolver uses

net = load_metric(value_pt, device="cpu")
net.eval()


def V(z, zg):
    with torch.no_grad():
        return net(z, zg).squeeze(-1)


files = sorted(probe_dir.glob("probe_*.pt"))
assert len(files) >= 2, f"need >=2 rounds, found {len(files)}"
rounds = [torch.load(f, map_location="cpu", weights_only=False) for f in files]

pairs, imag_vals, reached_vals = [], [], []
for t in range(len(rounds) - 1):
    a, b = rounds[t], rounds[t + 1]
    n = min(len(a["E"]), len(b["z0"]))
    imag = a["E"][:n].float()
    reached = V(b["z0"][:n].float(), a["zg"][:n].float())
    d = reached - imag
    pairs.append(d)
    imag_vals.append(imag)
    reached_vals.append(reached)

d = torch.cat(pairs)
imag = torch.cat(imag_vals)
reached = torch.cat(reached_vals)
print(f"pairs: {len(d)} (from {len(files)} rounds)")
print(f"imagined value : mean {imag.mean():.2f}  median {imag.median():.2f}")
print(f"reached  value : mean {reached.mean():.2f}  median {reached.median():.2f}")
print(f"DIVERGENCE     : mean {d.mean():.2f}  median {d.median():.2f}  "
      f"p90 {d.quantile(0.9):.2f}")
