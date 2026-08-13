"""Build l2window{N}.pt — the unlearned N-frame window-L2 cost blob.

N=1 is the paper's own terminal cost (plain latent L2 on the last imagined
frame) routed through the IDENTICAL metric hook the window-3 arms use, so the
w1 latent cells stay protocol-identical to the w3 cells: same hook, same goal
handling, only the declared latent_dim (and therefore the hook's frame count)
changes. Norm form matches the existing l2window3 control (torch.norm, not
squared) — rank-identical for CEM/top-k selection either way.

Reuses reacher/scripts_20260729/make_l2window.py verbatim for the trm/io.py
`l2` learner patch and the 3-frame blob (both idempotent), then saves the
N-frame blob with latent_dim = N * D.
"""

import argparse
import pathlib
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, required=True)
parser.add_argument("--latent-dim", type=int, default=192, help="base WM latent dim D (192 for lejepa/pldm reacher)")
parser.add_argument("--lag", type=int, default=5)
parser.add_argument("--out", default=None)
args = parser.parse_args()

# Idempotent: patches trm/io.py with the parameter-free `l2` learner and
# (re)builds /workspace/metrics/l2window3.pt with its own round-trip check.
subprocess.run(
    [sys.executable, str(pathlib.Path.home() / "repo/reacher/scripts_20260729/make_l2window.py")],
    check=True,
)

import torch  # noqa: E402

import stable_worldmodel.trm.io as io_mod  # noqa: E402

dim = args.frames * args.latent_dim
out = args.out or f"/workspace/metrics/l2window{args.frames}.pt"
arch = {"window_frames": args.frames, "window_lag": args.lag, "unlearned": True}
metric = io_mod.build_metric("l2", dim, arch)
x = torch.randn(4, dim, requires_grad=True)
y = torch.randn(4, dim)
cost = metric.cost(x, y)
cost.sum().backward()
assert cost.shape == (4,) and x.grad is not None, "cost must be (B,) and gradient-safe"
io_mod.save_metric(metric, "l2", dim, arch, out)
reloaded = io_mod.load_metric(out)
assert float((reloaded.cost(x.detach(), y) - cost.detach()).abs().max()) < 1e-6, "round-trip mismatch"
print(f"saved + round-trip verified -> {out} (latent_dim={dim} => hook _m={args.frames})")
