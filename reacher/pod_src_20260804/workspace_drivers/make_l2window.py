"""Build the UNLEARNED window3-L2 cost = the Latent+CEM-window arm.

RECONSTRUCTED 2026-07-30 after losing volume 3cv5zzezm9.

Role: this is the control that separates "the cost reads 3 imagined frames"
from "the cost is a learned quasimetric". Plain euclidean distance on the SAME
3-frame stacked window (hook feeds the last 3 imagined frames, goal tiled),
zero learning.

What it settled (lejepa, h25, n=6): latched 89.7 vs the LEARNED window
quasimetric's 86.7 -- so the latched window gain is information access, not
metric learning. Held-at-end it scores 42.7 (h25) but degrades with horizon
(42.3 at h50, 38.0 at h100) exactly where the learned metric's long-range
stitching takes over -- the clean crossover in the campaign.

Adds an `l2` learner branch to trm/io.py (parameter-free module riding the
standard metric contract), then saves one 576-d blob for the 192-d bases.
"""

import torch

# ---------------------------------------------------------------- patch io.py
P = "/workspace/swm_cem/stable_worldmodel/trm/io.py"
s = open(P).read()
if 'learner == "l2"' in s:
    print("io.py: l2 learner already present")
else:
    OLD = "def _build_inner(learner: str, latent_dim: int, arch: dict) -> nn.Module:"
    NEW = '''class L2WindowCost(nn.Module):
    """Unlearned L2 on (optionally frame-stacked) latents.

    Control for learned window metrics: declares ``latent_dim = m * D`` so the
    eval hook feeds the same m-frame imagined window (goal tiled), but the cost
    is plain euclidean distance -- separating "reads more frames" from
    "learned metric". Parameter-free; gradient-safe by construction (never
    decorate ``cost`` with no_grad: GradientSolver and LIP backprop through it).
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return torch.norm(z_pred - z_goal, dim=-1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        return self.cost(z_i, z_j)

    def extra_repr(self) -> str:
        return f"latent_dim={self.latent_dim} (unlearned)"


def _build_inner(learner: str, latent_dim: int, arch: dict) -> nn.Module:
    if learner == "l2":
        # unlearned control -- see L2WindowCost
        return L2WindowCost(latent_dim)'''
    assert s.count(OLD) == 1, f"anchor x{s.count(OLD)}"
    open(P, "w").write(s.replace(OLD, NEW))
    print("io.py: l2 learner added")

import ast

ast.parse(open(P).read())

# ---------------------------------------------------------------- save blob
import importlib

import stable_worldmodel.trm.io as io_mod

importlib.reload(io_mod)

ARCH = {"window_frames": 3, "window_lag": 5, "unlearned": True}
m = io_mod.build_metric("l2", 576, ARCH)
x, y = torch.randn(4, 576, requires_grad=True), torch.randn(4, 576)
c = m.cost(x, y)
c.sum().backward()
assert c.shape == (4,) and x.grad is not None, "cost must be (B,) and gradient-safe"
io_mod.save_metric(m, "l2", 576, ARCH, "/workspace/metrics/l2window3.pt")
m2 = io_mod.load_metric("/workspace/metrics/l2window3.pt")
assert float((m2.cost(x.detach(), y) - c.detach()).abs().max()) < 1e-6, "round-trip mismatch"
print("saved + round-trip verified -> /workspace/metrics/l2window3.pt "
      "(latent_dim=576 => hook _m=3)")
