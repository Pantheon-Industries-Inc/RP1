"""Pairwise metric heads for Trajectory Reachability Metrics (TRM).

The core object is a small scalar-output network operating on a pair of latent
states ``(z_i, z_j)`` through the feature map

    phi(z_i, z_j) = [ z_i , z_j , z_i - z_j , |z_i - z_j| ]

as described in "Beyond Euclidean Proximity" (arXiv 2605.22164). The head is a
two-hidden-layer MLP (256 units, SiLU) with a Softplus scalar output, so the
metric is non-negative.

Every metric module in this package exposes a common inference contract::

    module.cost(z_pred, z_goal) -> Tensor   # lower == more reachable / closer

so that :class:`stable_worldmodel.trm.cost.MetricCost` can use any learner
(regression / TD / contrastive) behind the same terminal-cost interface.
"""

from __future__ import annotations

import torch
from torch import nn


def pair_features(z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
    """Feature map ``[z_i, z_j, z_i - z_j, |z_i - z_j|]``.

    Args:
        z_i: Latent tensor of shape ``(..., D)``.
        z_j: Latent tensor of shape ``(..., D)`` (broadcastable to ``z_i``).

    Returns:
        Tensor of shape ``(..., 4 * D)``.
    """
    diff = z_i - z_j
    return torch.cat([z_i, z_j, diff, diff.abs()], dim=-1)


class PairwiseMetricHead(nn.Module):
    """Two-hidden-layer MLP scalar metric over latent pairs.

    Args:
        latent_dim: Dimension ``D`` of the latent states.
        hidden_dim: Width of each hidden layer (paper uses 256).
        depth: Number of hidden layers (paper uses 2).
        softplus: If ``True`` (default) apply Softplus to the output so the
            metric is non-negative. TD / contrastive learners that want a raw
            scalar can disable it.
        symmetric: If ``True``, evaluate both orderings and average, yielding a
            symmetric metric ``m(z_i, z_j) = m(z_j, z_i)``. The paper instead
            relies on random-order pair sampling, so this defaults to ``False``.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        softplus: bool = True,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.symmetric = symmetric
        self.use_softplus = softplus

        layers: list[nn.Module] = [nn.Linear(4 * latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Softplus() if softplus else nn.Identity()

    def _raw(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        feats = pair_features(z_i, z_j)
        return self.out(self.mlp(feats).squeeze(-1))

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Predicted scalar (e.g. temporal separation) for pairs ``(z_i, z_j)``.

        Shapes ``(..., D), (..., D) -> (...)``.
        """
        out = self._raw(z_i, z_j)
        if self.symmetric:
            out = 0.5 * (out + self._raw(z_j, z_i))
        return out

    # No @torch.no_grad() here: GradientSolver (Adam) backprops through the
    # terminal cost, and decorating this silently kills every TD+Adam run —
    # the latent-cost Adam path works, so the failure looks like a solver bug
    # rather than a missing gradient. CEM/MPPI lose nothing: their solve() is
    # already wrapped in inference_mode.
    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Terminal cost (lower == more reachable). For regression this is the
        predicted temporal distance directly."""
        return self.forward(z_pred, z_goal)


class QuasimetricHead(nn.Module):
    """Metric Residual Network (MRN) quasimetric head.

    Reachability (steps-to-go) is a *directed* quasimetric: non-negative, and it
    obeys the triangle inequality, which is what lets a value **stitch** long
    distances from short transitions. MRN parameterises::

        d(z_i -> z_j) = ||u(z_i) - u(z_j)||_2  +  max_k ReLU( v(z_j)_k - v(z_i)_k )

    a symmetric Euclidean term + an asymmetric residual; the sum is a quasimetric
    (Liu et al., 2022). Better inductive bias than a generic MLP for goal-
    conditioned TD distance, especially at long range.
    """

    def __init__(self, latent_dim, hidden_dim=256, embed_dim=128, depth=2, sym_frac=0.5):
        super().__init__()
        self.latent_dim = latent_dim
        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, embed_dim)]
        self.enc = nn.Sequential(*layers)
        self.sym_dim = max(1, int(embed_dim * sym_frac))

    def forward(self, z_i, z_j):
        ei, ej = self.enc(z_i), self.enc(z_j)
        s = self.sym_dim
        d_sym = (ei[..., :s] - ej[..., :s]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
        d_asym = torch.relu(ej[..., s:] - ei[..., s:]).max(dim=-1).values
        return d_sym + d_asym

    # No @torch.no_grad() here: GradientSolver (Adam) backprops through the
    # terminal cost, and decorating this silently kills every TD+Adam run —
    # the latent-cost Adam path works, so the failure looks like a solver bug
    # rather than a missing gradient. CEM/MPPI lose nothing: their solve() is
    # already wrapped in inference_mode.
    def cost(self, z_pred, z_goal):
        return self.forward(z_pred, z_goal)


__all__ = ["PairwiseMetricHead", "QuasimetricHead", "pair_features"]
