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

    @torch.no_grad()
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

    @torch.no_grad()
    def cost(self, z_pred, z_goal):
        return self.forward(z_pred, z_goal)




def _interval_union_length(s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Lebesgue measure of the union of intervals ``[s_j, e_j]`` along the last
    dim (an interval is empty when ``e_j <= s_j``).

    Sort by start, then sweep with an exclusive cumulative max of the ends: each
    interval contributes only the part beyond every earlier interval's reach.
    Fully vectorised and differentiable (argsort is a permutation; gradients
    flow through gather/cummax to the right elements).
    """
    order = s.argsort(dim=-1)
    s_ = s.gather(-1, order)
    e_ = e.gather(-1, order)
    cm = e_.cummax(dim=-1).values                     # inclusive
    neg_inf = torch.full_like(cm[..., :1], float("-inf"))
    prev = torch.cat([neg_inf, cm[..., :-1]], dim=-1)  # exclusive
    lo = torch.maximum(s_, prev)
    return (e_ - lo).clamp_min(0).sum(-1)


class IQEHead(nn.Module):
    """Interval Quasimetric Embedding head (Wang & Isola, 2022).

        d(z_i -> z_j) = scale * maxmean_k | union_j [ u(z_i)_kj , u(z_j)_kj ] |

    ``num_components`` k groups of ``embed_dim // k`` dims each. Asymmetric by
    construction (intervals only count where the target coordinate is larger),
    zero on the diagonal, and satisfies the triangle inequality -- so it still
    stitches, but without MRN's symmetric Euclidean term.
    """

    def __init__(self, latent_dim, hidden_dim=256, embed_dim=128, depth=2,
                 num_components=8, alpha_init=0.75):
        super().__init__()
        assert embed_dim % num_components == 0, "embed_dim must divide by num_components"
        self.latent_dim = latent_dim
        self.k = num_components
        self.d = embed_dim // num_components
        layers = [nn.Linear(latent_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, embed_dim)]
        self.enc = nn.Sequential(*layers)
        a = float(min(max(alpha_init, 1e-4), 1 - 1e-4))
        self.alpha_raw = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(a)))))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, z_i, z_j):
        ei = self.enc(z_i).unflatten(-1, (self.k, self.d))
        ej = self.enc(z_j).unflatten(-1, (self.k, self.d))
        comp = _interval_union_length(ei, ej)          # (..., k)
        a = torch.sigmoid(self.alpha_raw)
        agg = a * comp.max(dim=-1).values + (1.0 - a) * comp.mean(dim=-1)
        return self.log_scale.exp() * agg

    @torch.no_grad()
    def cost(self, z_pred, z_goal):
        return self.forward(z_pred, z_goal)


__all__ = ["PairwiseMetricHead", "QuasimetricHead", "IQEHead", "pair_features"]
