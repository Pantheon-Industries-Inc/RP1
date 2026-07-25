"""Contrastive value learning (contrastive RL / InfoNCE).

We learn a critic ``f(z_s, z_g) = phi(z_s) . psi(z_g)`` whose value is monotone
in discounted reachability, following contrastive RL (Eysenbach et al.). For a
batch of anchors ``S`` and geometric-future goals ``G`` (positives on the
diagonal), the symmetric InfoNCE objective is::

    logits = phi(S) @ psi(G)^T  / temperature        # (B, B)
    L = 0.5 * ( CE(logits, arange(B)) + CE(logits^T, arange(B)) )

A high critic value means "goal is reachable from state", so the terminal cost
handed to the planner is the negative critic: ``cost = -f(z_pred, z_goal)``
(lower == more reachable).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from loguru import logger as logging
from torch import nn
from tqdm import tqdm

from ..latent_cache import LatentCache
from ..samplers import GeometricFutureSampler


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class ContrastiveCritic(nn.Module):
    """Bilinear critic with separate state / goal encoders.

    ``forward`` returns the critic similarity ``f(z_a, z_b)`` (higher == more
    reachable); ``cost`` returns its negation for use as a terminal cost.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, rep_dim: int = 64, depth: int = 2):
        super().__init__()
        self.phi = _mlp(latent_dim, hidden_dim, rep_dim, depth)  # state encoder
        self.psi = _mlp(latent_dim, hidden_dim, rep_dim, depth)  # goal encoder

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return (self.phi(z_a) * self.psi(z_b)).sum(dim=-1)

    @torch.no_grad()
    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return -self.forward(z_pred, z_goal)


@dataclass
class ContrastiveConfig:
    hidden_dim: int = 256
    rep_dim: int = 64
    depth: int = 2
    gamma: float = 0.99
    temperature: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    steps: int = 5000
    seed: int = 0


def fit(cache: LatentCache, cfg: ContrastiveConfig, device: str = "cpu") -> ContrastiveCritic:
    """Train and return a :class:`ContrastiveCritic`."""
    torch.manual_seed(cfg.seed)
    critic = ContrastiveCritic(
        cache.latent_dim, hidden_dim=cfg.hidden_dim, rep_dim=cfg.rep_dim, depth=cfg.depth,
    ).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = GeometricFutureSampler(cache, gamma=cfg.gamma, seed=cfg.seed)

    critic.train()
    pbar = tqdm(range(cfg.steps), desc="contrastive")
    for step in pbar:
        batch = sampler.sample(cfg.batch_size)
        z_s = batch["z_s"].to(device)
        z_g = batch["z_g"].to(device)
        s_rep = critic.phi(z_s)  # (B, R)
        g_rep = critic.psi(z_g)  # (B, R)
        logits = (s_rep @ g_rep.t()) / cfg.temperature  # (B, B)
        labels = torch.arange(logits.shape[0], device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            acc = (logits.argmax(dim=1) == labels).float().mean()
            pbar.set_postfix(loss=loss.item(), acc=acc.item())
    logging.success(f"contrastive critic trained ({cfg.steps} steps), final loss={loss.item():.4f}")
    critic.eval()
    return critic


__all__ = ["ContrastiveCritic", "ContrastiveConfig", "fit"]
