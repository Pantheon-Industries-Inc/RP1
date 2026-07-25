"""Save / load trained TRM metric modules (regression / TD / contrastive)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .head import PairwiseMetricHead, QuasimetricHead
from .learners.contrastive import ContrastiveCritic


def save_metric(module: nn.Module, learner: str, latent_dim: int, arch: dict, path: str | Path) -> None:
    """Persist a metric module with the metadata needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"learner": learner, "latent_dim": latent_dim, "arch": arch,
         "state_dict": module.state_dict()},
        path,
    )


def build_metric(learner: str, latent_dim: int, arch: dict) -> nn.Module:
    """Construct an (untrained) metric module from architecture metadata."""
    if learner in ("regression", "td", "shuffled"):
        if arch.get("head") == "quasimetric":
            return QuasimetricHead(
                latent_dim,
                hidden_dim=arch.get("hidden_dim", 256),
                embed_dim=arch.get("embed_dim", 128),
                depth=arch.get("depth", 2),
            )
        return PairwiseMetricHead(
            latent_dim,
            hidden_dim=arch.get("hidden_dim", 256),
            depth=arch.get("depth", 2),
            softplus=arch.get("softplus", True),
            symmetric=arch.get("symmetric", False),
        )
    if learner == "contrastive":
        return ContrastiveCritic(
            latent_dim,
            hidden_dim=arch.get("hidden_dim", 256),
            rep_dim=arch.get("rep_dim", 64),
            depth=arch.get("depth", 2),
        )
    raise ValueError(f"unknown learner '{learner}'")


def load_metric(path: str | Path, device: str = "cpu") -> nn.Module:
    """Load a trained metric module saved by :func:`save_metric`."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    module = build_metric(blob["learner"], blob["latent_dim"], blob["arch"])
    module.load_state_dict(blob["state_dict"])
    return module.to(device).eval()


__all__ = ["save_metric", "build_metric", "load_metric"]
