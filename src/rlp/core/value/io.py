"""Portable Stable-WM artifacts for trained value metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from rlp.core.world_model import load_pretrained, save_pretrained

from .head import IQEHead, PairwiseMetricHead, QuasimetricHead
from .learners.contrastive import ContrastiveCritic


@runtime_checkable
class PretrainedMetric(Protocol):
    """Inference and serialization contract implemented by metric modules."""

    latent_dim: int

    def cost(self, z_pred: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor: ...

    def pretrained_config(self) -> dict[str, object]: ...


def save_metric(module: nn.Module, *, run_name: str, cache_dir: str | Path) -> Path:
    """Save a metric using Stable-WM's weights plus Hydra config layout."""
    if not isinstance(module, PretrainedMetric):
        raise TypeError(f"{type(module).__name__} does not implement the pretrained metric contract")
    cache_dir = Path(cache_dir)
    save_pretrained(module, run_name=run_name, config=module.pretrained_config(), cache_dir=str(cache_dir))
    return cache_dir.resolve() / "checkpoints" / run_name


def _int_setting(arch: Mapping[str, object], name: str, default: int) -> int:
    value = arch.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"metric architecture {name!r} must be an integer")
    return value


def _float_setting(arch: Mapping[str, object], name: str, default: float) -> float:
    value = arch.get(name, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"metric architecture {name!r} must be numeric")
    return float(value)


def _bool_setting(arch: Mapping[str, object], name: str, default: bool) -> bool:
    value = arch.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"metric architecture {name!r} must be boolean")
    return value


def build_metric(learner: str, latent_dim: int, arch: Mapping[str, object]) -> nn.Module:
    """Construct an untrained metric module from training architecture settings."""
    if learner in ("regression", "td", "shuffled"):
        if arch.get("head") == "iqe":
            return IQEHead(
                latent_dim,
                hidden_dim=_int_setting(arch, "hidden_dim", 256),
                embed_dim=_int_setting(arch, "embed_dim", 128),
                depth=_int_setting(arch, "depth", 2),
                num_components=_int_setting(arch, "num_components", 8),
            )
        if arch.get("head") == "quasimetric":
            return QuasimetricHead(
                latent_dim,
                hidden_dim=_int_setting(arch, "hidden_dim", 256),
                embed_dim=_int_setting(arch, "embed_dim", 128),
                depth=_int_setting(arch, "depth", 2),
            )
        return PairwiseMetricHead(
            latent_dim,
            hidden_dim=_int_setting(arch, "hidden_dim", 256),
            depth=_int_setting(arch, "depth", 2),
            softplus=_bool_setting(arch, "softplus", True),
            symmetric=_bool_setting(arch, "symmetric", False),
            scale=_float_setting(arch, "scale", 1.0),
        )
    if learner == "contrastive":
        return ContrastiveCritic(
            latent_dim,
            hidden_dim=_int_setting(arch, "hidden_dim", 256),
            rep_dim=_int_setting(arch, "rep_dim", 64),
            depth=_int_setting(arch, "depth", 2),
        )
    raise ValueError(f"unknown learner '{learner}'")


def load_metric(name: str | Path, device: str = "cpu", cache_dir: str | Path | None = None) -> nn.Module:
    """Load and validate a Stable-WM metric artifact."""
    path = Path(name).expanduser()
    if path.is_file() and not (path.parent / "config.json").is_file():
        raise ValueError(f"Unsupported metric checkpoint {path}; expected a Stable-WM artifact with config.json")
    if cache_dir is None and path.exists():
        resolved = path.resolve()
        checkpoint_root = next(
            (parent for parent in (resolved, *resolved.parents) if parent.name == "checkpoints"), None
        )
        if checkpoint_root is not None:
            cache_dir = checkpoint_root.parent
    try:
        module = load_pretrained(str(name), cache_dir=None if cache_dir is None else str(cache_dir))
    except Exception as error:
        if path.is_file():
            raise ValueError(
                f"Unsupported metric checkpoint {path}; expected a Stable-WM artifact with config.json"
            ) from error
        raise
    if not isinstance(module, nn.Module) or not callable(getattr(module, "cost", None)):
        raise TypeError(f"checkpoint {name!s} does not contain a value metric")
    return module.to(device).eval()


__all__ = ["PretrainedMetric", "build_metric", "load_metric", "save_metric"]
