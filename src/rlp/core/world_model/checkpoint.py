"""Checkpoint helpers for RLP models built on the published Stable-WM wheel."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf
from stable_worldmodel.wm.utils import load_pretrained as _load_pretrained
from stable_worldmodel.wm.utils import save_pretrained as _save_pretrained
from torch import nn


def load_pretrained(
    name: str | Path,
    cache_dir: str | None = None,
    extra_args: Any | None = None,
) -> nn.Module:
    """Load a local checkpoint before falling back to Stable-WM resolution.

    Stable-WM 0.1.1 resolves every relative name below its global checkpoint
    cache.  Repository configs intentionally use checkout-relative paths, so
    turn an existing local path into an absolute path before delegating.  A
    missing path is left untouched because it may be a Hugging Face repo ID.
    """
    path = Path(name).expanduser()
    resolved_name = str(path.resolve()) if path.exists() else str(name)
    return cast(nn.Module, _load_pretrained(resolved_name, cache_dir=cache_dir, extra_args=extra_args))


def save_pretrained(model: nn.Module, run_name: str, config: Any | None = None, **kwargs: Any) -> None:
    """Accept both plain mappings and OmegaConf configs.

    Stable-WM 0.1.1 unconditionally calls ``OmegaConf.to_container`` and
    therefore rejects the resolved dictionaries produced by several RLP
    trainers. Converting mappings back to an OmegaConf object preserves its
    normal checkpoint layout without patching site-packages.
    """
    if config is not None and not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    _save_pretrained(model, run_name=run_name, config=config, **kwargs)


__all__ = ["load_pretrained", "save_pretrained"]
