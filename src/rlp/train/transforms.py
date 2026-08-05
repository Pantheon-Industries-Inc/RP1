"""Shared nested Stable-Pretraining transforms."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import stable_pretraining as spt
import torch
from torchvision.transforms import v2


def nested_transform(transform: Callable[[Any], Any], source: str, target: str) -> Any:
    """Apply a Torch transform to one nested sample field."""
    factory = cast(Callable[..., Any], spt.data.transforms.WrapTorchTransform)
    return factory(transform, source=source, target=target)


def nested_resize(size: int, source: str = "image", target: str = "image") -> Any:
    """Resize one nested image field."""
    return nested_transform(v2.Resize(size), source, target)


def nested_clip(bound: float, source: str, target: str) -> Any:
    """Clamp one nested numeric field to a symmetric bound."""
    limit = float(bound)
    return nested_transform(lambda value: torch.as_tensor(value).clamp(-limit, limit), source, target)


def image_preprocessor(source: str, target: str, image_size: int = 224) -> Any:
    """Build the canonical ImageNet conversion and resize pipeline."""
    stats = spt.data.dataset_stats.ImageNet
    compose = cast(Callable[..., Any], spt.data.transforms.Compose)
    to_image = cast(Callable[..., Any], spt.data.transforms.ToImage)
    return compose(
        to_image(**stats, source=source, target=target),
        nested_resize(image_size, source=source, target=target),
    )


__all__ = ["image_preprocessor", "nested_clip", "nested_resize", "nested_transform"]
