"""Structural contracts for the world-model operations used by RLP."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class ActionEncoder(Protocol):
    def __call__(self, actions: torch.Tensor) -> torch.Tensor: ...


class LatentWorldModel(Protocol):
    @property
    def action_encoder(self) -> nn.Module: ...

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor: ...


class TokenWorldModel(Protocol):
    extra_encoders: dict[str, nn.Module]

    def predict(self, emb: torch.Tensor) -> torch.Tensor: ...


__all__ = ["ActionEncoder", "LatentWorldModel", "TokenWorldModel"]
