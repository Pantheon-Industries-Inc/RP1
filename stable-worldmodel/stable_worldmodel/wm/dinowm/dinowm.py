"""DINO-WM: a world model with a frozen DINOv2 latent + learned latent dynamics.

A pragmatic, MetricCost-compatible instantiation of DINO-WM: the encoder is a
*frozen* pretrained DINOv2 (mean-pooled patch tokens -> a single feature vector),
and a small action-conditioned predictor learns the latent dynamics. Because the
latent is a single vector (like ``LeWM``'s CLS token), it plugs directly into
:class:`stable_worldmodel.trm.MetricCost`, the latent cache, the metric learners,
and the CEM/Predictive-Sampling MPC -- no patch-level adapter needed.

This is the *non-position pixel latent* regime the TRM paper targets: Euclidean
distance in DINOv2 space is not a good proxy for reachability, so the raw-latent
terminal cost mis-ranks plans and TRM repairs it.

It subclasses :class:`StateWM` and only overrides construction + ``encode``;
``predict`` / ``rollout`` / ``criterion`` / ``get_cost`` are inherited unchanged.
"""

from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from ..statewm.statewm import StateWM, _Predictor, _mlp


class DinoWM(StateWM):
    def __init__(
        self,
        backbone_name: str = "dinov2_small",
        action_dim: int = 2,
        latent_dim: int | None = None,   # None -> use DINOv2 hidden size (no projection)
        hidden_dim: int = 512,
        act_emb_dim: int = 64,
        predictor_depth: int = 2,
        obs_key: str = "pixels",
        goal_key: str = "goal",
        interpolate_pos_encoding: bool = True,
    ):
        nn.Module.__init__(self)  # bypass StateWM.__init__ (different encoder)
        from stable_worldmodel.wm.prejepa.module import create_backbone

        self.obs_key = obs_key
        self.goal_key = goal_key
        self.state_scale = 1.0
        self.interpolate_pos_encoding = interpolate_pos_encoding

        self.backbone = create_backbone(backbone_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        dino_dim = int(self.backbone.config.hidden_size)
        self.dino_dim = dino_dim
        latent_dim = dino_dim if latent_dim is None else latent_dim
        self.latent_dim = latent_dim

        self.proj = nn.Identity() if latent_dim == dino_dim else nn.Linear(dino_dim, latent_dim)
        self.action_encoder = _mlp(action_dim, hidden_dim, act_emb_dim, 1)
        self.predictor = _Predictor(latent_dim, act_emb_dim, hidden_dim, predictor_depth)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep DINOv2 frozen / in eval (BN, dropout)
        return self

    @torch.no_grad()
    def _encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W) -> (N, dino_dim)`` mean-pooled DINOv2 patch features."""
        pixels = pixels.to(next(self.backbone.parameters()).dtype)
        kwargs = {"interpolate_pos_encoding": True} if self.interpolate_pos_encoding else {}
        out = self.backbone(pixels, **kwargs)
        h = out.last_hidden_state  # (N, 1 + P, D)
        return h[:, 1:, :].mean(dim=1)  # drop CLS, mean-pool patches

    def encode(self, info: dict) -> dict:
        x = info[self.obs_key].float()  # (B, T, C, H, W)
        b = x.size(0)
        x = rearrange(x, "b t ... -> (b t) ...")
        feat = self._encode_pixels(x)            # (B*T, dino_dim), no grad
        emb = self.proj(feat)                    # trainable projection (or identity)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"].float())
        return info


__all__ = ["DinoWM"]
