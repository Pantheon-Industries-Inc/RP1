"""Lightweight state/proprio world model.

A small JEPA-style latent world model over low-dimensional task state (e.g. the
TwoRoom agent position or OGBench proprio). It mirrors the public API of
``LeWM`` -- ``encode`` / ``predict`` / ``rollout`` / ``criterion`` / ``get_cost``
-- so it is a drop-in base for :class:`rlp.core.value.MetricCost`, the
``CEMSolver``, and the latent cache, but trains in minutes on CPU/MPS.

It is instantiable via Hydra (``_target_``) so it round-trips through the repo's
``save_pretrained`` / ``load_pretrained``. Crucially, even when the encoder is a
near-identity map (latent ~ position), the Euclidean terminal cost
``||z_hat_T - z_g||^2`` ignores the wall -- exactly the planner-facing metric
mismatch TRM repairs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from einops import rearrange
from stable_worldmodel.wm import LeWM
from torch import nn


def _mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    depth: int = 2,
    act: Callable[[], nn.Module] = nn.SiLU,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class _Predictor(nn.Module):
    """Per-timestep residual latent dynamics ``z_{t+1} = z_t + g(z_t, a_t)``."""

    def __init__(self, latent_dim: int, act_emb_dim: int, hidden_dim: int, depth: int = 2) -> None:
        super().__init__()
        self.num_frames = 1  # rollout truncates history to the last frame
        self.net = _mlp(latent_dim + act_emb_dim, hidden_dim, latent_dim, depth)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        delta = torch.as_tensor(self.net(torch.cat([emb, act_emb], dim=-1)))
        return emb + delta


class StateWM(LeWM):
    """LeWM planning machinery with low-dimensional encoders and goal keys."""

    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 2,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        act_emb_dim: int = 32,
        encoder_depth: int = 2,
        predictor_depth: int = 2,
        obs_key: str = "state",
        goal_key: str = "goal_state",
        state_scale: float = 1.0,
    ) -> None:
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.state_scale = state_scale
        super().__init__(
            encoder=_mlp(state_dim, hidden_dim, latent_dim, encoder_depth),
            action_encoder=_mlp(action_dim, hidden_dim, act_emb_dim, 1),
            predictor=_Predictor(latent_dim, act_emb_dim, hidden_dim, predictor_depth),
        )

    # ---- training-time API (mirrors LeWM) ----
    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = info[self.obs_key].float() / self.state_scale  # (B, T, S)
        b = x.size(0)
        x = rearrange(x, "b t ... -> (b t) ...")
        emb = torch.as_tensor(self.encoder(x))
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"].float())
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.predictor(emb, act_emb))

    # ---- inference / planning API (mirrors LeWM) ----
    def rollout(
        self,
        info: dict[str, torch.Tensor],
        action_sequence: torch.Tensor,
        history_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.obs_key not in info:
            raise KeyError(f"{self.obs_key} not in info_dict")

        # Stable-WM's LeWM rollout is observation-agnostic apart from the
        # conventional ``pixels`` key it uses to determine history length.
        # Alias our state key for that call instead of maintaining a fork of
        # its autoregressive rollout implementation.
        had_pixels = "pixels" in info
        original_pixels = info.get("pixels")
        info["pixels"] = info[self.obs_key]
        try:
            return cast(
                dict[str, torch.Tensor],
                super().rollout(info, action_sequence, history_size),
            )
        finally:
            if not had_pixels:
                info.pop("pixels", None)
            else:
                assert original_pixels is not None
                info["pixels"] = original_pixels

    def get_cost(self, info_dict: dict[str, torch.Tensor], action_candidates: torch.Tensor) -> torch.Tensor:
        assert self.goal_key in info_dict, f"{self.goal_key} not in info_dict"
        if "goal_emb" not in info_dict:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal_obs = {self.obs_key: goal[self.goal_key]}
            # LeWM's criterion expects an explicit candidate axis. Keeping it
            # singleton lets Stable-WM broadcast the goal over all plans.
            info_dict["goal_emb"] = self.encode(goal_obs)["emb"].unsqueeze(1)
        info_dict = self.rollout(info_dict, action_candidates)
        return cast(torch.Tensor, self.criterion(info_dict))


__all__ = ["StateWM"]
