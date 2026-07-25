"""Lightweight state/proprio world model.

A small JEPA-style latent world model over low-dimensional task state (e.g. the
TwoRoom agent position or OGBench proprio). It mirrors the public API of
``LeWM`` -- ``encode`` / ``predict`` / ``rollout`` / ``criterion`` / ``get_cost``
-- so it is a drop-in base for :class:`stable_worldmodel.trm.MetricCost`, the
``CEMSolver``, and the latent cache, but trains in minutes on CPU/MPS.

It is instantiable via Hydra (``_target_``) so it round-trips through the repo's
``save_pretrained`` / ``load_pretrained``. Crucially, even when the encoder is a
near-identity map (latent ~ position), the Euclidean terminal cost
``||z_hat_T - z_g||^2`` ignores the wall -- exactly the planner-facing metric
mismatch TRM repairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _mlp(in_dim, hidden, out_dim, depth=2, act=nn.SiLU):
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class _Predictor(nn.Module):
    """Per-timestep residual latent dynamics ``z_{t+1} = z_t + g(z_t, a_t)``."""

    def __init__(self, latent_dim, act_emb_dim, hidden_dim, depth=2):
        super().__init__()
        self.num_frames = 1  # rollout truncates history to the last frame
        self.net = _mlp(latent_dim + act_emb_dim, hidden_dim, latent_dim, depth)

    def forward(self, emb, act_emb):  # (B,T,D), (B,T,A) -> (B,T,D)
        delta = self.net(torch.cat([emb, act_emb], dim=-1))
        return emb + delta


class StateWM(nn.Module):
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
    ):
        super().__init__()
        self.obs_key = obs_key
        self.goal_key = goal_key
        self.state_scale = state_scale
        self.encoder = _mlp(state_dim, hidden_dim, latent_dim, encoder_depth)
        self.action_encoder = _mlp(action_dim, hidden_dim, act_emb_dim, 1)
        self.predictor = _Predictor(latent_dim, act_emb_dim, hidden_dim, predictor_depth)

    # ---- training-time API (mirrors LeWM) ----
    def encode(self, info: dict) -> dict:
        x = info[self.obs_key].float() / self.state_scale  # (B, T, S)
        b = x.size(0)
        x = rearrange(x, "b t ... -> (b t) ...")
        emb = self.encoder(x)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"].float())
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        return self.predictor(emb, act_emb)

    # ---- inference / planning API (mirrors LeWM) ----
    def rollout(self, info: dict, action_sequence: torch.Tensor, history_size: int | None = None) -> dict:
        if history_size is None:
            history_size = getattr(self.predictor, "num_frames", 1)
        assert self.obs_key in info, f"{self.obs_key} not in info_dict"
        H = info[self.obs_key].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        n_steps = T - H

        if "emb" not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info["emb"] = _init["emb"].detach().unsqueeze(1).expand(B, S, -1, -1)

        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_flat = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")
        all_act_emb = self.action_encoder(torch.cat([act_flat, act_future_flat], dim=1).float())

        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : H + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)
        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"]  # (B, S, T, D)
        goal_emb = info_dict["goal_emb"]       # (B, T, D)
        goal_emb = goal_emb[:, None, -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), reduction="none"
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)
        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        assert self.goal_key in info_dict, f"{self.goal_key} not in info_dict"
        if "goal_emb" not in info_dict:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal_obs = {self.obs_key: goal[self.goal_key]}
            info_dict["goal_emb"] = self.encode(goal_obs)["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


__all__ = ["StateWM"]
