"""Differentiable world-model unroll: the agent's way of imagining forward
through an externally-supplied, frozen world model.

Each function takes ``wm`` as a plain parameter and calls only ``wm.predict``/
``wm.action_encoder`` (or, for the DINO variant, ``wm.extra_encoders``) — they
own no weights of their own and work with any world model exposing that
interface, including the installed Stable World Model LeWM/PLDM backends. Used by :mod:`rlp.core.planner` training and
by :class:`rlp.core.solver.LIPSolver` at plan time.
"""

import torch

from .world_model.protocols import LatentWorldModel, TokenWorldModel

__all__ = [
    "rollout_terminal",
    "rollout_traj",
    "rollout_terminal_dino",
]


def rollout_terminal(
    wm: LatentWorldModel, z_hist: torch.Tensor, a_hist: torch.Tensor, plan: torch.Tensor
) -> torch.Tensor:
    """Pooled-latent WM: autoregressive H-block unroll; returns terminal latent.

    z_hist: (B, 3, D) latent history, a_hist: (B, 2, a) action-block history,
    plan: (B, H, a). Differentiable w.r.t. ``plan``.
    """
    return rollout_traj(wm, z_hist, a_hist, plan)[:, -1]


def rollout_traj(wm: LatentWorldModel, z_hist: torch.Tensor, a_hist: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
    """Like :func:`rollout_terminal` but returns all H imagined latents (B, H, D)."""
    embs = list(z_hist.unbind(dim=1))
    acts = list(a_hist.unbind(dim=1))
    outs: list[torch.Tensor] = []
    for t in range(plan.shape[1]):
        acts.append(plan[:, t])
        win_e = torch.stack(embs[-3:], dim=1)
        win_a = torch.stack(acts[-3:], dim=1)
        nxt = wm.predict(win_e, torch.as_tensor(wm.action_encoder(win_a)))[:, -1]
        embs.append(nxt)
        outs.append(nxt)
    return torch.stack(outs, dim=1)


def rollout_terminal_dino(
    wm: TokenWorldModel,
    toks_hist: torch.Tensor,
    plan: torch.Tensor,
    pix_dim: int = 384,
    act_emb_dim: int = 10,
) -> torch.Tensor:
    """Patch-token WM (PreJEPA): rollout in token space.

    toks_hist: (B, 3, P, D) tokens = pixel patches ++ tiled proprio-emb ++
    tiled action-emb. Each step injects the plan block's action embedding into
    the last frame's action slice, predicts the next frame's tokens, appends.
    Returns the terminal frame's mean-pooled pixel embedding (B, pix_dim) —
    the space the value function was trained on. Differentiable w.r.t. ``plan``
    (raw, denormalized action units).
    """
    toks = list(toks_hist.unbind(1))
    act_enc = wm.extra_encoders["action"]
    for t in range(plan.shape[1]):
        a_emb = act_enc(plan[:, t].unsqueeze(1))[:, 0]
        last = toks[-1]
        last = torch.cat(
            [
                last[..., :-act_emb_dim],
                a_emb.unsqueeze(1).expand(-1, last.shape[1], -1),
            ],
            dim=-1,
        )
        toks[-1] = last
        win = torch.stack(toks[-3:], dim=1)
        nxt = wm.predict(win)[:, -1]
        toks.append(nxt)
    return toks[-1][..., :pix_dim].mean(dim=1)
