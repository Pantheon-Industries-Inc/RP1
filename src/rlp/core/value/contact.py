"""Contact-consistency penalty on imagined trajectories (PushT).

The block is a passive body: it can only move while the agent touches it. The
frozen world model does not obey that constraint -- on failed RLP plans the
imagined block arrives ~50 px closer to the goal than the real one, with the
imagined agent accurate to 1 px (docs/campaigns/2026-09-03/PUSHT_DIAG.md, E16).
A planner that optimises any objective through such a rollout is scoring a
state that never happens.

This module turns the violation into a differentiable cost. A linear probe
decodes ``[agent xy, block xy, cos angle, sin angle]`` from a latent (held-out
R2 ~0.95 agent / ~0.97 block on the PushT training cache), so for an imagined
trajectory we can measure per step

    move  = |block pose change|            (px, angle scaled by ``angle_scale``)
    gap   = |agent - block centre| - contact_radius

and charge the conjunction "the block moved AND the agent was too far to have
moved it":

    penalty = mean_t relu(move_t - move_eps) * clamp((gap_t - gap_eps) / gap_norm, 0, 1)

Both factors are needed: block motion alone is legitimate, a distant agent
alone is legitimate, only their product is physically impossible. The gap
factor saturates, so the penalty is measured in *pixels of unsupported block
motion* and stays commensurable with the critic's energy; the caller's
``weight`` converts px into energy units. The thresholds sit above the probe's
own noise, so the term only fires on gross hallucination -- exactly the 30-50
px kind that separates the refiner from a sampler.

``contact_radius`` is not a geometry guess: :mod:`rlp.tools.data.fit_state_probe`
measures, over the expert transitions that actually move the block, how far the
agent centre ever is from the block centre, and stores that percentile in the
probe artifact. Passing ``contact_radius=None`` adopts it, so the penalty only
fires beyond a separation at which the data never shows block motion.

The penalty is one-sided by construction. The opposite world-model error,
imagining the block stays when it really moves, makes a plan look worse rather
than better, so it is not exploitable and is deliberately not penalised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from rlp.logging import logger


class StateProbe(nn.Module):
    """Affine latent -> ``[ax, ay, bx, by, cos, sin]`` decoder (ridge fit)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, meta: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.register_buffer("weight", weight.float())  # (D, 6)
        self.register_buffer("bias", bias.float())  # (6,)
        self.meta = meta or {}

    @property
    def latent_dim(self) -> int:
        return int(cast(torch.Tensor, self.weight).shape[0])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` (..., D) -> (..., 6). A windowed latent uses its newest frame."""
        d = self.latent_dim
        if z.shape[-1] != d:
            if z.shape[-1] % d:
                raise ValueError(f"probe dim {d} does not divide latent width {z.shape[-1]}")
            z = z[..., -d:]  # window layout is oldest-first, newest last
        return z.float() @ cast(torch.Tensor, self.weight) + cast(torch.Tensor, self.bias)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> StateProbe:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        probe = cls(blob["weight"], blob["bias"], blob.get("meta"))
        logger.info(f"State probe loaded from {path} (dim {probe.latent_dim}, meta {probe.meta})")
        return probe.to(device).eval()


class ContactPenalty(nn.Module):
    """Contact-consistency penalty over an imagined trajectory.

    Args:
        probe: latent -> state decoder.
        weight: cost weight (0 disables; the caller should skip construction).
        move_eps: block pose change below this is treated as no motion (px).
        gap_eps: agent-to-block gap below this is treated as possible contact (px).
        contact_radius: separation subtracted from the agent-block centre
            distance. ``None`` adopts the data-measured value in the probe's
            metadata (the percentile of separations at which the expert still
            moves the block).
        angle_scale: px per radian of block rotation when measuring pose change.
        gap_norm: px beyond ``gap_eps`` at which the gap factor saturates at 1,
            so the penalty reads as pixels of unsupported block motion.
    """

    def __init__(
        self,
        probe: StateProbe,
        weight: float = 1.0,
        move_eps: float = 15.0,
        gap_eps: float = 0.0,
        contact_radius: float | None = None,
        angle_scale: float = 40.0,
        gap_norm: float = 50.0,
    ) -> None:
        super().__init__()
        self.probe = probe
        self.weight = float(weight)
        self.move_eps = float(move_eps)
        self.gap_eps = float(gap_eps)
        if contact_radius is None:
            measured = probe.meta.get("contact_radius_p99")
            if measured is None:
                raise ValueError("contact_radius=None requires a probe fitted with the contact calibration")
            contact_radius = float(measured)
            logger.info(f"Contact radius taken from the probe's expert calibration: {contact_radius:.1f} px")
        self.contact_radius = float(contact_radius)
        self.angle_scale = float(angle_scale)
        self.gap_norm = float(gap_norm)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.probe(z))

    def terms(self, trajectory: torch.Tensor, start: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-step ``(move, gap)`` in px for ``trajectory`` (B, H, D)."""
        states = self.decode(trajectory)  # (B, H, 6)
        if start is not None:
            states = torch.cat([self.decode(start).unsqueeze(1), states], dim=1)
        block = states[..., 2:4]
        angle = torch.atan2(states[..., 5], states[..., 4])
        d_pos = (block[:, 1:] - block[:, :-1]).norm(dim=-1)
        d_ang = (angle[:, 1:] - angle[:, :-1] + torch.pi) % (2 * torch.pi) - torch.pi
        move = d_pos + self.angle_scale * d_ang.abs()
        # the agent must be close at BOTH ends of a step to have caused the motion
        centre = (states[..., :2] - states[..., 2:4]).norm(dim=-1) - self.contact_radius
        gap = torch.minimum(centre[:, 1:], centre[:, :-1])
        return move, gap

    def forward(self, trajectory: torch.Tensor, start: torch.Tensor | None = None) -> torch.Tensor:
        """Penalty per batch row for ``trajectory`` (B, H, D); 0 when consistent."""
        move, gap = self.terms(trajectory, start)
        gap_factor = ((gap - self.gap_eps) / self.gap_norm).clamp(0.0, 1.0)
        viol = torch.relu(move - self.move_eps) * gap_factor
        return self.weight * viol.mean(dim=-1)

    @torch.no_grad()
    def report(self, trajectory: torch.Tensor, start: torch.Tensor | None = None) -> dict[str, float]:
        """Diagnostics for one batch: unsupported px, fraction of steps firing."""
        move, gap = self.terms(trajectory, start)
        gap_factor = ((gap - self.gap_eps) / self.gap_norm).clamp(0.0, 1.0)
        viol = torch.relu(move - self.move_eps) * gap_factor
        return {
            "unsupported_px_mean": float(viol.mean()),
            "unsupported_px_max": float(viol.max()),
            "fraction_steps_firing": float((viol > 1.0).float().mean()),
            "move_px_mean": float(move.mean()),
            "gap_px_mean": float(gap.mean()),
        }

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, device: str | torch.device = "cpu") -> ContactPenalty | None:
        """Build from a mapping with ``probe`` and optional overrides; None when off."""
        if not config:
            return None
        weight = float(config.get("weight", 0.0) or 0.0)
        path = config.get("probe")
        if weight <= 0 or not path:
            return None
        keys = ("move_eps", "gap_eps", "angle_scale", "gap_norm")
        kwargs: dict[str, Any] = {k: float(config[k]) for k in keys if config.get(k) is not None}
        kwargs["contact_radius"] = None if config.get("contact_radius") is None else float(config["contact_radius"])
        penalty = cls(StateProbe.load(str(path), device=device), weight=weight, **kwargs)
        logger.info(
            f"Contact-consistency penalty on: weight={penalty.weight} move_eps={penalty.move_eps} "
            f"gap_eps={penalty.gap_eps} contact_radius={penalty.contact_radius} angle_scale={penalty.angle_scale}"
        )
        return penalty.to(device)


__all__ = ["ContactPenalty", "StateProbe"]
