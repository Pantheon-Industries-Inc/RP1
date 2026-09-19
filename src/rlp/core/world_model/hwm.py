"""High-level (macro-action) world model for hierarchical planning.

Port of the 2026-07 HWM campaign's high level (`train_hwm.py`, itself a port of
arXiv 2604.03208 to the LeWM stack) into this repository, because the original
lives in a pasted Stable-WM source tree whose ``stable_worldmodel.trm`` module
does not exist in the published wheel.

Two components:

* ``PosteriorMLP`` / ``PosteriorTF`` — ``A_psi``: a chunk of ``chunk_len``
  primitive actions -> one ``macro_dim`` macro-action (LayerNorm'd). The MLP is
  the paper's ``PosteriorContinuous``; the transformer-CLS variant is the one
  they use on Franka/PushT.
* ``HWM`` — ``F2``: a clone of the LeWM predictor conditioned on an
  ``Embedder`` of the macro-action. It deliberately exposes the same
  ``predict`` / ``action_encoder`` surface as the low-level WM, so
  :func:`rlp.core.rollout.rollout_traj` and the planner code work at the high
  level unchanged.

Why fixed stride matters here: with ``stride=25`` and ``horizon=8`` a plan spans
200 primitive steps, which is what an h200 cube task needs (episodes are 201
steps). Hi-LeWM's released cube high level cannot reach that — its waypoints cap
at 5 x ``max_span`` 15 = 75 primitive steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
from torch import nn

from rlp.core.rollout import rollout_traj

__all__ = ["HWM", "PosteriorMLP", "PosteriorTF", "load_hwm", "save_hwm"]


class PosteriorMLP(nn.Module):
    """Primitive-action chunk -> deterministic macro-action (LayerNorm'd mu)."""

    def __init__(self, chunk_len: int, act_dim: int, macro_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.chunk_len, self.act_dim, self.macro_dim = chunk_len, act_dim, macro_dim
        self.net = nn.Sequential(
            nn.Linear(chunk_len * act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, macro_dim),
        )
        self.ln = nn.LayerNorm(macro_dim)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """``(B, chunk_len, act_dim) -> (B, macro_dim)``."""
        return cast(torch.Tensor, self.ln(self.net(chunk.reshape(chunk.shape[0], -1))))


class PosteriorTF(nn.Module):
    """Transformer-CLS action encoder (the paper's Franka/PushT ``A_psi``)."""

    def __init__(
        self,
        chunk_len: int,
        act_dim: int,
        macro_dim: int,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(act_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, chunk_len + 1, width) * 0.02)
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(width, macro_dim)
        self.ln = nn.LayerNorm(macro_dim)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        x = self.embed(chunk)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        return cast(torch.Tensor, self.ln(self.head(self.tf(x)[:, 0])))


class HWM(nn.Module):
    """High-level world model over macro-actions.

    Exposes ``predict`` / ``action_encoder`` so it satisfies
    :class:`rlp.core.world_model.protocols.LatentWorldModel` and can be unrolled
    by :func:`rlp.core.rollout.rollout_traj` exactly like the low-level WM.
    """

    def __init__(
        self,
        macro_dim: int,
        z_dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        mlp_dim: int = 2048,
        dim_head: int = 64,
        num_frames: int = 3,
        dropout: float = 0.1,
        ae: str = "mlp",
        chunk_len: int = 25,
        act_dim: int = 5,
        use_pred_proj: bool = True,
    ) -> None:
        super().__init__()
        self.macro_dim, self.z_dim = macro_dim, z_dim
        self.chunk_len, self.act_dim = chunk_len, act_dim
        self.posterior: nn.Module = (
            PosteriorMLP(chunk_len, act_dim, macro_dim)
            if ae == "mlp"
            else PosteriorTF(chunk_len, act_dim, macro_dim)
        )
        self.action_encoder = Embedder(input_dim=macro_dim, smoothed_dim=macro_dim, emb_dim=z_dim)
        self.predictor = Predictor(
            num_frames=num_frames,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            input_dim=z_dim,
            hidden_dim=z_dim,
            output_dim=z_dim,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.pred_proj: nn.Module = (
            MLP(z_dim, 2048, z_dim, norm_fn=nn.BatchNorm1d) if use_pred_proj else nn.Identity()
        )

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """Mirrors ``LeWM.predict``: ``(B, T, D), (B, T, D) -> (B, T, D)``."""
        preds = self.predictor(emb, act_emb)
        b, t, d = preds.shape
        return cast(torch.Tensor, self.pred_proj(preds.reshape(b * t, d)).reshape(b, t, d))

    def encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """``(B, T, chunk_len, act_dim) -> (B, T, macro_dim)``."""
        b, t = chunks.shape[:2]
        return cast(torch.Tensor, self.posterior(chunks.reshape(b * t, *chunks.shape[2:])).reshape(b, t, -1))

    def rollout_from(self, z0: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
        """Solver-convention rollout: z-history = ``z0`` x3, zero macro history.

        The deployed solver is stateless — it has no real waypoint history at
        plan time — so training and deployment must both use this convention or
        the high level is evaluated off-distribution.
        """
        z_hist = z0.unsqueeze(1).expand(-1, 3, -1)
        a_hist = torch.zeros(z0.shape[0], 2, self.macro_dim, device=z0.device, dtype=macros.dtype)
        return rollout_traj(self, z_hist, a_hist, macros)


def save_hwm(model: HWM, path: str | Path, cfg: dict[str, Any], stats: dict[str, Any]) -> Path:
    """Persist weights plus the config//stats a solver needs to rebuild it."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "hwm", "sd": model.state_dict(), "cfg": cfg, **stats}, out)
    return out


def load_hwm(path: str | Path, device: str = "cpu") -> tuple[HWM, dict[str, Any]]:
    """Load a checkpoint written by :func:`save_hwm`; returns ``(model, blob)``."""
    blob = torch.load(str(path), map_location=device, weights_only=False)
    if blob.get("kind") != "hwm":
        raise ValueError(f"{path}: not an HWM checkpoint (kind={blob.get('kind')!r})")
    cfg = blob["cfg"]
    model = HWM(
        macro_dim=cfg["macro_dim"],
        z_dim=cfg.get("z_dim", 192),
        depth=cfg.get("depth", 6),
        heads=cfg.get("heads", 16),
        mlp_dim=cfg.get("mlp_dim", 2048),
        dim_head=cfg.get("dim_head", 64),
        num_frames=cfg.get("num_frames", 3),
        dropout=cfg.get("dropout", 0.1),
        ae=cfg.get("ae", "mlp"),
        chunk_len=cfg.get("stride", 25),
        act_dim=cfg["act_dim"],
        use_pred_proj=cfg.get("use_pred_proj", True),
    ).to(device)
    model.load_state_dict(blob["sd"])
    model.eval()
    return model, blob
