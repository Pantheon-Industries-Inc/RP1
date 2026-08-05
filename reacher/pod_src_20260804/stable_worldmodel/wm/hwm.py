"""HWM high level for LeWM: action encoder A_psi + high-level predictor F2.

Port of "Hierarchical Planning with Latent World Models" (arXiv 2604.03208)
onto the LeWM pooled-latent stack. The HWM wrapper exposes the SAME
``predict`` / ``action_encoder`` API as :class:`stable_worldmodel.wm.lewm.LeWM`,
so :func:`stable_worldmodel.solver.lip.rollout_traj` (and the whole LIP
machinery) runs at the high level unchanged — one "action" is a latent
macro-action summarizing ``stride`` primitive env steps.

Class definitions must stay IDENTICAL to scripts/train_hwm.py (checkpoints are
plain state_dicts).
"""

import torch
from torch import nn

from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor


class PosteriorMLP(nn.Module):
    """Chunk of primitive actions -> deterministic macro-action (mu, LayerNorm'd)."""

    def __init__(self, chunk_len, act_dim, macro_dim, hidden=256):
        super().__init__()
        self.chunk_len, self.act_dim, self.macro_dim = chunk_len, act_dim, macro_dim
        self.net = nn.Sequential(
            nn.Linear(chunk_len * act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, macro_dim),
        )
        self.ln = nn.LayerNorm(macro_dim)

    def forward(self, chunk):                     # (B, chunk_len, act_dim)
        return self.ln(self.net(chunk.reshape(chunk.shape[0], -1)))


class PosteriorTF(nn.Module):
    """Transformer-CLS action encoder (paper's Franka/PushT A_psi)."""

    def __init__(self, chunk_len, act_dim, macro_dim, width=64, layers=2, heads=4):
        super().__init__()
        self.embed = nn.Linear(act_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.randn(1, chunk_len + 1, width) * 0.02)
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width,
                                         batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(width, macro_dim)
        self.ln = nn.LayerNorm(macro_dim)

    def forward(self, chunk):                     # (B, chunk_len, act_dim)
        x = self.embed(chunk)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        return self.ln(self.head(self.tf(x)[:, 0]))


class HWM(nn.Module):
    """High-level WM with the SAME predict/action_encoder API as LeWM.
    ``action_encoder`` consumes raw macro-actions (B, T, macro_dim)."""

    def __init__(self, macro_dim, z_dim=192, depth=6, heads=16, mlp_dim=2048,
                 dim_head=64, num_frames=3, dropout=0.1, ae="mlp",
                 chunk_len=25, act_dim=5, use_pred_proj=True):
        super().__init__()
        self.macro_dim, self.z_dim = macro_dim, z_dim
        self.posterior = (PosteriorMLP(chunk_len, act_dim, macro_dim) if ae == "mlp"
                          else PosteriorTF(chunk_len, act_dim, macro_dim))
        self.action_encoder = Embedder(input_dim=macro_dim,
                                       smoothed_dim=macro_dim, emb_dim=z_dim)
        self.predictor = Predictor(num_frames=num_frames, depth=depth, heads=heads,
                                   mlp_dim=mlp_dim, input_dim=z_dim, hidden_dim=z_dim,
                                   output_dim=z_dim, dim_head=dim_head, dropout=dropout)
        self.pred_proj = (MLP(z_dim, 2048, z_dim, norm_fn=nn.BatchNorm1d)
                          if use_pred_proj else nn.Identity())

    def predict(self, emb, act_emb):              # mirrors LeWM.predict
        preds = self.predictor(emb, act_emb)
        B, T, D = preds.shape
        return self.pred_proj(preds.reshape(B * T, D)).reshape(B, T, D)

    def encode_chunks(self, chunks):              # (B, T, chunk_len, act_dim) -> (B, T, macro)
        B, T = chunks.shape[:2]
        return self.posterior(chunks.reshape(B * T, *chunks.shape[2:])).reshape(B, T, -1)

    def rollout_from(self, z0, macros):
        """Solver-convention rollout: z-history = z0 x3, zero macro history.
        z0 (B, D), macros (B, H, macro_dim) raw -> (B, H, D)."""
        from stable_worldmodel.solver.lip import rollout_traj
        zh = z0.unsqueeze(1).expand(-1, 3, -1)
        ah = torch.zeros(z0.shape[0], 2, self.macro_dim,
                         device=z0.device, dtype=macros.dtype)
        return rollout_traj(self, zh, ah, macros)


def load_hwm(path, device="cpu"):
    """Load a train_hwm.py checkpoint -> (HWM in eval mode, blob dict)."""
    blob = torch.load(path, map_location=device, weights_only=False)
    assert blob.get("kind") == "hwm", f"not an HWM checkpoint: {path}"
    cfg = blob["cfg"]
    hwm = HWM(cfg["macro_dim"], z_dim=cfg["z_dim"], depth=cfg["depth"],
              dropout=cfg["dropout"], ae=cfg["ae"], chunk_len=cfg["chunk_len"],
              act_dim=cfg["act_dim"], use_pred_proj=cfg["use_pred_proj"]).to(device)
    hwm.load_state_dict(blob["sd"])
    hwm.eval()
    hwm.requires_grad_(False)
    return hwm, blob


__all__ = ["HWM", "PosteriorMLP", "PosteriorTF", "load_hwm"]
