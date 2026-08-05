"""DinoWMTokens: the original DINO-WM (PreJEPA-trained) exposed through the
pooled-latent WM API with the FULL patch-token field as the latent.

Motivation: the released TwoRoom DINO-WM bases are PreJEPA checkpoints — a
frozen DINOv2-small over 196 patches, each token = [pixel_emb(384) |
proprio_emb(10) | action_emb(10)], with a frame-causal ViT predictor over the
(T*P) token sequence. The prior cube LIP-on-DINO run pooled the pixel tokens
to 384-d for the value; per directive this class does NO pooling anywhere:

    emb per frame = flatten(P x [pixel_emb | proprio_emb])   (196*394 = 77,224-d
    with proprio; 196*384 = 75,264-d without)

The action slice is NOT part of the state latent — it is injected per frame at
``predict`` time from the action embedding, exactly like the LeWM/PLDM
``predict(emb, act_emb)`` contract, so ``rollout_traj``/``rollout_terminal``,
``train_lip_ac.py``, ``cache_latents.py`` and ``LIPSolver`` (kind='lip4') all
work unchanged on the flat latent.

Normalization contract (matches PreJEPA training, scripts/train/prejepa.py):
  - pixels: ImageNet-normalized floats at any size; internally resized to
    ``img_size`` (196 for the TwoRoom bases: predictor pos-emb is 3*196).
  - proprio: RAW env units in ``info['proprio']``; z-scored inside with the
    ``pro_mu/pro_std`` buffers (training-dataset stats, baked at conversion).
  - actions: z-scored action BLOCKS (fs*a_dim,), the train_lip_ac/eval
    convention. ``act_mu/act_std`` buffers hold the block stats the caller's
    z-scoring is assumed to use; ``action_encoder`` re-maps caller units ->
    training units internally ((a*std+mu - train_mu)/train_std collapses to
    identity when caller stats == training stats, the TwoRoom case).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def _required_int(value: object, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


class _BlockActionEncoder(nn.Module):
    ext_mu: torch.Tensor
    ext_std: torch.Tensor
    train_mu: torch.Tensor
    train_std: torch.Tensor

    """Caller-units -> training-units -> frozen PreJEPA action Embedder.

    Caller feeds z-scored blocks (B, T, A). Buffers hold the caller stats
    (``ext_*``, what the caller z-scored with) and the training stats
    (``train_*``). With identical stats the affine re-map is the identity.
    """

    def __init__(self, embedder: nn.Module, a_dim: int) -> None:
        super().__init__()
        self.embedder = embedder
        self.register_buffer("ext_mu", torch.zeros(a_dim))
        self.register_buffer("ext_std", torch.ones(a_dim))
        self.register_buffer("train_mu", torch.zeros(a_dim))
        self.register_buffer("train_std", torch.ones(a_dim))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        raw = a * self.ext_std + self.ext_mu
        return torch.as_tensor(self.embedder((raw - self.train_mu) / self.train_std))


class DinoWMTokens(nn.Module):
    pro_mu: torch.Tensor
    pro_std: torch.Tensor

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        proprio_encoder: nn.Module | None = None,
        img_size: int = 196,
        action_dim: int = 10,
        proprio_dim: int = 2,
        history_size: int = 3,
        interpolate_pos_encoding: bool = True,
        obs_key: str = "pixels",
        predict_bf16: bool = False,
        grad_checkpoint: bool = False,
        compile_predictor: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = encoder
        self.backbone.requires_grad_(False)
        self.predict_bf16 = predict_bf16
        self.grad_checkpoint = grad_checkpoint
        self.compile_predictor = compile_predictor
        self._pred_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.predictor = predictor
        self.action_encoder = _BlockActionEncoder(action_encoder, action_dim)
        self.proprio_encoder = proprio_encoder
        self.img_size = img_size
        self.history_size = history_size
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.obs_key = obs_key
        num_patches = _required_int(getattr(predictor, "num_patches", None), "predictor.num_patches")
        config = getattr(self.backbone, "config", None)
        hidden_size = _required_int(getattr(config, "hidden_size", None), "backbone.config.hidden_size")
        act_emb_dim = _required_int(getattr(action_encoder, "emb_dim", None), "action_encoder.emb_dim")
        pro_emb_dim = (
            _required_int(getattr(proprio_encoder, "emb_dim", None), "proprio_encoder.emb_dim")
            if proprio_encoder is not None
            else 0
        )
        self.num_patches = num_patches
        self.pix_dim = hidden_size
        self.act_emb_dim = act_emb_dim
        self.pro_emb_dim = pro_emb_dim
        # state token dim (pixel [+ proprio]); action slice lives only inside predict()
        self.tok_dim = self.pix_dim + self.pro_emb_dim
        self.latent_dim = self.num_patches * self.tok_dim
        if proprio_encoder is not None:
            self.register_buffer("pro_mu", torch.zeros(proprio_dim))
            self.register_buffer("pro_std", torch.ones(proprio_dim))

    @property
    def wants_proprio(self) -> bool:
        return self.proprio_encoder is not None

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _pixel_tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) ImageNet-normalized -> (N, P, pix_dim) patch tokens."""
        if pixels.shape[-1] != self.img_size or pixels.shape[-2] != self.img_size:
            pixels = F.interpolate(
                pixels,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                antialias=True,
                align_corners=False,
            )
        pixels = pixels.to(next(self.backbone.parameters()).dtype)
        # transformers >=5 plumbs interpolate_pos_encoding through forward();
        # 4.49's Dinov2 auto-interpolates and rejects the kwarg. Pass it only
        # when the backbone accepts it — the result is identical either way.
        kwargs: dict[str, Any] = {}
        if self.interpolate_pos_encoding:
            import inspect

            if "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters:
                kwargs["interpolate_pos_encoding"] = True
        h = self.backbone(pixels, **kwargs).last_hidden_state
        return torch.as_tensor(h[:, 1:, :])

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = info[self.obs_key].float()  # (B, T, C, H, W)
        b, t = x.shape[:2]
        toks = self._pixel_tokens(rearrange(x, "b t ... -> (b t) ..."))  # (BT, P, D)
        if self.wants_proprio:
            if "proprio" not in info:
                raise KeyError(
                    "DinoWMTokens (proprio variant): encode() needs info['proprio'] (raw env units) alongside pixels"
                )
            pro = info["proprio"].float().reshape(b, t, -1)
            pro = (pro - self.pro_mu) / self.pro_std
            if self.proprio_encoder is None:
                raise RuntimeError("proprio encoder is unavailable")
            pemb = torch.as_tensor(self.proprio_encoder(pro))
            pemb = pemb.reshape(b * t, 1, self.pro_emb_dim).expand(-1, self.num_patches, -1)
            toks = torch.cat([toks, pemb], dim=-1)  # (BT, P, pix+pro)
        info["emb"] = rearrange(
            toks.reshape(b * t, -1), "(b t) d -> b t d", b=b
        ).contiguous()  # flat (B, T, P*tok_dim) — all tokens, no pooling
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"].float())
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """(B, T, P*tok_dim) flat state latents + (B, T, act_emb) -> same-shape
        predictions; position t predicts frame t+1 (frame-causal predictor)."""
        B, T, _ = emb.shape
        toks = emb.reshape(B, T, self.num_patches, self.tok_dim)
        a = act_emb.unsqueeze(2).expand(-1, -1, self.num_patches, -1)  # tile per patch
        full = torch.cat([toks, a], dim=-1)  # (B, T, P, tok+act)
        x = rearrange(full, "b t p d -> b (t p) d")
        if self._pred_fn is None:
            self._pred_fn = torch.compile(self.predictor) if self.compile_predictor else self.predictor
        fn = self._pred_fn
        assert fn is not None
        # the K-step LIP unroll keeps every iteration's rollout graph alive for
        # the final backward — full fp32 activations OOM an 80GB H100 at B=128,
        # so recompute predictor activations in backward instead of storing them
        use_ckpt = self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad
        if self.predict_bf16 and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False) if use_ckpt else fn(x)
            out = out.float()
        elif use_ckpt:
            out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        else:
            out = fn(x)
        out = rearrange(out, "b (t p) d -> b t p d", t=T)
        out = out[..., : self.tok_dim]  # strip predicted action slice
        return out.reshape(B, T, self.latent_dim)


__all__ = ["DinoWMTokens"]
