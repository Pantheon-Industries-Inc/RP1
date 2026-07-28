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

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class _BlockActionEncoder(nn.Module):
    """Caller-units -> training-units -> frozen PreJEPA action Embedder.

    Caller feeds z-scored blocks (B, T, A). Buffers hold the caller stats
    (``ext_*``, what the caller z-scored with) and the training stats
    (``train_*``). With identical stats the affine re-map is the identity.
    """

    def __init__(self, embedder: nn.Module, a_dim: int):
        super().__init__()
        self.embedder = embedder
        self.register_buffer("ext_mu", torch.zeros(a_dim))
        self.register_buffer("ext_std", torch.ones(a_dim))
        self.register_buffer("train_mu", torch.zeros(a_dim))
        self.register_buffer("train_std", torch.ones(a_dim))

    def forward(self, a):
        raw = a * self.ext_std + self.ext_mu
        return self.embedder((raw - self.train_mu) / self.train_std)


class DinoWMTokens(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        proprio_encoder=None,
        img_size: int = 196,
        action_dim: int = 10,
        proprio_dim: int = 2,
        history_size: int = 3,
        interpolate_pos_encoding: bool = True,
        obs_key: str = "pixels",
        predict_bf16: bool = False,
        grad_checkpoint: bool = False,
        compile_predictor: bool = False,
        cost_chunk: int = 128,
        keep_pred_frames: int = 2,
    ):
        super().__init__()
        self.backbone = encoder
        self.backbone.requires_grad_(False)
        self.predict_bf16 = predict_bf16
        self.grad_checkpoint = grad_checkpoint
        self.compile_predictor = compile_predictor
        self._pred_fn = None  # lazily compiled AFTER weights load (key-name safe)
        self.predictor = predictor
        self.action_encoder = _BlockActionEncoder(action_encoder, action_dim)
        self.proprio_encoder = proprio_encoder
        self.img_size = img_size
        self.history_size = history_size
        self.interpolate_pos_encoding = interpolate_pos_encoding
        self.obs_key = obs_key
        self.num_patches = int(predictor.num_patches)
        self.pix_dim = int(self.backbone.config.hidden_size)
        self.act_emb_dim = int(action_encoder.emb_dim)
        self.pro_emb_dim = int(proprio_encoder.emb_dim) if proprio_encoder is not None else 0
        # state token dim (pixel [+ proprio]); action slice lives only inside predict()
        self.tok_dim = self.pix_dim + self.pro_emb_dim
        self.latent_dim = self.num_patches * self.tok_dim
        if proprio_encoder is not None:
            self.register_buffer("pro_mu", torch.zeros(proprio_dim))
            self.register_buffer("pro_std", torch.ones(proprio_dim))
        # planning knobs (not part of the checkpoint)
        self.cost_chunk = int(cost_chunk)
        self.keep_pred_frames = int(keep_pred_frames)
        # One-time additive calibration of the predictor's output offset. The
        # released PreJEPA predictors land in a SHIFTED copy of the encoder's
        # feature space: pred = f(z) + c with c fixed (cosine 0.999 across
        # independent data splits). Subtracting c takes the 1-step
        # pred/copy-last MSE ratio from ~2.6 to ~0.5 on held-out reacher data.
        # Defaults to zeros => byte-identical behaviour for every existing
        # checkpoint; a calibrated checkpoint simply ships a non-zero buffer.
        self.register_buffer("pred_bias", torch.zeros(self.latent_dim))
        # tolerate checkpoints written before pred_bias existed (strict load)
        self._register_load_state_dict_pre_hook(self._pred_bias_compat)

    def _pred_bias_compat(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "pred_bias"
        if key not in state_dict:
            state_dict[key] = torch.zeros(self.latent_dim)

    @property
    def wants_proprio(self) -> bool:
        return self.proprio_encoder is not None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _pixel_tokens(self, pixels: torch.Tensor) -> torch.Tensor:
        """(N, C, H, W) ImageNet-normalized -> (N, P, pix_dim) patch tokens."""
        if pixels.shape[-1] != self.img_size or pixels.shape[-2] != self.img_size:
            pixels = F.interpolate(
                pixels, size=(self.img_size, self.img_size),
                mode="bilinear", antialias=True, align_corners=False,
            )
        pixels = pixels.to(next(self.backbone.parameters()).dtype)
        # transformers >=5 plumbs interpolate_pos_encoding through forward();
        # 4.49's Dinov2 auto-interpolates and rejects the kwarg. Pass it only
        # when the backbone accepts it — the result is identical either way.
        kwargs = {}
        if self.interpolate_pos_encoding:
            import inspect
            if "interpolate_pos_encoding" in inspect.signature(self.backbone.forward).parameters:
                kwargs["interpolate_pos_encoding"] = True
        h = self.backbone(pixels, **kwargs).last_hidden_state
        return h[:, 1:, :]  # drop CLS; keep ALL patch tokens

    def encode(self, info: dict) -> dict:
        x = info[self.obs_key].float()  # (B, T, C, H, W)
        b, t = x.shape[:2]
        toks = self._pixel_tokens(rearrange(x, "b t ... -> (b t) ..."))  # (BT, P, D)
        if self.wants_proprio:
            if "proprio" not in info:
                raise KeyError(
                    "DinoWMTokens (proprio variant): encode() needs info['proprio'] "
                    "(raw env units) alongside pixels"
                )
            pro = info["proprio"].float().reshape(b, t, -1)
            pro = (pro - self.pro_mu) / self.pro_std
            pemb = self.proprio_encoder(pro)  # (B, T, pro_emb)
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
            self._pred_fn = (torch.compile(self.predictor)
                             if self.compile_predictor else self.predictor)
        fn = self._pred_fn
        # the K-step LIP unroll keeps every iteration's rollout graph alive for
        # the final backward — full fp32 activations OOM an 80GB H100 at B=128,
        # so recompute predictor activations in backward instead of storing them
        use_ckpt = (self.grad_checkpoint and torch.is_grad_enabled()
                    and x.requires_grad)
        if self.predict_bf16 and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_ckpt:
                    out = torch.utils.checkpoint.checkpoint(
                        fn, x, use_reentrant=False)
                else:
                    out = fn(x)
            out = out.float()
        elif use_ckpt:
            out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
        else:
            out = fn(x)
        out = rearrange(out, "b (t p) d -> b t p d", t=T)
        out = out[..., : self.tok_dim]  # strip predicted action slice
        return out.reshape(B, T, self.latent_dim) - self.pred_bias

    ####################
    ## Inference only ##
    ####################

    def _pad_hist(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, history_size, D); short histories repeat frame 0."""
        n = self.history_size
        if z.shape[1] < n:
            pad = z[:, :1].expand(-1, n - z.shape[1], -1)
            z = torch.cat([pad, z], dim=1)
        return z[:, -n:]

    def rollout_traj(self, z_hist, plan, a_hist=None):
        """Autoregressive block unroll — the exact convention LIP plans with
        (``solver.lip.rollout_traj``).

        z_hist: (B, history_size, D) latent history,
        plan:   (B, H, action_dim) z-scored action BLOCKS (fs * a_dim),
        a_hist: (B, history_size - 1, action_dim) past blocks; LIP uses zeros
                (the true past blocks are not available to the planner).
        Returns the H imagined latents (B, H, D). Differentiable in ``plan``.
        """
        hs = self.history_size
        if a_hist is None:
            a_hist = plan.new_zeros(plan.shape[0], hs - 1, plan.shape[-1])
        embs = list(z_hist.unbind(dim=1))
        acts = list(a_hist.unbind(dim=1))
        outs = []
        for t in range(plan.shape[1]):
            acts.append(plan[:, t])
            win_e = torch.stack(embs[-hs:], dim=1)
            win_a = torch.stack(acts[-hs:], dim=1)
            nxt = self.predict(win_e, self.action_encoder(win_a))[:, -1]
            embs.append(nxt)
            outs.append(nxt)
        return torch.stack(outs, dim=1)

    def _plan_encode(self, info_dict: dict, pixels_key: str, proprio_keys):
        """Encode a (B, [S,] T, C, H, W) stack -> (B, T, D) flat token latents.

        The sample axis inserted by the solvers is dropped (the observation is
        identical across action candidates), mirroring ``LeWM.rollout``.
        """
        px = info_dict[pixels_key]
        sampled = px.ndim == 6  # (B, S, T, C, H, W): solvers tile over samples
        if sampled:
            px = px[:, 0]
        dev = next(self.parameters()).device
        px = px.to(dev).float()
        enc_in = {self.obs_key: px}
        if self.wants_proprio:
            pro = next(
                (info_dict[k] for k in proprio_keys if info_dict.get(k) is not None),
                None,
            )
            if pro is None:
                raise KeyError(
                    f"proprio-variant DinoWMTokens: info_dict lacks any of "
                    f"{list(proprio_keys)} (raw proprio for the {pixels_key} latent)"
                )
            if not torch.is_tensor(pro):
                import numpy as np

                pro = torch.as_tensor(np.asarray(pro))
            pro = pro.to(dev).float()
            if sampled and pro.ndim >= 3:
                pro = pro[:, 0]
            B, T = px.shape[0], px.shape[1]
            pdim = int(self.pro_mu.numel())
            flat = pro.reshape(B, -1)
            if flat.shape[1] == T * pdim:  # one proprio per frame
                pro = flat.reshape(B, T, pdim)
            else:  # a single state (the goal case): tile it over the frames
                pro = flat[:, -pdim:].unsqueeze(1).expand(-1, T, -1)
            enc_in["proprio"] = pro
        return self.encode(enc_in)["emb"].float()

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Goal-MSE planning cost for CEM / MPPI / gradient solvers.

        ``action_candidates``: (B, S, H, action_dim) z-scored action blocks.
        Returns (B, S) — the solvers assert exactly that shape.

        The rollout is the one LIP already uses on this class: ``history_size``
        encoded frames (short histories repeat frame 0), zero past action
        blocks, H autoregressive predictor steps, cost = MSE between the
        terminal imagined latent and the goal latent. Observation and goal
        latents are cached in ``info_dict`` so the DINOv2 encoder runs once per
        solve rather than once per optimizer iteration (as ``LeWM.get_cost``
        does). The rollout is chunked over the flattened (B*S) axis because the
        flat-token latent is ~75k-d and the predictor sees history_size*196
        tokens per candidate.
        """
        assert "goal" in info_dict, "goal not in info_dict"
        assert action_candidates.ndim == 4, (
            f"expected (B, S, H, A) action candidates, got {tuple(action_candidates.shape)}"
        )
        B, S, H, A = action_candidates.shape
        dev = next(self.parameters()).device

        # -- goal latent (cached across solver iterations)
        if "goal_emb" not in info_dict:
            zg = self._plan_encode(info_dict, "goal", ("goal_proprio", "goal_state"))
            info_dict["goal_emb"] = zg[:, -1:].contiguous()  # (B, 1, D), keeps a time axis
        zg = info_dict["goal_emb"][:, -1]  # (B, D)

        # -- observation history latent (cached across solver iterations)
        if "emb" not in info_dict:
            z = self._plan_encode(info_dict, self.obs_key, ("proprio",))
            info_dict["emb"] = self._pad_hist(z).contiguous()  # (B, hs, D)
        z_hist = info_dict["emb"]

        plans = action_candidates.reshape(B * S, H, A).to(dev).float()
        env_of = torch.arange(B, device=dev).repeat_interleave(S)
        chunk = self.cost_chunk if self.cost_chunk > 0 else B * S

        costs, kept = [], []
        for i in range(0, B * S, chunk):
            idx = env_of[i : i + chunk]
            traj = self.rollout_traj(z_hist[idx], plans[i : i + chunk])  # (n, H, D)
            costs.append(
                F.mse_loss(traj[:, -1], zg[idx], reduction="none").mean(dim=-1)
            )
            if self.keep_pred_frames > 0:
                kept.append(traj[:, -self.keep_pred_frames :])

        if kept:  # (B, S, k, D) — what the TD/value cost hook re-scores
            info_dict["predicted_emb"] = torch.cat(kept).reshape(
                B, S, -1, self.latent_dim
            )
        return torch.cat(costs).view(B, S)


__all__ = ["DinoWMTokens"]
