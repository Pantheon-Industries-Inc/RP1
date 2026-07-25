"""Train the HWM high level (action encoder A_psi + high-level predictor F2)
on frozen v2WM pooled latents (fs1 LatentCache) + raw actions (npz).

Port of HWM (arXiv 2604.03208) to the LeWM cube stack:
- fixed high-level stride (env steps) between waypoints (their maze recipe),
- macro-action = deterministic posterior mean over the primitive-action chunk
  (MLP, LayerNorm on mu — their PosteriorContinuous; --ae tf gives the
  paper's transformer-CLS variant used on Franka/PushT),
- F2 mirrors the LeWM low-level Predictor (AdaLN ConditionalBlock transformer,
  same dims) conditioned on an Embedder of the macro-action and exposes the
  SAME predict/action_encoder API, so solver.lip.rollout_traj / PlannerNet /
  train_lip_ac machinery work at the high level unchanged,
- autoregressive rollout loss (their maze L2 objective; state never reset,
  macro from the true chunk per step), MSE by default, --loss l1 optional,
  --gamma-tf adds a teacher-forced term,
- CRITICAL train==deploy convention: rollouts go through solver.lip's
  rollout_traj with z-history = z0 repeated 3x and ZERO macro history —
  exactly what the (stateless) solver feeds at plan time.

Holdout episodes (>= --holdout-ep) are excluded from training; every eval
interval we report k-step rollout error on them, and after training the Fig-6
comparison against the frozen low-level WM's autoregressive rollout
(--fig6-wm).

Checkpoint: single .pt with the HWM state_dict + macro stats
(mean/std/p2/p98 per dim over aligned train chunks) + action norm stats.
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.solver.lip import rollout_traj
from stable_worldmodel.trm import LatentCache
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
    """High-level WM with the SAME predict/action_encoder API as LeWM, so
    solver.lip.rollout_traj / rollout_terminal work on it unchanged.
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
        zh = z0.unsqueeze(1).expand(-1, 3, -1)
        ah = torch.zeros(z0.shape[0], 2, self.macro_dim,
                         device=z0.device, dtype=macros.dtype)
        return rollout_traj(self, zh, ah, macros)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="fs1 LatentCache (stride 1)")
    p.add_argument("--actions", required=True, help="actions npz (action, ep_len, ep_offset)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=25, help="HL stride in env steps")
    p.add_argument("--macro-dim", type=int, default=8)
    p.add_argument("--pred-horizon", type=int, default=6, help="rollout steps per sample")
    p.add_argument("--ae", choices=["mlp", "tf"], default="mlp")
    p.add_argument("--loss", choices=["mse", "l1"], default="mse")
    p.add_argument("--gamma-tf", type=float, default=0.0, help="teacher-forced loss weight")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-pred-proj", action="store_true")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=3e-5)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--holdout-ep", type=int, default=9500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=256)
    p.add_argument("--fig6-wm", default="", help="v2WM dir for low-level rollout comparison")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------ data
    c = LatentCache.load(a.cache)
    assert (c.meta or {}).get("stride", 1) == 1, "need the fs1 cache"
    z_all = c.z.to(dev).float()                            # (N, 192)
    npz = np.load(a.actions)
    act = npz["action"]
    ep_len = npz["ep_len"].astype(np.int64)
    ep_off = npz["ep_offset"].astype(np.int64)
    amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    act_n = np.nan_to_num((act - amu) / astd).astype(np.float32)
    act_t = torch.from_numpy(act_n).to(dev)

    n_ep, T_ep = len(ep_len), int(ep_len[0])
    assert (ep_len == T_ep).all(), "variable episode lengths unsupported"
    eps = c.episodes()
    rows_lut = torch.full((n_ep, T_ep), -1, dtype=torch.long)
    for e, r in eps.items():
        rr = np.asarray(r)
        assert len(rr) == T_ep
        rows_lut[e] = torch.from_numpy(rr)
    rows_lut = rows_lut.to(dev)
    ep_off_t = torch.from_numpy(ep_off).to(dev)

    K, H = a.stride, a.pred_horizon
    max_t0 = (T_ep - 1) - H * K
    assert max_t0 >= 0, f"pred-horizon {H} x stride {K} exceeds episode span {T_ep - 1}"
    train_eps = np.arange(0, a.holdout_ep)
    hold_eps = np.arange(a.holdout_ep, n_ep)
    act_dim = act.shape[-1]
    H_full = (T_ep - 1) // K                       # max whole-episode macro steps
    print(f"episodes train/hold {len(train_eps)}/{len(hold_eps)}  T={T_ep}  "
          f"stride={K} horizon={H} (full {H_full}) max_t0={max_t0}", flush=True)

    def batch_from(ep_pool, B, t0=None, horizon=H, gen=rng):
        e = torch.from_numpy(gen.choice(ep_pool, B)).to(dev)
        if t0 is None:
            t0 = torch.from_numpy(
                gen.integers(0, (T_ep - 1) - horizon * K + 1, B)).to(dev)
        else:
            t0 = torch.full((B,), t0, dtype=torch.long, device=dev)
        wp_t = t0[:, None] + K * torch.arange(horizon + 1, device=dev)[None]   # (B, H+1)
        z_wp = z_all[rows_lut[e[:, None], wp_t]]                               # (B, H+1, D)
        base = ep_off_t[e][:, None, None] + t0[:, None, None] \
            + K * torch.arange(horizon, device=dev)[None, :, None] \
            + torch.arange(K, device=dev)[None, None, :]
        chunks = act_t[base.reshape(B, -1)].reshape(B, horizon, K, act_dim)
        return z_wp, chunks

    # ------------------------------------------------------------ model
    hwm = HWM(a.macro_dim, z_dim=z_all.shape[-1], depth=a.depth, dropout=a.dropout,
              ae=a.ae, chunk_len=K, act_dim=act_dim,
              use_pred_proj=not a.no_pred_proj).to(dev)
    n_par = sum(x.numel() for x in hwm.parameters())
    print(f"HWM params {n_par/1e6:.2f}M (ae={a.ae}, macro={a.macro_dim}, "
          f"depth={a.depth})", flush=True)
    opt = torch.optim.AdamW(hwm.parameters(), lr=a.lr, weight_decay=a.wd)
    lossfn = F.mse_loss if a.loss == "mse" else F.l1_loss

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        f = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr_final + 0.5 * (a.lr - a.lr_final) * (1 + math.cos(math.pi * f))

    @torch.no_grad()
    def evaluate(step):
        hwm.eval()
        gen = np.random.default_rng(1234)
        Hh = min(H_full, max(H, 1))
        z_wp, chunks = batch_from(hold_eps, a.eval_episodes, t0=0, horizon=Hh, gen=gen)
        preds = hwm.rollout_from(z_wp[:, 0], hwm.encode_chunks(chunks))
        scale = z_wp[:, 1:].abs().mean().item()
        msg = [f"eval@{step}"]
        for k in sorted({1, 2, 4, Hh}):
            if 1 <= k <= Hh:
                l1 = (preds[:, k - 1] - z_wp[:, k]).abs().mean().item()
                msg.append(f"ro{k}(={k*K}env) l1 {l1:.4f}")
        msg.append(f"| z-scale {scale:.4f}")
        print("  ".join(msg), flush=True)
        hwm.train()

    # ------------------------------------------------------------ train
    hwm.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        z_wp, chunks = batch_from(train_eps, a.batch)
        macros = hwm.encode_chunks(chunks)
        preds = hwm.rollout_from(z_wp[:, 0], macros)
        loss = lossfn(preds, z_wp[:, 1:])
        if a.gamma_tf > 0:
            n_ctx = min(3, H)
            tf_pred = hwm.predict(z_wp[:, :n_ctx], hwm.action_encoder(macros[:, :n_ctx]))
            loss = loss + a.gamma_tf * lossfn(tf_pred, z_wp[:, 1:n_ctx + 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hwm.parameters(), 10.0)
        opt.step()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.5f} lr {lr_at(step):.2e}", flush=True)
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            evaluate(step + 1)

    # ------------------------------------------------------------ macro stats (aligned grid, train eps)
    hwm.eval()
    ls = []
    with torch.no_grad():
        for i0 in range(0, len(train_eps), 512):
            ee = train_eps[i0:i0 + 512]
            B = len(ee)
            e = torch.from_numpy(ee).to(dev)
            base = ep_off_t[e][:, None, None] \
                + K * torch.arange(H_full, device=dev)[None, :, None] \
                + torch.arange(K, device=dev)[None, None, :]
            chunks = act_t[base.reshape(B, -1)].reshape(B, H_full, K, act_dim)
            ls.append(hwm.encode_chunks(chunks).reshape(-1, a.macro_dim).cpu())
    L = torch.cat(ls)
    l_mean, l_std = L.mean(0), L.std(0) + 1e-6
    l_p2 = torch.quantile(L, 0.02, dim=0)
    l_p98 = torch.quantile(L, 0.98, dim=0)
    print("macro stats mean", [round(v, 4) for v in l_mean.tolist()],
          "\nstd", [round(v, 4) for v in l_std.tolist()],
          "\np2", [round(v, 4) for v in l_p2.tolist()],
          "\np98", [round(v, 4) for v in l_p98.tolist()], flush=True)

    cfg = dict(macro_dim=a.macro_dim, z_dim=z_all.shape[-1], stride=K,
               pred_horizon=H, ae=a.ae, depth=a.depth, dropout=a.dropout,
               loss=a.loss, gamma_tf=a.gamma_tf, steps=a.steps,
               use_pred_proj=not a.no_pred_proj, chunk_len=K, act_dim=act_dim,
               holdout_ep=a.holdout_ep, seed=a.seed)
    torch.save({"kind": "hwm", "cfg": cfg, "sd": hwm.state_dict(),
                "l_mean": l_mean, "l_std": l_std, "l_p2": l_p2, "l_p98": l_p98,
                "amu": amu, "astd": astd}, a.out)
    print(f"saved -> {a.out}\nconfig {json.dumps(cfg)}", flush=True)

    # ------------------------------------------------------------ Fig-6 probe vs low level
    if a.fig6_wm:
        wm = swm.wm.utils.load_pretrained(a.fig6_wm).to(dev).eval()
        wm.requires_grad_(False)
        gen = np.random.default_rng(4321)
        z_wp, chunks = batch_from(hold_eps, a.eval_episodes, t0=0, horizon=H_full, gen=gen)
        with torch.no_grad():
            preds_hl = hwm.rollout_from(z_wp[:, 0], hwm.encode_chunks(chunks))
        B = z_wp.shape[0]
        fs = 5
        assert K % fs == 0, "stride must be a multiple of the WM frameskip"
        blocks = chunks.reshape(B, H_full * K // fs, fs * act_dim)
        zh = z_wp[:, 0:1].expand(-1, 3, -1)
        ah = torch.zeros(B, 2, fs * act_dim, device=dev)
        with torch.no_grad():
            traj_ll = rollout_traj(wm, zh, ah, blocks)      # (B, H_full*K/fs, D)
        print("FIG6 (holdout, t0=0; l1 per env-horizon):", flush=True)
        for k in range(1, H_full + 1):
            hl = (preds_hl[:, k - 1] - z_wp[:, k]).abs().mean().item()
            ll = (traj_ll[:, k * K // fs - 1] - z_wp[:, k]).abs().mean().item()
            print(f"  {k*K:4d} env steps: F2 {hl:.4f}  vs F1-rollout {ll:.4f}  "
                  f"({'F2 wins' if hl < ll else 'F1 wins'})", flush=True)


if __name__ == "__main__":
    main()
