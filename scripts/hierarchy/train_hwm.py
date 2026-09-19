"""Train the high-level world model F2 (and its action encoder A_psi).

Port of the 2026-07 campaign's `train_hwm.py` onto this repo's data layout
(fs1 LatentCache + action h5 from tools/build_action_h5).

Objective: autoregressive rollout MSE over the waypoint grid — the state is
never reset during the unroll and each macro comes from the true action chunk,
so the model is trained exactly the way it is deployed. Rollouts go through
``HWM.rollout_from``, i.e. the solver convention (z-history = z0 x3, ZERO macro
history), because the deployed solver is stateless and has no real waypoint
history at plan time.

Geometry for h200: stride 25 with a horizon-8 unroll spans 200 primitive steps,
which is the whole cube episode (201 steps, 9 waypoints).

    pixi run python scripts/hierarchy/train_hwm.py \
        --cache /path/cube_v2wm_fs1.pt --h5 /path/cube_actions.h5 \
        --out /path/hwm_s25m8.pt --stride 25 --macro-dim 8 --pred-horizon 8
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rlp.core.world_model.hwm import HWM, save_hwm
from rlp.data import LatentCache


def _cos(base: float, final: float | None, t: int, T: int) -> float:
    if final is None or T <= 0:
        return base
    t = min(t, T)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t / T))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="fs1 (dense) LatentCache")
    p.add_argument("--h5", required=True, help="action h5 (tools/build_action_h5)")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=25, help="primitive steps per macro")
    p.add_argument("--macro-dim", type=int, default=8)
    p.add_argument("--pred-horizon", type=int, default=8, help="macro steps unrolled per sample")
    p.add_argument("--ae", choices=["mlp", "tf"], default="mlp")
    p.add_argument("--loss", choices=["mse", "l1"], default="mse")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--holdout-ep", type=int, default=-1)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    import h5py
    import hdf5plugin  # noqa: F401  (Blosc filters used by the cube h5)

    cache = LatentCache.load(a.cache)
    assert (cache.meta or {}).get("stride", 1) == 1, "need the fs1 (dense) cache"
    z = cache.z.to(dev).float()
    eps = cache.episodes()

    with h5py.File(a.h5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
        ep_len = h["ep_len"][:] if "ep_len" in h else None
    # cube pads episode-terminal steps with NaN actions: plain mean/std would
    # poison every normalised action (see reacher-h5-nan-actions).
    amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    act_n = np.nan_to_num((act - amu) / astd).astype(np.float32)
    act_dim = act.shape[-1]
    K = a.stride
    H = a.pred_horizon

    # waypoint grid: need H+1 waypoints, i.e. H*K primitive steps of room
    usable = []
    for ep_id in sorted(eps):
        if a.holdout_ep > 0 and ep_id >= a.holdout_ep:
            continue
        rows = np.asarray(eps[ep_id])
        if len(rows) >= H * K + 1:
            usable.append((ep_id, rows))
    assert usable, f"no episode long enough for horizon {H} at stride {K}"
    print(
        f"{len(usable)} training episodes; waypoints/episode "
        f"{len(usable[0][1]) // K + 1}; plan span {H * K} primitive steps",
        flush=True,
    )

    model = HWM(
        macro_dim=a.macro_dim,
        z_dim=cache.latent_dim,
        ae=a.ae,
        chunk_len=K,
        act_dim=act_dim,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    lossfn = F.mse_loss if a.loss == "mse" else F.l1_loss

    def sample(batch: int, pool: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (z_wp (B,H+1,D), chunks (B,H,K,act_dim))."""
        zs, cs = [], []
        for _ in range(batch):
            ep_id, rows = pool[int(rng.integers(len(pool)))]
            max_start = len(rows) - H * K - 1
            s0 = int(rng.integers(0, max_start + 1))
            wp_rows = [rows[s0 + k * K] for k in range(H + 1)]
            zs.append(z[torch.as_tensor(np.asarray(wp_rows), dtype=torch.long)])
            base = int(ep_off[ep_id])
            ch = []
            for k in range(H):
                h0 = base + s0 + k * K
                if ep_len is not None:
                    h0 = min(h0, base + max(0, int(ep_len[ep_id]) - K))
                ch.append(act_n[h0 : h0 + K])
            cs.append(np.stack(ch))
        return torch.stack(zs), torch.as_tensor(np.stack(cs), device=dev)

    holdout = []
    if a.holdout_ep > 0:
        for ep_id in sorted(eps):
            if ep_id >= a.holdout_ep:
                rows = np.asarray(eps[ep_id])
                if len(rows) >= H * K + 1:
                    holdout.append((ep_id, rows))

    for step in range(a.steps):
        lr = _cos(a.lr, a.lr_final, step, a.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        z_wp, chunks = sample(a.batch, usable)
        macros = model.encode_chunks(chunks)
        preds = model.rollout_from(z_wp[:, 0], macros)
        loss = lossfn(preds, z_wp[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % a.eval_every == 0 or step == a.steps - 1:
            msg = [f"step {step}: train {loss.item():.4f}"]
            if holdout:
                with torch.no_grad():
                    zh, chh = sample(min(256, a.batch * 4), holdout)
                    ph = model.rollout_from(zh[:, 0], model.encode_chunks(chh))
                    # per-horizon l1 in primitive-step units: the h200-relevant
                    # number is the last one (k=H, i.e. H*K env steps out)
                    for k in (1, H // 2, H):
                        l1 = (ph[:, k - 1] - zh[:, k]).abs().mean().item()
                        # IDENTITY baseline: predict "nothing moves" (z0). This is
                        # the control that decides whether F2 has learned anything
                        # at this horizon -- comparing to zscale (the mean |z|)
                        # flatters the model, since a latent that barely moves
                        # makes any predictor look good on that scale.
                        ident = (zh[:, 0] - zh[:, k]).abs().mean().item()
                        msg.append(f"ro{k}(={k * K}env) l1 {l1:.4f} vs identity {ident:.4f}")
                    msg.append(f"zscale {zh.abs().mean().item():.3f}")
            print("  ".join(msg), flush=True)

    cfg = {
        "macro_dim": a.macro_dim,
        "z_dim": cache.latent_dim,
        "act_dim": act_dim,
        "stride": K,
        "ae": a.ae,
        "pred_horizon": H,
    }
    with torch.no_grad():
        mac = model.encode_chunks(sample(512, usable)[1]).reshape(-1, a.macro_dim).cpu().numpy()
    stats = {
        "l_p2": torch.tensor(np.percentile(mac, 2, axis=0)),
        "l_p98": torch.tensor(np.percentile(mac, 98, axis=0)),
        "act_mean": torch.tensor(amu),
        "act_std": torch.tensor(astd),
    }
    path = save_hwm(model.cpu(), Path(a.out), cfg, stats)
    print(f"saved {path}", flush=True)
    print(f"macro 2/98 pct: {stats['l_p2'].tolist()} / {stats['l_p98'].tolist()}", flush=True)


if __name__ == "__main__":
    main()
