"""QRL-style quasimetric training (Wang & Isola): induce TRUE asymmetry offline.

    max E_{random pairs}[phi(d(a, b))]
    s.t. E_{transitions}[relu(d(z_t, z_{t+1}) - 1)^2] <= eps^2

Lagrangian with dual ascent on lambda. Directions unconstrained by data
(e.g. un-pushing the block) get pushed UP -> real irreversibility asymmetry,
unlike expectile TD which only fits forward targets and generalizes symmetric.
Saved via save_metric (learner 'td') so load_metric/eval hook work unchanged.
"""

import argparse

import numpy as np
import torch
from loguru import logger as logging

from stable_worldmodel.trm import LatentCache
from stable_worldmodel.trm.head import QuasimetricHead
from stable_worldmodel.trm.io import save_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--eps", type=float, default=0.25, help="allowed rms slack on edge constraint")
    p.add_argument("--phi-cap", type=float, default=300.0, help="cap on pushed distances (~max steps)")
    p.add_argument("--lam-init", type=float, default=10.0)
    p.add_argument("--lam-lr", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--p-cross", type=float, default=0.5, help="fraction of push pairs drawn cross-episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    c = LatentCache.load(args.cache)
    z = c.z.float().to(dev)
    ep = c.episode_idx.numpy()
    N = len(z)
    # transition edge index: consecutive rows within an episode
    same = np.flatnonzero(ep[1:] == ep[:-1])  # row i -> i+1 valid
    eps_rows = {e: np.flatnonzero(ep == e) for e in np.unique(ep)}
    keys = [k for k in eps_rows if len(eps_rows[k]) > 10]

    value = QuasimetricHead(z.shape[1], hidden_dim=args.hidden_dim,
                            embed_dim=args.embed_dim, depth=args.depth).to(dev)
    opt = torch.optim.AdamW(value.parameters(), lr=args.lr, weight_decay=1e-5)
    lam = torch.tensor(args.lam_init, device=dev)

    B = args.batch_size
    for step in range(args.steps):
        # constraint batch: observed 1-step transitions
        i = torch.from_numpy(rng.choice(same, B)).to(dev)
        d_edge = value(z[i], z[i + 1])
        viol = torch.relu(d_edge - 1.0).pow(2).mean()

        # push batch: half same-episode forward pairs, half cross-episode
        n_cross = int(B * args.p_cross)
        a_idx = rng.integers(0, N, n_cross)
        b_idx = rng.integers(0, N, n_cross)
        sa, sb = [], []
        for _ in range(B - n_cross):
            e = keys[rng.integers(len(keys))]
            rows = eps_rows[e]
            t = rng.integers(0, len(rows) - 1)
            t2 = rng.integers(t + 1, len(rows))
            sa.append(rows[t])
            sb.append(rows[t2])
        pa = torch.from_numpy(np.concatenate([a_idx, np.array(sa)])).to(dev)
        pb = torch.from_numpy(np.concatenate([b_idx, np.array(sb)])).to(dev)
        d_push = value(z[pa], z[pb])
        push = torch.clamp(d_push, max=args.phi_cap).mean()

        loss = -push + lam.detach() * viol
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        opt.step()
        with torch.no_grad():
            lam = (lam * torch.exp(torch.tensor(args.lam_lr, device=dev)
                                   * (viol.detach() - args.eps ** 2))).clamp(1e-2, 1e5)
        if step % 500 == 0:
            print(f"step {step}: push {push.item():.1f} viol {viol.item():.4f} "
                  f"lam {lam.item():.1f} edge_mean {d_edge.mean().item():.2f}", flush=True)

    value.eval()
    arch = {"head": "quasimetric", "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim, "depth": args.depth}
    save_metric(value.cpu(), "td", z.shape[1], arch, args.out)
    logging.success(f"saved QRL quasimetric -> {args.out}")


if __name__ == "__main__":
    main()
