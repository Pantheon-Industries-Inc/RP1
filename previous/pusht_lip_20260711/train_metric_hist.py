"""History-conditioned TD quasimetric: V([z, Δz], [z_g, 0]).

Δz = z_t − z_{t−5} (block-scale, matches the planner's imagined-frame spacing).
Goal deltas are zeroed in training AND eval ("arrive at rest") so the two sides
stay consistent. Everything else = the standard offline TD recipe (n-step HER,
expectile toward the min, quasimetric head). Saved via save_metric with
latent_dim = 2D, so load_metric works unchanged; the eval hook detects the 2x
input dim and feeds [z_T, z_T − z_{T-1}] / [z_g, 0].
"""

import argparse
import copy

import torch
from loguru import logger as logging

from stable_worldmodel.trm import LatentCache
from stable_worldmodel.trm.io import save_metric
from stable_worldmodel.trm.learners.td import TDConfig, _expectile_loss, _make_head
from stable_worldmodel.trm.samplers import NStepGoalSampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expectile", type=float, default=0.03)
    p.add_argument("--n-step", type=int, default=5)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--delta-stride", type=int, default=5)
    p.add_argument("--input-noise", type=float, default=0.0,
                   help="gaussian noise (x per-dim std) on all value inputs")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    c = LatentCache.load(a.cache)
    z = c.z.float()
    ep = c.episode_idx
    D = z.shape[1]
    k = a.delta_stride
    dz = torch.zeros_like(z)
    same = ep[k:] == ep[:-k]
    dz[k:][same] = z[k:][same] - z[:-k][same]
    aug = LatentCache(z=torch.cat([z, dz], 1), episode_idx=c.episode_idx,
                      step_idx=c.step_idx, state=c.state,
                      meta={**(c.meta or {}), "hist_delta_stride": k})

    cfg = TDConfig(head="quasimetric", hidden_dim=a.hidden_dim, depth=a.depth,
                   embed_dim=a.embed_dim, n_step=a.n_step, expectile=a.expectile,
                   p_cross=a.p_cross, steps=a.steps, batch_size=a.batch_size,
                   seed=a.seed, gamma=a.gamma)

    # fork of learners.td.fit with the delta half of goal inputs zeroed
    device = a.device
    torch.manual_seed(cfg.seed)
    value = _make_head(cfg, aug.latent_dim).to(device)
    target = copy.deepcopy(value).to(device)
    for prm in target.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = NStepGoalSampler(aug, n_step=cfg.n_step, p_cross=cfg.p_cross,
                               n_buckets=cfg.n_buckets, balanced=cfg.balanced,
                               seed=cfg.seed, max_delta=cfg.max_delta)
    z_std = aug.z.std(0).to(device)
    g = cfg.gamma
    value.train()
    for step in range(cfg.steps):
        b = sampler.sample(cfg.batch_size)
        z_t, z_tn, z_g = b["z_t"].to(device), b["z_tn"].to(device), b["z_g"].to(device)
        z_g = z_g.clone()
        z_g[:, D:] = 0.0  # goals arrive "at rest"
        if a.input_noise > 0:
            z_t = z_t + a.input_noise * z_std * torch.randn_like(z_t)
            z_tn = z_tn + a.input_noise * z_std * torch.randn_like(z_tn)
            z_g = z_g + a.input_noise * z_std * torch.randn_like(z_g)
        ne, reached, dist = b["n_eff"].to(device), b["reached"].to(device), b["dist"].to(device)
        with torch.no_grad():
            d_next = target(z_tn, z_g)
            if g >= 1.0:
                cst, disc = ne, torch.ones_like(ne)
            else:
                disc = g ** ne
                cst = (1.0 - disc) / (1.0 - g)
            tgt = reached * dist + (1.0 - reached) * (cst + disc * d_next)
        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters()):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * sp)
        if step % 500 == 0:
            print(f"step {step}: loss {loss.item():.4f} pred {pred.mean().item():.2f}", flush=True)

    value.eval()
    arch = {"head": "quasimetric", "hidden_dim": a.hidden_dim,  # noqa
            "embed_dim": a.embed_dim, "depth": a.depth,
            "hist_delta_stride": k}
    save_metric(value, "td", aug.latent_dim, arch, a.out)
    logging.success(f"saved history-TD metric (in_dim={aug.latent_dim}) -> {a.out}")


if __name__ == "__main__":
    main()
