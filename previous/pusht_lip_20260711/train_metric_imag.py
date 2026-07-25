"""Imagination-augmented TD (DYNA-style input matching).

At plan time the value is queried on WM-imagined latents; standard TD trains
only on real encoded latents. Here, with prob --p-imag the state input z_t is
replaced by a WM rollout of 1..4 action blocks (true dataset actions) ending
at t — the TD target (n_eff / dist / real z_tn bootstrap) is unchanged, so the
value learns "drifted imagined latent of state s" maps to the same cost-to-go
as s. Same quasimetric head / expectile / n-step recipe as the winner.
"""

import argparse
import copy

import h5py
import numpy as np
import torch
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.solver.lip import rollout_traj
from stable_worldmodel.trm import LatentCache
from stable_worldmodel.trm.io import save_metric
from stable_worldmodel.trm.learners.td import TDConfig, _expectile_loss, _make_head
from stable_worldmodel.trm.samplers import NStepGoalSampler

FS = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", default="lewm_pusht_official")
    p.add_argument("--out", required=True)
    p.add_argument("--expectile", type=float, default=0.01)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--p-imag", type=float, default=0.5)
    p.add_argument("--max-roll", type=int, default=4, help="max imagination depth (blocks)")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    dev = a.device
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)

    c = LatentCache.load(a.cache)
    z_cpu = c.z.float()
    z = z_cpu.to(dev)
    ep = c.episode_idx.numpy()
    step = c.step_idx.numpy()
    N = len(z)
    with h5py.File(a.h5, "r") as h:
        act = h["action"][:N]
    amu, asd = act.mean(0), act.std(0) + 1e-6
    act_n = torch.from_numpy(((act - amu) / asd).astype(np.float32)).to(dev)

    # rows eligible as imagination endpoints: t with >= (2 + max_roll) blocks
    # of same-episode history before them
    need = (2 + a.max_roll) * FS
    ok = np.zeros(N, bool)
    ok[need:] = ep[need:] == ep[:-need]

    def imagine(rows_t):
        """Roll the WM to time t for each row in rows_t; returns (B, D)."""
        B = len(rows_t)
        r = rng.integers(1, a.max_roll + 1, size=B)  # imagination depth per sample
        zh, ah, plans, depth = [], [], [], []
        for i, t in enumerate(rows_t):
            k = int(r[i])
            s0 = t - k * FS  # rollout starts here; history is s0-2FS, s0-FS, s0
            zh.append(torch.stack([z[s0 - 2 * FS], z[s0 - FS], z[s0]]))
            ah.append(torch.stack([act_n[s0 - 2 * FS: s0 - FS].reshape(-1),
                                   act_n[s0 - FS: s0].reshape(-1)]))
            blocks = [act_n[s0 + j * FS: s0 + (j + 1) * FS].reshape(-1)
                      for j in range(a.max_roll)]
            plans.append(torch.stack(blocks))
            depth.append(k - 1)  # index into rollout traj
        with torch.no_grad():
            traj = rollout_traj(wm, torch.stack(zh), torch.stack(ah), torch.stack(plans))
        return traj[torch.arange(B), torch.tensor(depth, device=dev)]

    cfg = TDConfig(head="quasimetric", hidden_dim=a.hidden_dim, depth=a.depth,
                   embed_dim=a.embed_dim, n_step=a.n_step, expectile=a.expectile,
                   p_cross=a.p_cross, steps=a.steps, batch_size=a.batch_size, seed=a.seed)
    value = _make_head(cfg, c.latent_dim).to(dev)
    target = copy.deepcopy(value).to(dev)
    for prm in target.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    class RowSampler(NStepGoalSampler):
        """NStepGoalSampler that also returns the global cache row of z_t."""

        def sample(self, batch_size):
            import numpy as np
            t_idx = np.empty(batch_size, np.int64)
            tn_idx = np.empty(batch_size, np.int64)
            g_idx = np.empty(batch_size, np.int64)
            n_eff = np.empty(batch_size, np.float32)
            reached = np.zeros(batch_size, np.float32)
            dist = np.zeros(batch_size, np.float32)
            for b in range(batch_size):
                e = self.ep_ids[self.rng.integers(0, len(self.ep_ids))]
                rows = self.episodes[e]
                L = len(rows)
                t = int(self.rng.integers(0, L - 1))
                ne = min(self.n, L - 1 - t)
                t_idx[b], tn_idx[b], n_eff[b] = rows[t], rows[t + ne], ne
                if self.rng.random() < self.p_cross:
                    g_idx[b] = int(self.rng.integers(0, self.n_total))
                else:
                    _hi = L - 1 - t
                    if self.max_delta is not None:
                        _hi = min(_hi, self.max_delta)
                    delta = self._offset(_hi)
                    g_idx[b] = rows[t + delta]
                    if delta <= ne:
                        reached[b], dist[b] = 1.0, float(delta)
            return {
                "z_t": self.z[t_idx], "z_tn": self.z[tn_idx], "z_g": self.z[g_idx],
                "n_eff": torch.from_numpy(n_eff), "reached": torch.from_numpy(reached),
                "dist": torch.from_numpy(dist), "t_idx": torch.from_numpy(t_idx),
            }

    sampler = RowSampler(c, n_step=cfg.n_step, p_cross=cfg.p_cross,
                         n_buckets=cfg.n_buckets, balanced=cfg.balanced, seed=cfg.seed)

    value.train()
    n_im_total = 0
    for it in range(cfg.steps):
        b = sampler.sample(cfg.batch_size)
        z_t, z_tn, z_g = b["z_t"].to(dev), b["z_tn"].to(dev), b["z_g"].to(dev)
        ne, reached, dist = b["n_eff"].to(dev), b["reached"].to(dev), b["dist"].to(dev)
        # replace a fraction of state inputs with imagined versions
        if a.p_imag > 0:
            rows = b["t_idx"].numpy()
            cand = np.flatnonzero(ok[rows] & (rng.random(len(rows)) < a.p_imag))
            if len(cand) > 0:
                z_t = z_t.clone()
                z_t[torch.from_numpy(cand).to(dev)] = imagine(rows[cand])
                n_im_total += len(cand)
        with torch.no_grad():
            d_next = target(z_tn, z_g)
            tgt = reached * dist + (1.0 - reached) * (ne + d_next)
        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters()):
                tp.mul_(1.0 - cfg.tau).add_(cfg.tau * sp)
        if it % 500 == 0:
            print(f"step {it}: loss {loss.item():.4f} imag_used {n_im_total}", flush=True)

    if n_im_total == 0:
        raise SystemExit("imagination never used; aborting")
    value.eval()
    arch = {"head": "quasimetric", "hidden_dim": a.hidden_dim,
            "embed_dim": a.embed_dim, "depth": a.depth}
    save_metric(value.cpu(), "td", c.latent_dim, arch, a.out)
    logging.success(f"saved imagination-augmented TD ({n_im_total} imagined inputs) -> {a.out}")


if __name__ == "__main__":
    main()
