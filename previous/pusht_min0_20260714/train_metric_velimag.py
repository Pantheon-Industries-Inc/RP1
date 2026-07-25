"""Velocity + imagination-consistent TD quasimetric for aliased single-frame latents.

Fixes the two PushT teacher failures in one move:
  F1 velocity aliasing: state-side input is [z_t, Δz_t] (Δ stride 5 = one action
     block = the spacing of imagined frames), goal side [z_g, 0] ("arrive at
     rest" — success is pose-only). hist re-run: this alone LOST (118 vs 154).
  F2 train/plan mismatch: with prob --p-imag the state input is replaced by
     [ẑ_t, ẑ_t − ẑ_{t−5}] where BOTH frames come from a frozen-WM rollout of the
     true dataset actions ending at t (depth 2..--max-roll blocks, uniform).
     The TD target (n_eff / dist / real-input bootstrap) is unchanged: the
     imagined rendition of s must map to the same cost-to-go as s. Differencing
     amplifies WM smoothing, so the velocity feature is exactly the input whose
     imagined distribution must be trained on, not calibrated after the fact.

--no-vel drops the Δz halves (single-frame + imagination only) — the
imag-only ablation. --p-imag 0 keeps Δz but never imagines — the hist
ablation. Default = both mechanisms on.

--win-frames K (K≥2) replaces [z, Δz] with a K-frame block-spaced window
[z_{t-5(K-1)}, ..., z_t] (goal side = z_g tiled K times = "at rest at the
goal"); short histories clamp to the episode's first frame (the solver pads
the same way). Imagined replacements take the last K frames of a true-action
rollout, so the depth profile matches the planner's ẑ_{H-K+1..H} queries.

Same recipe as the deep-sweep winner otherwise: offline n-step HER TD,
expectile toward the min, quasimetric head, EMA target, γ=1, frozen-latent
fs1 cache. Saved via save_metric with latent_dim = 2D (or D with --no-vel),
so the eval hook's 2× input-dim detection feeds [z_T, z_T − z_{T−1}] / [z_g, 0]
unchanged.
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

FS = 5  # primitive steps per action block == Δz stride == imagined-frame spacing


class RowSampler(NStepGoalSampler):
    """NStepGoalSampler that also returns global cache rows for t / t+n / g.

    ``n_list``: optional set of backup lengths sampled uniformly per pair
    (mixed-n TD: short n for local crispness, long n for direct far-field
    supervision).
    """

    def __init__(self, *args, n_list=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_list = list(n_list) if n_list else None

    def sample(self, batch_size):
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
            n = self.n if self.n_list is None else int(
                self.n_list[self.rng.integers(0, len(self.n_list))])
            ne = min(n, L - 1 - t)
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
            "t_idx": torch.from_numpy(t_idx), "tn_idx": torch.from_numpy(tn_idx),
            "g_idx": torch.from_numpy(g_idx), "n_eff": torch.from_numpy(n_eff),
            "reached": torch.from_numpy(reached), "dist": torch.from_numpy(dist),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="fs1 latent cache")
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", default="lewm_pusht_official")
    p.add_argument("--out", required=True)
    p.add_argument("--expectile", type=float, default=0.01)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--n-mix", default="",
                   help="comma list (e.g. '1,5,25'): per-sample n drawn uniformly "
                        "from the set — n=1 keeps the local field crisp while long "
                        "n supervises the far field directly (TD(lambda)-flavored)")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--p-imag", type=float, default=0.5)
    p.add_argument("--min-roll", type=int, default=2,
                   help="min imagination depth in blocks (≥2 so Δẑ is fully imagined)")
    p.add_argument("--max-roll", type=int, default=5,
                   help="max imagination depth (plan-time terminal queries sit at 5)")
    p.add_argument("--no-vel", action="store_true",
                   help="drop the Δz halves: single-frame input + imagination only")
    p.add_argument("--win-frames", type=int, default=0,
                   help="K>=2: state = K-frame block-spaced window instead of [z, Δz]; "
                        "goal = z_g tiled K times")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    K = a.win_frames
    mode = "win" if K >= 2 else ("single" if a.no_vel else "delta")
    assert not (K >= 2 and a.no_vel), "--win-frames and --no-vel are exclusive"
    if mode == "delta":
        assert a.min_roll >= 2, "min-roll < 2 would mix a real frame into the imagined Δẑ"
    if mode == "win":
        assert K <= a.max_roll, "window longer than the deepest rollout"
    r_lo = max(a.min_roll, K) if mode == "win" else a.min_roll

    dev = a.device
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    c = LatentCache.load(a.cache)
    z = c.z.float().to(dev)
    N, D = z.shape
    ep = c.episode_idx.numpy()
    step_gpu = c.step_idx.long().to(dev)

    # real block-scale Δz (zero for the first FS steps of an episode, as in hist)
    dz = torch.zeros_like(z)
    same = torch.from_numpy(ep[FS:] == ep[:-FS]).to(dev)
    dz[FS:][same] = z[FS:][same] - z[:-FS][same]

    wm = None
    act_n = None
    ok = np.zeros(N, bool)
    if a.p_imag > 0:
        wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
        wm.requires_grad_(False)
        with h5py.File(a.h5, "r") as h:
            act = h["action"][:]
        amu, asd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6  # full-h5 stats (= tandem/solver)
        act_n = torch.from_numpy(((act[:N] - amu) / asd).astype(np.float32)).to(dev)
        act_n = torch.nan_to_num(act_n)  # episode-terminal pad rows must not poison rollouts
        need = (2 + a.max_roll) * FS  # 3-frame history + deepest rollout, same episode
        ok[need:] = ep[need:] == ep[:-need]
        logging.info(f"imagination-eligible rows: {ok.sum()}/{N}")

    ar = torch.arange  # noqa: E741

    def imagine(rows_t):
        """WM-imagined state input for cache rows t (true dataset actions).

        Rolls r ∈ [r_lo, max_roll] blocks ending at t. Returns (B, in_dim):
        delta  -> [ẑ_t, ẑ_t − ẑ_{t−FS}] (both frames imagined)
        win    -> last K imagined frames, flattened
        single -> ẑ_t
        """
        B = len(rows_t)
        r = torch.from_numpy(rng.integers(r_lo, a.max_roll + 1, size=B)).to(dev)
        s0 = torch.as_tensor(rows_t, device=dev) - r * FS
        zh = z[torch.stack([s0 - 2 * FS, s0 - FS, s0], 1)]                  # (B,3,D)
        hidx = s0[:, None] + torch.arange(-2 * FS, 0, device=dev)[None]     # (B,2*FS)
        ah = act_n[hidx].reshape(B, 2, -1)                                  # (B,2,FS*a)
        # clamp: rollout blocks beyond depth r are computed-and-discarded, but
        # their action reads must not run past the cache tail
        pidx = (s0[:, None] + torch.arange(a.max_roll * FS, device=dev)[None]).clamp_(max=N - 1)
        plans = act_n[pidx].reshape(B, a.max_roll, -1)                      # (B,R,FS*a)
        with torch.no_grad():
            traj = rollout_traj(wm, zh, ah, plans)                          # (B,R,D)
        bi = ar(B, device=dev)
        if mode == "win":
            frames = [traj[bi, r - K + j] for j in range(K)]                # depths r-K+1..r
            return torch.cat(frames, -1)
        zt = traj[bi, r - 1]
        if mode == "single":
            return zt
        return torch.cat([zt, zt - traj[bi, r - 2]], -1)

    def aug(rows, goal=False):
        rows = torch.as_tensor(rows, device=dev)
        base = z[rows]
        if mode == "single":
            return base
        if mode == "win":
            if goal:
                return torch.cat([base] * K, -1)
            frames = []
            for j in range(K):                          # oldest -> newest, clamp to ep start
                off = torch.clamp(step_gpu[rows], max=FS * (K - 1 - j))
                frames.append(z[rows - off])
            return torch.cat(frames, -1)
        d = torch.zeros_like(base) if goal else dz[rows]
        return torch.cat([base, d], -1)

    in_dim = {"single": D, "delta": 2 * D, "win": K * D}[mode]
    cfg = TDConfig(head="quasimetric", hidden_dim=a.hidden_dim, depth=a.depth,
                   embed_dim=a.embed_dim, n_step=a.n_step, expectile=a.expectile,
                   p_cross=a.p_cross, steps=a.steps, batch_size=a.batch_size,
                   seed=a.seed, gamma=a.gamma)
    value = _make_head(cfg, in_dim).to(dev)
    target = copy.deepcopy(value).to(dev)
    for prm in target.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW(value.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_list = [int(x) for x in a.n_mix.split(",") if x.strip()] if a.n_mix else None
    sampler = RowSampler(c, n_step=cfg.n_step, p_cross=cfg.p_cross,
                         n_buckets=cfg.n_buckets, balanced=cfg.balanced,
                         seed=cfg.seed, max_delta=cfg.max_delta, n_list=n_list)
    if n_list:
        logging.info(f"mixed-n TD: n per pair from {n_list}")
    g = cfg.gamma
    value.train()
    n_im_total = 0
    for step in range(cfg.steps):
        b = sampler.sample(cfg.batch_size)
        rows_t = b["t_idx"].to(dev)
        z_t = aug(rows_t)
        z_tn = aug(b["tn_idx"].to(dev))
        z_g = aug(b["g_idx"].to(dev), goal=True)
        if a.p_imag > 0:
            rt = b["t_idx"].numpy()
            cand = np.flatnonzero(ok[rt] & (rng.random(len(rt)) < a.p_imag))
            if len(cand) > 0:
                ct = torch.from_numpy(cand).to(dev)
                z_t = z_t.clone()
                z_t[ct] = imagine(rt[cand])
                n_im_total += len(cand)
        ne, reached, dist = b["n_eff"].to(dev), b["reached"].to(dev), b["dist"].to(dev)
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
            print(f"step {step}: loss {loss.item():.4f} pred {pred.mean().item():.2f} "
                  f"imag_used {n_im_total}", flush=True)

    if a.p_imag > 0 and n_im_total == 0:
        raise SystemExit("imagination never used; aborting")
    value.eval()
    arch = {"head": "quasimetric", "hidden_dim": a.hidden_dim,
            "embed_dim": a.embed_dim, "depth": a.depth}
    if mode == "delta":
        arch["hist_delta_stride"] = FS
    elif mode == "win":
        arch["win_frames"] = K
    save_metric(value.cpu(), "td", in_dim, arch, a.out)
    logging.success(f"saved velimag TD (mode={mode}, in_dim={in_dim}, p_imag={a.p_imag}, "
                    f"imag_used={n_im_total}) -> {a.out}")


if __name__ == "__main__":
    main()
