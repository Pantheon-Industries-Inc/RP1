"""3-frame WINDOW quasimetric: d([z_{t-2L}, z_{t-L}, z_t], [z_g] x3).

RECONSTRUCTED 2026-07-30 after losing volume 3cv5zzezm9 (re-authored from the
original's spec; it produced window3_lejepa.pt / window3_pldm.pt).

Why a window can beat a single-frame terminal cost:
  * it averages WM noise over 3 frames (the imagined terminal alone carries
    0.110 rad open-loop error against a 0.05 rad tolerance);
  * it prefers plans that DWELL near the goal over plans that graze it -- a
    wider imagined basin is more robust to imagination error;
  * unlike the at-rest delta ([z, z-z_prev]) there is no explicitly hackable
    "zero-motion" coordinate: the window is raw frames, so a zero-action plan
    only scores well if its frames actually sit at the goal.

Deploy contract: eval_wm.py's metric hook sees declared latent_dim = 3*D,
infers _m = 3, and feeds the LAST 3 IMAGINED frames of the plan with the goal
TILED 3x. Deploy-legal at the 1-frame policy config -- the policy still receives
ONE real frame; only imagined frames are scored.

Frame spacing L is in PRIMITIVE steps and must equal action_block (5): the
hook's window frames are consecutive imagined latents, one action block apart.
Episode starts clamp to the earliest row (frames duplicate), matching the
planner's own padded starts.

Measured with this value (lejepa, h25, n=6): latched 86.7 as TD+CEM, and as
LIP's critic held 44.2 pooled / latched 90.6. The tiled-goal diagnostic came out
benign (deploy-style query at ~8% of random-pair distance), unlike the at-rest
value whose deploy convention was far off-manifold.
"""

import argparse

import numpy as np
import torch
from loguru import logger as logging

from stable_worldmodel.trm import LatentCache, learners, save_metric
from stable_worldmodel.trm.learners.td import TDConfig

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--out", required=True)
p.add_argument("--lag", type=int, default=5, help="frame spacing in primitive steps (= action_block)")
p.add_argument("--frames", type=int, default=3)
p.add_argument("--expectile", type=float, default=0.1)
p.add_argument("--n-step", type=int, default=50)
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--gamma", type=float, default=1.0)
p.add_argument("--boundary", choices=["legacy", "smooth", "disc"], default="legacy")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda")
a = p.parse_args()

c = LatentCache.load(a.cache)
Z, st = c.z, c.step_idx.numpy()
N, D = Z.shape
F = a.frames

# window row indices [t-(F-1)L, ..., t-L, t], clamped at episode starts
idxs = []
for k in range(F - 1, -1, -1):
    off = k * a.lag
    prev = np.arange(N) - off
    clamp = st < off
    prev[clamp] = np.arange(N)[clamp] - st[clamp]  # earliest row of the episode
    idxs.append(torch.from_numpy(prev).long())
n_clamped = int((st < (F - 1) * a.lag).sum())
logging.info(f"window F={F} lag={a.lag}: {n_clamped}/{N} rows clamp at episode starts")

Zs = torch.cat([Z[i] for i in idxs], dim=1)  # (N, F*D), oldest -> newest
stacked = LatentCache(z=Zs, episode_idx=c.episode_idx, step_idx=c.step_idx,
                      state=c.state, meta={**(c.meta or {}),
                                           "window_frames": F, "window_lag": a.lag})
logging.info(f"stacked cache {tuple(Zs.shape)}")

cfg = TDConfig(head="quasimetric", hidden_dim=256, depth=2, embed_dim=128,
               n_step=a.n_step, gamma=a.gamma, expectile=a.expectile,
               boundary=a.boundary,
               p_cross=0.3, balanced=True, batch_size=1024, steps=a.steps, seed=a.seed)
module = learners.td.fit(stacked, cfg, a.device)

# diagnostic: cost of the DEPLOY-style tiled goal vs the data-window goal. If
# tiled queries are far more expensive at the same location, the deploy
# convention is out-of-distribution (the at-rest failure mode).
with torch.no_grad():
    m = module.to(a.device).eval()
    g = torch.Generator().manual_seed(0)
    pick = torch.randperm(N, generator=g)[:4096]
    win = Zs[pick].to(a.device)
    tiled = Z[pick].repeat(1, F).to(a.device)
    d_self_win = m(win, win).mean().item()
    d_self_tiled = m(tiled, tiled).mean().item()
    d_win_to_tiled = m(win, tiled).mean().item()
    rnd = Zs[torch.randperm(N, generator=g)[:4096]].to(a.device)
    d_rand = m(win, rnd).mean().item()
logging.info(f"[tiled-goal diag] d(win,win)={d_self_win:.3f} d(tiled,tiled)={d_self_tiled:.3f} "
             f"d(win -> SAME loc tiled)={d_win_to_tiled:.3f} vs random-pair {d_rand:.3f}")
if d_win_to_tiled > 0.25 * d_rand:
    logging.warning("tiled-goal queries are far from the data manifold -- deploy convention suspect")

save_metric(module.cpu(), "td", F * D,
            {"head": "quasimetric", "hidden_dim": 256, "depth": 2, "embed_dim": 128,
             "softplus": True, "symmetric": False,
             "window_frames": F, "window_lag": a.lag}, a.out)
logging.success(f"saved -> {a.out} (declared latent_dim={F * D} => hook _m={F})")
