"""Subgoal quality probe v2: chunked CEM rollouts + projected-subgoal preview.

For held-out (z0, zg) pairs at several goal offsets, run the HL planner
(lip and cem variants), extract the subgoal (first predicted waypoint), and
measure value-reachability d_LL(z0, sg), progress d_LL(sg, zg), manifold
distance (NN in a subsampled train-latent pool), and the ridge-readout block
position. Each row is repeated for the NN-PROJECTED subgoal (the solver's
subgoal_project fix) to preview its effect.
"""
import numpy as np
import torch

from stable_worldmodel.solver.lip import PlannerNet, rollout_traj
from stable_worldmodel.trm import LatentCache, load_metric
from stable_worldmodel.wm.hwm import load_hwm

dev = "cuda"
c = LatentCache.load("/workspace/caches/cube_v2wm_fs1.pt")
z = c.z.to(dev).float()
st = c.state.to(dev).float()
eps = c.episodes()
T = 201
rows = torch.full((10000, T), -1, dtype=torch.long)
for e, r in eps.items():
    rows[e] = torch.as_tensor(np.asarray(r))
rows = rows.to(dev)
hwm, blob = load_hwm("/workspace/hwm/hwm_s25m8.pt", device=dev)
K = blob["cfg"]["stride"]
ck = torch.load("/workspace/actors/lip_hl_s25m8_s0.pt", map_location=dev, weights_only=False)
actor = PlannerNet(ck["z_dim"], horizon=ck["horizon"], a_dim=ck["a_dim"], amax=ck["amax"],
                   use_zg=False, use_gate=False, use_z0=False, use_grad=True).to(dev)
actor.load_state_dict(ck["sd"])
actor.eval()
vhl = load_metric(ck["value"], device=dev)
vhl.eval()
vll = load_metric("/workspace/metrics/lipabl_ctrl_s0_value.pt", device=dev)
vll.eval()

idx = torch.randperm(1900000, device=dev)[:200000]
Zc = torch.cat([z[idx], torch.ones(200000, 1, device=dev)], 1)
W = torch.linalg.lstsq(Zc.T @ Zc + 1e-3 * torch.eye(193, device=dev), Zc.T @ st[idx]).solution
r2 = 1 - ((Zc @ W - st[idx]) ** 2).sum() / ((st[idx] - st[idx].mean(0)) ** 2).sum()
print(f"ridge readout R2 {r2.item():.3f}", flush=True)


def readout(x):
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1) @ W


hold = np.arange(9500, 10000)
B = 192
gen = np.random.default_rng(7)
ee = torch.from_numpy(gen.choice(hold, B)).to(dev)

CHUNK = 16384


def roll_chunked(zh, ah, plans):
    outs = []
    for i0 in range(0, plans.shape[0], CHUNK):
        outs.append(rollout_traj(hwm, zh[i0:i0 + CHUNK], ah[i0:i0 + CHUNK],
                                 plans[i0:i0 + CHUNK]))
    return torch.cat(outs)


def hl_lip(z0, zg):
    zh = z0.unsqueeze(1).expand(-1, 3, -1)
    ah = torch.zeros(len(z0), 2, hwm.macro_dim, device=dev)
    A = torch.zeros(len(z0), actor.h, actor.a, device=dev)
    for k in range(ck["iters"]):
        with torch.enable_grad():
            Ain = A.detach().requires_grad_(True)
            tr = rollout_traj(hwm, zh, ah, Ain)
            (gA,) = torch.autograd.grad(vhl(tr[:, -1], zg).sum(), Ain)
        with torch.no_grad():
            trf = rollout_traj(hwm, zh, ah, A)
            E = vhl(trf[:, -1], zg)
            A = actor(A, gA, E, z0, zg, trf, k=k)
    with torch.no_grad():
        wp = rollout_traj(hwm, zh, ah, A)
    return wp[:, 0]


def hl_cem(z0, zg, H):
    zh = z0.unsqueeze(1).expand(-1, 3, -1)
    ah = torch.zeros(len(z0), 2, hwm.macro_dim, device=dev)
    N = 512
    mean = torch.zeros(len(z0), H, hwm.macro_dim, device=dev)
    var = torch.ones_like(mean)
    with torch.no_grad():
        for _ in range(6):
            cand = (torch.randn(len(z0), N, H, hwm.macro_dim, device=dev)
                    * var.unsqueeze(1) + mean.unsqueeze(1)).clamp(-3, 3)
            cand[:, 0] = mean
            tr = roll_chunked(zh.repeat_interleave(N, 0), ah.repeat_interleave(N, 0),
                              cand.reshape(-1, H, hwm.macro_dim))
            cost = (tr[:, -1] - zg.repeat_interleave(N, 0)).pow(2).sum(-1).view(len(z0), N)
            w = torch.softmax(-cost / 0.01, dim=1)
            mean = (w[..., None, None] * cand).sum(1)
            var = ((w[..., None, None] * (cand - mean.unsqueeze(1)) ** 2).sum(1)).sqrt().clamp(min=0.05)
        wp = rollout_traj(hwm, zh, ah, mean)
    return wp[:, 0]


sample_pool = z[rows[:9000:5].reshape(-1)]


def nn_project(q, chunk=200000):
    best_d = torch.full((len(q),), 1e9, device=dev)
    best_i = torch.zeros(len(q), dtype=torch.long, device=dev)
    for i0 in range(0, len(sample_pool), chunk):
        d = torch.cdist(q, sample_pool[i0:i0 + chunk])
        m, ix = d.min(1)
        upd = m < best_d
        best_d[upd] = m[upd]
        best_i[upd] = ix[upd] + i0
    return sample_pool[best_i], best_d


real_hold = z[rows[ee, 60]]
_, bd = nn_project(real_hold)
print(f"NN-dist baseline (real held-out latents): {bd.mean().item():.3f}", flush=True)

for d_env in (25, 50, 100, 200):
    z0 = z[rows[ee, 0]]
    zg = z[rows[ee, min(d_env, T - 1)]]
    d0g = vll(z0, zg).mean().item()
    for name, fn in (("lip", lambda: hl_lip(z0, zg)),
                     ("cem", lambda: hl_cem(z0, zg, max(2, d_env // K)))):
        sg = fn().float()
        sgp, nnd = nn_project(sg)
        for tag, s in (("raw ", sg), ("proj", sgp)):
            d0s = vll(z0, s).mean().item()
            dsg = vll(s, zg).mean().item()
            bs, b0, bg = readout(s), readout(z0), readout(zg)
            print(f"d={d_env:3d} {name} {tag}: d(z0,zg) {d0g:6.1f} | d(z0,sg) {d0s:6.1f} "
                  f"| d(sg,zg) {dsg:6.1f} | NN {nnd.mean().item() if tag=='raw ' else 0:.2f} "
                  f"| dz {(bs[:, 2] - b0[:, 2]).mean().item():+.3f} "
                  f"| xy->goal {(bs[:, :2] - bg[:, :2]).norm(dim=1).mean().item():.3f}", flush=True)
print("PROBE2 DONE", flush=True)
