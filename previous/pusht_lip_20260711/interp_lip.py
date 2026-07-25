"""Interp: why does LIP underperform TD+CEM on PushT?

A) Plan-time optimization comparison on eval-like tasks (same frozen WM + value):
   who reaches lower terminal value E?  zero-plan / LIP 1-shot (true vs zeroed
   action history — the solver zeroes it at eval, training used true) /
   LIP+8 restarts / Adam gradient descent / CEM 300x30.
B) Teacher quality: Spearman rank corr of V(z_t, z_{t+k}) vs true step gap k,
   compared against raw latent L2 distance — is the TD value even a better
   ranking signal than the latent cost it replaced?
"""

import sys

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

import stable_worldmodel as swm
from stable_worldmodel.solver.lip import PlannerNet, rollout_traj
from stable_worldmodel.trm import LatentCache, load_metric

DEV = "cuda"
VALUE_PT = "/workspace/metrics/td_e0.03_n5.pt"
ACTOR_PT = "/workspace/actors/lip_k8_lr3e-4_st8000.pt"
CACHE5 = "/workspace/caches/pusht_official_fs5.pt"
CACHE1 = "/workspace/caches/pusht_official_fs1.pt"
H5 = "/workspace/swm_home/datasets/pusht_expert_train.h5"
NT = 64  # tasks per horizon
rng = np.random.default_rng(0)

wm = swm.wm.utils.load_pretrained("lewm_pusht_official").to(DEV).eval()
wm.requires_grad_(False)
value = load_metric(VALUE_PT, device=DEV)
value.eval()
ck = torch.load(ACTOR_PT, map_location=DEV, weights_only=False)
actor = PlannerNet(ck["z_dim"], horizon=ck["horizon"], a_dim=ck["a_dim"], feed="none").to(DEV)
actor.load_state_dict(ck["sd"])
actor.eval()
K, HOR, ADIM = ck["iters"], ck["horizon"], ck["a_dim"]

c = LatentCache.load(CACHE5)
z = c.z.to(DEV).float()
eps = c.episodes()
with h5py.File(H5, "r") as h:
    act = h["action"][:]
    ep_off = h["ep_offset"][:]
amu, astd = act.mean(0), act.std(0) + 1e-6
act_n = ((act - amu) / astd).astype(np.float32)


def blocks(e, t):
    h0 = int(ep_off[e] + 5 * t)
    return act_n[h0:h0 + 5].reshape(-1)


def make_tasks(delta):
    zh, ah, zg = [], [], []
    keys = [k for k in eps if len(eps[k]) > delta + 4]
    for _ in range(NT):
        e = keys[rng.integers(len(keys))]
        rows = eps[e]
        L = len(rows)
        t = int(rng.integers(2, L - delta))
        zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
        ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
        zg.append(z[rows[min(t + delta, L - 1)]])
    return (torch.stack(zh),
            torch.from_numpy(np.stack(ah)).to(DEV).float(),
            torch.stack(zg))


def E_of(A, zh, ah, zg):
    with torch.no_grad():
        return value(rollout_traj(wm, zh, ah, A)[:, -1], zg)


def lip_solve(zh, ah, zg, restarts=1, noise=0.5):
    B = zh.shape[0]
    R = restarts
    zh_r = zh.repeat_interleave(R, 0)
    ah_r = ah.repeat_interleave(R, 0)
    zg_r = zg.repeat_interleave(R, 0)
    z0 = zh_r[:, -1]
    A = noise * torch.randn(B * R, HOR, ADIM, device=DEV)
    A[::R] = 0.0
    for _ in range(K):
        with torch.enable_grad():
            A_in = A.detach().requires_grad_(True)
            traj = rollout_traj(wm, zh_r, ah_r, A_in)
            (gA,) = torch.autograd.grad(value(traj[:, -1], zg_r).sum(), A_in)
        with torch.no_grad():
            traj_f = rollout_traj(wm, zh_r, ah_r, A)
            E_feat = value(traj_f[:, -1], zg_r)
            A = actor(A, gA, E_feat, z0, zg_r, traj_f)
    E = E_of(A, zh_r, ah_r, zg_r).view(B, R)
    return E.min(dim=1).values


def adam_solve(zh, ah, zg, steps=60, lr=0.3):
    A = torch.zeros(zh.shape[0], HOR, ADIM, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    for _ in range(steps):
        E = value(rollout_traj(wm, zh, ah, A)[:, -1], zg).sum()
        opt.zero_grad()
        E.backward()
        opt.step()
        with torch.no_grad():
            A.clamp_(-2.5, 2.5)
    return E_of(A.detach(), zh, ah, zg)


def cem_solve(zh, ah, zg, samples=300, iters=30, topk=30, chunk=8):
    B = zh.shape[0]
    out = torch.zeros(B, device=DEV)
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        n = hi - lo
        zh_c = zh[lo:hi].repeat_interleave(samples, 0)
        ah_c = ah[lo:hi].repeat_interleave(samples, 0)
        zg_c = zg[lo:hi].repeat_interleave(samples, 0)
        mu = torch.zeros(n, HOR, ADIM, device=DEV)
        sd = torch.ones(n, HOR, ADIM, device=DEV)
        best = torch.full((n,), float("inf"), device=DEV)
        for _ in range(iters):
            A = mu.repeat_interleave(samples, 0) + sd.repeat_interleave(samples, 0) * torch.randn(
                n * samples, HOR, ADIM, device=DEV)
            E = E_of(A, zh_c, ah_c, zg_c).view(n, samples)
            best = torch.minimum(best, E.min(dim=1).values)
            idx = E.argsort(dim=1)[:, :topk]
            elite = torch.gather(
                A.view(n, samples, HOR, ADIM), 1,
                idx[:, :, None, None].expand(-1, -1, HOR, ADIM))
            mu = elite.mean(1)
            sd = elite.std(1) + 1e-4
        out[lo:hi] = best
    return out


print("=== A) plan-time optimization comparison (terminal value E, lower better)")
for name, delta in [("h25", 5), ("h50", 10)]:
    zh, ah, zg = make_tasks(delta)
    ah0 = torch.zeros_like(ah)
    res = {
        "zero-plan (true ahist)": E_of(torch.zeros(NT, HOR, ADIM, device=DEV), zh, ah, zg),
        "LIP 1shot true-ahist": lip_solve(zh, ah, zg),
        "LIP 1shot zero-ahist(eval)": lip_solve(zh, ah0, zg),
        "LIP r8 zero-ahist": lip_solve(zh, ah0, zg, restarts=8),
        "Adam60 true-ahist": adam_solve(zh, ah, zg),
        "CEM true-ahist": cem_solve(zh, ah, zg),
        "CEM zero-ahist": cem_solve(zh, ah0, zg),
    }
    cem = res["CEM true-ahist"]
    print(f"-- {name} (delta={delta} blocks, {NT} tasks)")
    for k, v in res.items():
        gap = (v - cem).mean().item()
        win = (v <= cem + 1e-6).float().mean().item() * 100
        print(f"  {k:28s} E={v.mean().item():7.3f}  gap-vs-CEM={gap:+7.3f}  <=CEM {win:4.0f}%")

print("\n=== B) teacher ranking quality (fs1 pairs, k = true primitive-step gap)")
c1 = LatentCache.load(CACHE1)
z1 = c1.z.to(DEV).float()
eps1 = c1.episodes()
keys1 = [k for k in eps1 if len(eps1[k]) > 60]
V_list, L_list, k_list = [], [], []
for _ in range(4000):
    e = keys1[rng.integers(len(keys1))]
    rows = eps1[e]
    t = int(rng.integers(0, len(rows) - 55))
    k = int(rng.integers(1, 51))
    a_, b_ = z1[rows[t]], z1[rows[t + k]]
    V_list.append((a_, b_))
    k_list.append(k)
za = torch.stack([p[0] for p in V_list])
zb = torch.stack([p[1] for p in V_list])
with torch.no_grad():
    V = value(za, zb).cpu().numpy()
L = torch.norm(za - zb, dim=1).cpu().numpy()
kk = np.array(k_list)
print(f"  Spearman(V_td, k)     = {spearmanr(V, kk).statistic:.3f}")
print(f"  Spearman(latent_L2, k)= {spearmanr(L, kk).statistic:.3f}")
for kmax in (10, 25, 50):
    m = kk <= kmax
    print(f"    k<={kmax:2d}: V {spearmanr(V[m], kk[m]).statistic:.3f}  L2 {spearmanr(L[m], kk[m]).statistic:.3f}")
