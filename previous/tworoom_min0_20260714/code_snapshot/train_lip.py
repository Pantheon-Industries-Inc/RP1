"""Learned Iterative Planner (LIP): a planning ALGORITHM learned purely by
optimizing against the value function. No behavior cloning anywhere.

The planner is an update rule f_theta(A_k, grad_A V, V_k, z0, zg) -> dA applied
for K iterations from A_0 = 0. Training unrolls the frozen WM for the full
horizon, evaluates the frozen value at the terminal latent, and backprops
loss = V(z_T(A_K), g) + 0.1 * mean_k V_k  through the whole refinement chain
to theta ONLY. Gradient features are detached (standard learned-optimizer
practice). Goals via HER (future + cross-episode), no MC labels needed.
"""
import argparse
import h5py
import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.solver.lip import PlannerNet, rollout_terminal, rollout_traj
from stable_worldmodel.trm import LatentCache, load_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--h5", required=True)
    p.add_argument("--wm", required=True)
    p.add_argument("--value", required=True, help="frozen value = the teacher")
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--max-delta", type=int, default=10)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mean-weight", type=float, default=0.1)
    p.add_argument("--arch", choices=["v4", "mlp"], default="v4",
                   help="'v4' (default) = LIPv4: minimal-input gate-free MLP "
                        "[A, grad V, E] (kind='lip4'); 'mlp' = legacy full-input "
                        "gated PlannerNet")
    p.add_argument("--amax", type=float, default=2.5,
                   help="plan clamp in z-scored action units")
    p.add_argument("--head-scale", type=float, default=1.0,
                   help="scale the update-head init (e.g. 0.01 = near-identity start)")
    p.add_argument("--feed", choices=["none", "end", "traj"], default="none",
                   help="extra refiner input: imagined terminal (end) or full path (traj). "
                        "'end' tightens training draws at equal peak (LIP-v2)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)
    value = load_metric(a.value, device=dev)
    for prm in value.parameters():
        prm.requires_grad_(False)
    value.eval()

    c = LatentCache.load(a.cache)
    z = c.z.to(dev).float()
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    with h5py.File(a.h5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
    amu, astd = act.mean(0), act.std(0) + 1e-6
    act_n = ((act - amu) / astd).astype(np.float32)

    def blocks(e, t):
        h0 = int(ep_off[e] + 5 * t)
        return act_n[h0:h0 + 5].reshape(-1)

    def sample(B):
        zh, ah, zg = [], [], []
        for _ in range(B):
            e = ep_ids[rng.integers(len(ep_ids))]
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
            if rng.random() < a.p_cross:
                e2 = ep_ids[rng.integers(len(ep_ids))]
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t + d, L - 1)]])
        return (torch.stack(zh), torch.from_numpy(np.stack(ah)).to(dev), torch.stack(zg))

    a_dim = act.shape[-1] * 5          # action-block dim = env action dim x frameskip
    if a.arch == "v4":
        net = PlannerNet(z.shape[-1], horizon=a.horizon, a_dim=a_dim, feed=a.feed,
                         amax=a.amax, use_zg=False, use_gate=False, use_z0=False,
                         head_scale=a.head_scale).to(dev)
    else:
        net = PlannerNet(z.shape[-1], horizon=a.horizon, a_dim=a_dim, feed=a.feed,
                         amax=a.amax, head_scale=a.head_scale).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)

    for step in range(a.steps):
        zh, ah, zg = sample(a.batch)
        z0 = zh[:, -1]
        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)
        e_path = []
        for k in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            with torch.enable_grad():
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(wm, zh, ah, A_in)
                (gA,) = torch.autograd.grad(value(traj[:, -1], zg).sum(), A_in)
            traj_f = rollout_traj(wm, zh, ah, A).detach()
            E_feat = value(traj_f[:, -1], zg).detach()
            A = net(A, gA.detach(), E_feat, z0, zg, traj_f)  # theta-dependent update
            e_path.append(value(rollout_terminal(wm, zh, ah, A), zg).mean())
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step}: E_final {e_path[-1].item():.3f} "
                  f"E_first {e_path[0].item():.3f}", flush=True)

    net.eval()
    if a.arch == "v4":
        kind = "lip4"
    else:
        kind = "lip" if a.feed == "none" else "lip2"
    torch.save({"kind": kind, "feed": a.feed, "sd": net.cpu().state_dict(),
                "z_dim": z.shape[-1], "horizon": a.horizon, "iters": a.iters,
                "a_dim": a_dim, "amax": a.amax,
                "use_gate": net.use_gate, "use_zg": net.use_zg,
                "use_z0": net.use_z0, "head_scale": a.head_scale,
                "value": a.value}, a.out)
    print(f"saved learned planner -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
