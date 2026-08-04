"""Tandem HIGH-LEVEL LIPv4 trainer: TD critic (dense fs1 cache, env-step
units) + PlannerNet actor over latent MACRO-actions imagined through a frozen
HWM high level (train_hwm.py checkpoint).

Adaptation of train_lip_ac.py (the schedamax-6k recipe) with:
- actions = macro-actions (a_dim = hwm.macro_dim, raw LayerNorm'd space, no
  z-scoring; --amax clamps it, ~ the paper's 2/98-percentile bounds),
- imagination through the HWM F2 (1 step = hwm stride K env steps),
- solver-convention contexts everywhere: z-history = z0 repeated 3x, ZERO
  macro history (matches HWM training and the stateless HLIP solver exactly;
  no real waypoint history exists at plan time),
- goals: in-episode waypoint at t0 + d*K env steps (d in [1, max-delta]) or
  cross-episode w.p. --p-cross,
- critic identical to train_lip_ac.py: n-step expectile-Huber quasimetric TD
  on the DENSE fs1 cache (td-cache-stride lesson: never train the value on a
  coarse cache), warm-started from --init-value, EMA teacher.

Checkpoint is a standard kind='lip4' actor + extra keys (hwm path, macro_dim,
hl_stride) consumed by HLIPSolver.
"""

import argparse
import copy
import math

import numpy as np
import torch

from stable_worldmodel.solver.lip import PlannerNet, rollout_traj
from stable_worldmodel.trm import LatentCache, load_metric
from stable_worldmodel.trm.io import build_metric, save_metric
from stable_worldmodel.trm.learners.td import _expectile_loss
from stable_worldmodel.trm.samplers import NStepGoalSampler
from stable_worldmodel.wm.hwm import load_hwm


def _cos(base, final, t, T):
    """Cosine decay base -> final over T steps; base when final is None."""
    if final is None or T <= 0:
        return base
    t = min(t, T)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t / T))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="fs1 (dense) LatentCache")
    p.add_argument("--hwm", required=True, help="train_hwm.py checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--out-value", required=True)
    p.add_argument("--init-value", default="")
    # actor
    p.add_argument("--horizon", type=int, default=8, help="plan length in MACRO steps")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--actor-lr-final", type=float, default=3e-5)
    p.add_argument("--amax", type=float, default=3.0,
                   help="macro-action clamp (LayerNorm'd macro space)")
    p.add_argument("--mean-weight", type=float, default=0.1)
    p.add_argument("--max-delta", type=int, default=8,
                   help="max in-episode goal offset in MACRO steps")
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--holdout-ep", type=int, default=-1,
                   help="exclude episodes >= this from actor sampling (-1 = use all; "
                        "set to the hwm's holdout for strict splits)")
    # critic (train_lip_ac parity)
    p.add_argument("--head", choices=["mlp", "quasimetric"], default="quasimetric")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--expectile", type=float, default=0.1)
    p.add_argument("--expectile-final", type=float, default=0.03)
    p.add_argument("--n-step", type=int, default=50)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--td-batch", type=int, default=1024)
    p.add_argument("--td-p-cross", type=float, default=0.3)
    p.add_argument("--td-max-delta", type=int, default=None)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr-final", type=float, default=1e-4)
    p.add_argument("--critic-wd", type=float, default=1e-4)
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--pretrain", type=int, default=-1)
    p.add_argument("--critic-ratio", type=int, default=1)
    p.add_argument("--ema-tau", type=float, default=0.005)
    p.add_argument("--freeze-critic-frac", type=float, default=0.8)
    p.add_argument("--expand-weight", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # ------------------------------------------------------------ frozen high level
    hwm, blob = load_hwm(a.hwm, device=dev)
    K = blob["cfg"]["stride"]
    macro_dim = hwm.macro_dim
    print(f"HWM {a.hwm}: stride {K} env steps/macro, macro_dim {macro_dim}, "
          f"p2 {[round(v,2) for v in blob['l_p2'].tolist()]} "
          f"p98 {[round(v,2) for v in blob['l_p98'].tolist()]} (amax {a.amax})", flush=True)

    # ------------------------------------------------------------ actor data (fs1 cache)
    c = LatentCache.load(a.cache)
    assert (c.meta or {}).get("stride", 1) == 1, "need the fs1 cache"
    z = c.z.to(dev).float()
    eps = c.episodes()
    keys = sorted(eps.keys())
    if a.holdout_ep > 0:
        keys = [k for k in keys if k < a.holdout_ep]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    T_ep = len(ep_rows[ep_ids[0]])
    assert all(len(r) == T_ep for r in ep_rows.values())
    print(f"actor episodes {len(ep_ids)}  T={T_ep}", flush=True)

    def sample(B):
        zh, zg = [], []
        for _ in range(B):
            e = ep_ids[rng.integers(len(ep_ids))]
            rows = ep_rows[e]
            t0 = int(rng.integers(0, T_ep - K))
            zh.append(z[rows[t0]])
            if rng.random() < a.p_cross:
                e2 = ep_ids[rng.integers(len(ep_ids))]
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t0 + d * K, T_ep - 1)]])
        z0 = torch.stack(zh)
        return z0, torch.stack(zg)

    # ------------------------------------------------------------ critic (same fs1 cache)
    if a.init_value:
        vb = torch.load(a.init_value, map_location="cpu", weights_only=False)
        assert vb["latent_dim"] == c.latent_dim, "init-value latent dim mismatch"
        arch = vb["arch"]
        critic = load_metric(a.init_value, device=dev)
    else:
        arch = {"head": a.head, "hidden_dim": a.hidden_dim, "depth": a.depth,
                "embed_dim": a.embed_dim, "softplus": True, "symmetric": False}
        critic = build_metric("td", c.latent_dim, arch).to(dev)
    critic.train()
    teacher = copy.deepcopy(critic).to(dev)
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    teacher.eval()

    td_sampler = NStepGoalSampler(c, n_step=a.n_step, p_cross=a.td_p_cross,
                                  n_buckets=10, balanced=True, seed=a.seed,
                                  max_delta=a.td_max_delta)
    c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)

    n_plan_env = a.horizon * K                    # plan span in env steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan_env), 1.0
    else:
        plan_disc = a.gamma ** n_plan_env
        plan_cost = (1.0 - plan_disc) / (1.0 - a.gamma)

    def critic_step(expand=None, tau=None, lr=None):
        tau = a.expectile if tau is None else tau
        if lr is not None:
            for pg in c_opt.param_groups:
                pg["lr"] = lr
        b = td_sampler.sample(a.td_batch)
        z_t, z_tn, z_g = b["z_t"].to(dev), b["z_tn"].to(dev), b["z_g"].to(dev)
        ne, reached, dist = b["n_eff"].to(dev), b["reached"].to(dev), b["dist"].to(dev)
        with torch.no_grad():
            d_next = teacher(z_tn, z_g)
            if a.gamma >= 1.0:
                cost, disc = ne, torch.ones_like(ne)
            else:
                disc = a.gamma ** ne
                cost = (1.0 - disc) / (1.0 - a.gamma)
            tgt = reached * dist + (1.0 - reached) * (cost + disc * d_next)
        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)
        if expand is not None:
            z0e, zTe, zge = expand
            with torch.no_grad():
                tgt_e = plan_cost + plan_disc * teacher(zTe, zge)
            loss = loss + a.expand_weight * _expectile_loss(
                critic(z0e, zge) - tgt_e, tau, a.huber_beta)
        c_opt.zero_grad(set_to_none=True)
        loss.backward()
        c_opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), critic.parameters()):
                tp.mul_(1.0 - a.ema_tau).add_(a.ema_tau * sp)
        return loss.item()

    # ------------------------------------------------------------ actor (v4)
    net = PlannerNet(z.shape[-1], horizon=a.horizon, a_dim=macro_dim,
                     amax=a.amax, use_zg=False, use_gate=False, use_z0=False,
                     use_grad=True).to(dev)
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=1e-5)

    def actor_step():
        z0, zg = sample(a.batch)
        zh = z0.unsqueeze(1).expand(-1, 3, -1)            # solver convention
        ah = torch.zeros(a.batch, 2, macro_dim, device=dev)
        A = torch.zeros(a.batch, a.horizon, macro_dim, device=dev)
        e_path, zT = [], None
        for k in range(a.iters):
            with torch.enable_grad():
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(hwm, zh, ah, A_in)
                (gA,) = torch.autograd.grad(teacher(traj[:, -1], zg).sum(), A_in)
            traj_f = traj.detach()
            E_feat = teacher(traj_f[:, -1], zg).detach()
            A = net(A, gA.detach(), E_feat, z0, zg, traj_f, k=k)
            tr = rollout_traj(hwm, zh, ah, A)
            zT = tr[:, -1]
            e_path.append(teacher(zT, zg).mean())
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        expand = (z0.detach(), zT.detach(), zg.detach())
        return e_path[0].item(), e_path[-1].item(), expand

    # ------------------------------------------------------------ schedule (train_lip_ac parity)
    pretrain = a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000)
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            print(f"pretrain {i}: td_loss {cl:.4f}", flush=True)

    freeze_at = int(a.freeze_critic_frac * a.steps)
    expand = None
    for step in range(a.steps):
        critic_live = step < freeze_at
        if step == freeze_at:
            print(f"step {step}: critic+teacher frozen", flush=True)
        tau_s = a.expectile if a.expectile_final is None else (
            a.expectile + (a.expectile_final - a.expectile)
            * min(step, freeze_at) / max(freeze_at, 1))
        clr_s = _cos(a.critic_lr, a.critic_lr_final, step, freeze_at)
        alr_s = _cos(a.actor_lr, a.actor_lr_final, step, a.steps)
        for pg in a_opt.param_groups:
            pg["lr"] = alr_s
        cl = float("nan")
        if critic_live:
            for _ in range(a.critic_ratio):
                cl = critic_step(expand if a.expand_weight > 0 else None,
                                 tau=tau_s, lr=clr_s)
                expand = None
        e_first, e_final, expand = actor_step()
        if step % 500 == 0:
            print(f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                  f"td_loss {cl:.4f} tau {tau_s:.3f} alr {alr_s:.2e}", flush=True)

    # ------------------------------------------------------------ save
    save_metric(teacher.cpu(), "td", c.latent_dim, arch, a.out_value)
    print(f"saved teacher value -> {a.out_value}", flush=True)
    net.eval()
    torch.save({"kind": "lip4", "feed": "none", "sd": net.cpu().state_dict(),
                "z_dim": z.shape[-1], "horizon": a.horizon, "iters": a.iters,
                "a_dim": macro_dim, "amax": a.amax,
                "use_gate": False, "use_zg": False, "use_z0": False, "use_grad": True,
                "value": a.out_value, "hwm": a.hwm, "hl_stride": K,
                "macro_dim": macro_dim}, a.out)
    print(f"saved HL learned planner -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
