#!/usr/bin/env python3
"""PWM-style feedforward policy, trained actor-critic on the MRN quasimetric.

Adapts PWM (arXiv 2407.02466, code github.com/imgeorgiev/PWM) in the one place
that matters -- the actor is a **feedforward closed-loop policy** whose gradient
comes from a first-order rollout through the frozen world model -- while keeping
this project's value function and training scheme unchanged.

What is taken from PWM
    * the actor is ``pi(z, z_g)``, evaluated once per imagined step, closed loop
      in latent space (PWM: ``actions = tanh(self.actor(z))`` inside the rollout)
    * the objective is a discounted sum over an imagined H-step latent rollout,
      normalised by the horizon and averaged over the batch
      (PWM: ``actor_loss /= self.horizon ; actor_loss.mean()``)
    * Adam at ``actor_lr 5e-4`` with a linear schedule, grad-norm clip 100

What is NOT taken from PWM
    * **no 3-critic ensemble, no MSE-on-TD(lambda) critic, no reward head.**
      The value is the MRN quasimetric this project already uses,
      ``C(z_i -> z_j) = ||u_i - u_j|| + max_k relu(v_j - v_i)_k``: non-negative,
      obeys the triangle inequality, so long distances compose from short
      transitions. Because MRN is zero at identity and grows with separation it
      is a *cost*, so it enters the objective negated -- ``V = -C``.
    * **no learned reward.** In a goal-reaching task the sparse success reward is
      a threshold of goal distance, i.e. of the very quantity C already models,
      so a reward head adds a learned approximation of information we have
      exactly. The per-step signal instead comes from C evaluated at every
      imagined step, which is *denser* than the sparse reward would be.
    * **the world model stays frozen.** PWM keeps a third optimiser
      (``wm_optimizer``, ``model_lr 3e-4``) fine-tuning the model; ours are
      author-released checkpoints and fine-tuning them would break
      comparability with the rest of the campaign.

Actor-critic structure mirrors ``train_lip_ac.py`` so the two differ *only* in
the actor:
    critic  d_phi   MRN head, expectile-TD on cached transitions
    teacher d_bar   EMA(d_phi) (Polyak ``--ema-tau``), serves bootstrap targets
                    and scores the actor, so the actor sees a stationary value
    actor   pi      feedforward, first-order through the frozen WM

Objective (default ``--dense``)::

    J = - sum_{t=1..H} gamma^t * d_bar(z_t, z_g)
    actor_loss = -J / H, averaged over the batch

``--terminal`` uses only ``-gamma^H d_bar(z_H, z_g)``, which is LIPv4's
objective, isolating the actor architecture as the single variable.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.solver.lip import rollout_traj  # noqa: E402
from stable_worldmodel.trm import LatentCache, load_metric, save_metric  # noqa: E402
from stable_worldmodel.trm.learners.td import _expectile_loss  # noqa: E402
from stable_worldmodel.trm.samplers import NStepGoalSampler  # noqa: E402


class PWMActor(torch.nn.Module):
    """``pi(z, z_g) -> action block``; feedforward, one pass per imagined step.

    Goal enters as a displacement ``proj(z_g - z)``, matching PlannerNetV3's
    ``goal_mode='diff'`` so this actor and LIPv4's refiner see the same goal
    parameterisation. ``tanh * amax`` puts the block in the same action box the
    LIPv4 actors clip to.
    """

    def __init__(self, z_dim: int, a_dim: int, width: int = 512, layers: int = 3,
                 amax: float = 2.2, zproj: int = 256) -> None:
        super().__init__()
        self.amax = float(amax)
        self.zp = torch.nn.Linear(z_dim, zproj)
        self.gp = torch.nn.Linear(z_dim, zproj)
        net: list[torch.nn.Module] = [torch.nn.Linear(2 * zproj, width), torch.nn.SiLU()]
        for _ in range(layers - 1):
            net += [torch.nn.Linear(width, width), torch.nn.SiLU()]
        net += [torch.nn.Linear(width, a_dim)]
        self.net = torch.nn.Sequential(*net)

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.zp(z), self.gp(z_g - z)], dim=-1)
        return torch.tanh(self.net(h)) * self.amax


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True, help='fs5 cache (one latent per action block)')
    p.add_argument('--cache-td', default='', help='fs1 cache for the critic (default: --cache)')
    p.add_argument('--wm', required=True, help='frozen world-model checkpoint name')
    p.add_argument('--init-value', required=True,
                   help='MRN checkpoint from train_metric.py to warm-start the critic')
    p.add_argument('--out', required=True)
    p.add_argument('--out-value', required=True, help='final teacher (save_metric format)')
    # objective
    p.add_argument('--dense', action='store_true', default=True,
                   help='per-step cost sum (default); denser than a sparse reward')
    p.add_argument('--terminal', dest='dense', action='store_false',
                   help="terminal cost only = LIPv4's objective, isolating the actor")
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--gamma', type=float, default=0.99, help="PWM's discount")
    p.add_argument('--amax', type=float, default=2.2)
    # actor
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--actor-lr', type=float, default=5e-4, help="PWM's actor_lr")
    p.add_argument('--actor-lr-final', type=float, default=5e-5)
    p.add_argument('--grad-norm', type=float, default=100.0)
    # critic (co-trained; mirrors train_lip_ac.py)
    p.add_argument('--critic-lr', type=float, default=1e-3)
    p.add_argument('--critic-lr-final', type=float, default=1e-4)
    p.add_argument('--critic-ratio', type=int, default=1,
                   help='critic steps per actor step')
    p.add_argument('--expectile', type=float, default=0.1)
    p.add_argument('--expectile-final', type=float, default=0.03)
    p.add_argument('--huber-beta', type=float, default=1.0)
    p.add_argument('--ema-tau', type=float, default=0.005, help='teacher Polyak rate')
    p.add_argument('--pretrain', type=int, default=0,
                   help='critic-only warmup steps (0 when warm-started)')
    p.add_argument('--freeze-at', type=float, default=1.0,
                   help='freeze critic+teacher after this fraction of steps')
    # sampling / misc
    p.add_argument('--steps', type=int, default=8000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--n-step', type=int, default=50)
    p.add_argument('--p-cross', type=float, default=0.3)
    p.add_argument('--max-delta', type=int, default=12)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    dev = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    c_act = LatentCache.load(a.cache)
    c_td = LatentCache.load(a.cache_td) if a.cache_td else c_act
    wm = swm.wm.utils.load_pretrained(a.wm).to(dev).eval()
    wm.requires_grad_(False)                       # frozen, deliberately

    blob = torch.load(a.init_value, map_location='cpu', weights_only=False)
    _ld, _cd = int(blob['latent_dim']), int(c_td.latent_dim)
    assert _cd and _ld % _cd == 0 and _ld // _cd in (1, 3), (
        f"init-value width {_ld} is neither 1x nor 3x the cache dim {_cd}")
    vframes = _ld // _cd
    _wlag = int(blob['arch'].get('window_lag', 5)) if vframes > 1 else 1
    if vframes == 3:
        print(f'[vframes] 3-frame window value, lag {_wlag}', flush=True)
    arch = blob['arch']
    critic = load_metric(a.init_value, device=dev)
    Ztd, st_td = c_td.z, c_td.step_idx.numpy()

    def _wpair(traj, zg):
        """Stack the last `vframes` imagined frames; tile the goal to match."""
        if vframes <= 1:
            return traj[:, -1], zg
        cols = []
        for k in range(vframes - 1, -1, -1):
            i = traj.shape[1] - 1 - k
            cols.append(traj[:, i] if i >= 0 else traj[:, 0])
        return torch.cat(cols, dim=-1), zg.repeat(*([1] * (zg.dim() - 1)), vframes)

    def _wrow(idx):
        """Window-stack TD-cache rows exactly as train_window.py builds them."""
        if vframes <= 1:
            return Ztd[idx]
        st = torch.as_tensor(st_td)
        cols = []
        for k in range(vframes - 1, -1, -1):
            off = k * _wlag
            j = idx - off
            j = torch.where(st[idx] < off, idx - st[idx], j)
            cols.append(Ztd[j])
        return torch.cat(cols, dim=-1)
    teacher = copy.deepcopy(critic).to(dev)
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    teacher.eval()

    a_dim = int(getattr(c_act, 'action', torch.zeros(1, 10)).shape[-1])
    hs = int(getattr(wm, 'history_size', 3))
    actor = PWMActor(c_act.latent_dim, a_dim, a.width, a.layers, a.amax).to(dev)

    a_opt = torch.optim.Adam(actor.parameters(), lr=a.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=a.critic_lr)
    a_sched = torch.optim.lr_scheduler.LinearLR(
        a_opt, 1.0, a.actor_lr_final / a.actor_lr, total_iters=a.steps)
    c_sched = torch.optim.lr_scheduler.LinearLR(
        c_opt, 1.0, a.critic_lr_final / a.critic_lr, total_iters=a.steps)

    s_act = NStepGoalSampler(c_act, n_step=a.n_step, p_cross=a.p_cross,
                             balanced=True, seed=a.seed, max_delta=a.max_delta)
    s_td = NStepGoalSampler(c_td, n_step=a.n_step, p_cross=a.p_cross,
                            balanced=True, seed=a.seed + 1, max_delta=a.max_delta)

    logging.info(
        f'PWM actor-critic | z_dim={c_act.latent_dim} a_dim={a_dim} H={a.horizon} '
        f'gamma={a.gamma} amax={a.amax} objective={"dense" if a.dense else "terminal"} '
        f'| MRN critic co-trained, EMA teacher tau={a.ema_tau}, WM frozen')

    freeze_step = int(a.freeze_at * a.steps)

    def critic_step(tau: float) -> float:
        b = s_td.sample(a.batch)
        z_t, z_tn, z_g = (b['z_t'].to(dev), b['z_tn'].to(dev), b['z_g'].to(dev))
        ne = b['n_eff'].to(dev)
        reached, dist = b['reached'].to(dev), b['dist'].to(dev)
        with torch.no_grad():
            d_next = (teacher(_wrow(b['tn_idx']).to(dev), _wrow(b['g_idx']).to(dev))
                      if vframes > 1 else teacher(z_tn, z_g))
            if a.gamma >= 1.0:
                c, disc = ne, torch.ones_like(ne)
            else:
                disc = a.gamma ** ne
                c = (1.0 - disc) / (1.0 - a.gamma)
            tgt = reached * dist + (1.0 - reached) * (c + disc * d_next)
        loss = _expectile_loss(
            (critic(_wrow(b['t_idx']).to(dev), _wrow(b['g_idx']).to(dev))
             if vframes > 1 else critic(z_t, z_g)) - tgt, tau, a.huber_beta)
        c_opt.zero_grad(set_to_none=True)
        loss.backward()
        c_opt.step()
        with torch.no_grad():                       # Polyak teacher
            for tp, sp in zip(teacher.parameters(), critic.parameters()):
                tp.mul_(1.0 - a.ema_tau).add_(a.ema_tau * sp)
        return float(loss.item())

    # ---- critic-only warmup
    for i in range(a.pretrain):
        critic_step(a.expectile)

    pbar = tqdm(range(a.steps), desc=f'pwm-ac(H={a.horizon},g={a.gamma})')
    for step in pbar:
        frac = step / max(a.steps - 1, 1)
        tau = a.expectile + (a.expectile_final - a.expectile) * frac

        if step < freeze_step:
            for _ in range(a.critic_ratio):
                c_loss = critic_step(tau)
        else:
            c_loss = float('nan')

        # ---- actor: first-order rollout through the frozen WM
        b = s_act.sample(a.batch)
        z0 = b['z_t'].to(dev).float()
        zg = b['z_g'].to(dev).float()
        B = z0.shape[0]
        z_hist = z0.unsqueeze(1).expand(-1, hs, -1).contiguous()
        a_hist = torch.zeros(B, max(hs - 1, 1), a_dim, device=dev)

        plan: list[torch.Tensor] = []
        z = z0
        J = torch.zeros(B, device=dev)
        disc = 1.0
        for t in range(a.horizon):
            plan.append(actor(z, zg))
            # committed rollout_traj is (wm, z_hist, a_hist, plan) with a fixed
            # 3-frame attention window -- the same path LIPv4 trained through
            traj = rollout_traj(wm, z_hist, a_hist, torch.stack(plan, 1))
            z = traj[:, -1]
            disc = disc * a.gamma
            if a.dense:                              # dense per-step cost
                J = J - disc * teacher(*_wpair(traj, zg))
        if not a.dense:                              # terminal only (LIPv4-like)
            J = -disc * teacher(*_wpair(traj, zg))

        a_loss = (-J / a.horizon).mean()
        a_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        if a.grad_norm:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), a.grad_norm)
        a_opt.step()
        a_sched.step()
        if step < freeze_step:
            c_sched.step()

        if step % 200 == 0:
            with torch.no_grad():
                E = teacher(*_wpair(traj, zg)).mean().item()
            pbar.set_postfix(actor=a_loss.item(), critic=c_loss, E_final=E, tau=tau)

    os.makedirs(Path(a.out).parent, exist_ok=True)
    torch.save({
        'kind': 'pwm', 'sd': actor.cpu().state_dict(),
        'z_dim': c_act.latent_dim, 'a_dim': a_dim, 'horizon': a.horizon,
        'gamma': a.gamma, 'amax': a.amax, 'width': a.width, 'layers': a.layers,
        'objective': 'dense' if a.dense else 'terminal',
        'value_path': str(a.out_value), 'wm': a.wm, 'seed': a.seed,
    }, a.out)
    save_metric(teacher.cpu(), 'td', c_td.latent_dim * vframes, arch, a.out_value)
    logging.success(f'saved PWM actor -> {a.out} (teacher -> {a.out_value})')


if __name__ == '__main__':
    main()
