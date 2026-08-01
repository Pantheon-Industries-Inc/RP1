"""Deploy-time solver for the reactive PWM-style policy.

Deliberately separate from :mod:`stable_worldmodel.solver.lip`: LIPSolver is an
iterative *refiner* -- it proposes a whole H-block plan and improves it over K
passes -- whereas this is a reactive mapping ``pi(z, z_g) -> action block`` with
no plan state, no refinement and no search. Sharing a file would tangle two
different control schemes; nothing here imports from ``lip``.

Trained by ``scripts/plan/train_pwm_ac.py`` (actor-critic on the MRN
quasimetric; see that file for the objective and for what is and is not taken
from PWM, arXiv 2407.02466).

Control scheme and its cost
---------------------------
``solve`` must return a plan of ``horizon`` action blocks, because the
receding-horizon MPC in :class:`WorldModelPolicy` executes
``receding_horizon`` of them before replanning. Two regimes follow, and both are
supported by the same code:

* ``plan_config.receding_horizon == 1`` -- **pure reactive**. Only the first
  block is ever executed, so only the first is needed: one encoder pass and one
  MLP pass, and **zero world-model rollouts**. The policy re-observes reality
  every action block.
* ``receding_horizon == horizon`` (the campaign's protocol, 5 blocks = 25
  primitive steps, matching the LeWM paper's "entire optimized action sequence
  is executed before replanning") -- the remaining blocks have to come from
  somewhere, so the policy is rolled forward *in imagination*: H policy passes
  and H world-model steps. That is 5 rollouts per planning step against LIPv4's
  8 and CEM's 3,000 (paper's 10 iterations) to 9,000 (30 iterations).

``--fill`` chooses how the tail is produced when more than one block is
consumed: ``imagine`` (default, autoregressive through the world model) or
``repeat`` (hold the first block, zero rollouts but open loop for 25 steps).
Which regime you report matters and should be stated: the reactive policy is
*defined* by acting on the current observation, so replanning every block is its
natural setting, and it is also its cheapest.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging
from torch import nn

__all__ = ['PWMActor', 'PWMSolver']


class PWMActor(nn.Module):
    """``pi(z, z_g) -> action block``. Must match ``train_pwm_ac.PWMActor``.

    Goal enters as a displacement ``proj(z_g - z)``. Note this is a *different*
    input interface from the LIPv4 refiner we compare against: that one is
    ``PlannerNet`` with ``use_zg=use_z0=False``, so it consumes only
    ``[A, grad_A V, E]`` and no latents at all -- the goal reaches it through
    the critic. A single-pass policy has no value or gradient to read, so it
    must consume the latents directly. ``tanh * amax`` does reproduce the
    action box the LIPv4 actors clip to.
    """

    def __init__(self, z_dim: int, a_dim: int, width: int = 512, layers: int = 3,
                 amax: float = 2.2, zproj: int = 256) -> None:
        super().__init__()
        self.amax = float(amax)
        self.zp = nn.Linear(z_dim, zproj)
        self.gp = nn.Linear(z_dim, zproj)
        net: list[nn.Module] = [nn.Linear(2 * zproj, width), nn.SiLU()]
        for _ in range(layers - 1):
            net += [nn.Linear(width, width), nn.SiLU()]
        net += [nn.Linear(width, a_dim)]
        self.net = nn.Sequential(*net)

    def forward(self, z: torch.Tensor, z_g: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.zp(z), self.gp(z_g - z)], dim=-1)
        return torch.tanh(self.net(h)) * self.amax


class PWMSolver:
    """Reactive policy as a Solver. No sampling, no refinement, no plan state.

    Args:
        model: the world model (Costable). Used only to encode, and to imagine
            the plan tail when ``fill='imagine'``.
        actor_path: checkpoint written by ``train_pwm_ac.py`` (``kind='pwm'``).
        fill: ``'imagine'`` (autoregressive through the WM) or ``'repeat'``
            (hold the first block; zero rollouts, open loop).
        batch_size: environments per forward pass.
    """

    def __init__(self, model, actor_path: str = '', fill: str = 'imagine',
                 batch_size: int = 1, device: str | torch.device = 'cuda',
                 seed: int = 0, **_ignored) -> None:
        if not actor_path:
            raise ValueError('PWMSolver needs solver.actor_path=<pwm actor .pt>')
        if fill not in ('imagine', 'repeat'):
            raise ValueError(f"fill must be 'imagine' or 'repeat', got {fill!r}")
        self.model = model
        self.fill = fill
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed)

        ck = torch.load(actor_path, map_location=device, weights_only=False)
        if ck.get('kind') != 'pwm':
            raise ValueError(
                f"{actor_path} is kind={ck.get('kind')!r}, not 'pwm'. "
                'LIPv4 refiners load with solver=lip, not solver=pwm.')
        self.actor = PWMActor(ck['z_dim'], ck['a_dim'], ck.get('width', 512),
                              ck.get('layers', 3), ck.get('amax', 2.2)).to(device)
        self.actor.load_state_dict(ck['sd'])
        self.actor.eval()
        self.actor.requires_grad_(False)
        self._a_dim_ck = int(ck['a_dim'])
        self._train_h = int(ck.get('horizon', 5))
        logging.info(
            f"[pwm] reactive policy z_dim={ck['z_dim']} a_dim={self._a_dim_ck} "
            f"amax={ck.get('amax')} objective={ck.get('objective')} "
            f"trained H={self._train_h} | fill={fill}")

    # -- Solver protocol ---------------------------------------------------
    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        assert isinstance(action_space, Box), 'PWMSolver expects a Box action space'
        self._n_envs = int(n_envs)
        self._horizon = int(config.horizon)
        self._action_dim = int(np.prod(action_space.shape[1:])) * int(config.action_block)
        if self._action_dim != self._a_dim_ck:
            raise ValueError(
                f'action_dim {self._action_dim} != actor a_dim {self._a_dim_ck}; '
                'the actor was trained for a different action-block size')
        self._recede = int(getattr(config, 'receding_horizon', self._horizon))

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def horizon(self) -> int:
        return self._horizon

    # -- encoding ----------------------------------------------------------
    def _base(self):
        """Unwrap the metric-cost wrapper if the eval path installed one."""
        m = self.model
        return getattr(m, 'base', m)

    @torch.no_grad()
    def _encode(self, info_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (z_now, z_goal), both (B, D). Mirrors the LIP encode conventions."""
        wm = self._base()
        px = info_dict['pixels'].to(self.device)
        enc_in = {'pixels': px}
        if getattr(wm, 'wants_proprio', False):
            pro = info_dict.get('proprio')
            if pro is None:
                raise KeyError("proprio-variant WM: info_dict lacks 'proprio'")
            pro = torch.as_tensor(np.asarray(pro), dtype=torch.float32,
                                  device=self.device)
            enc_in['proprio'] = pro.reshape(px.shape[0], px.shape[1], -1)
        z_now = wm.encode(enc_in)['emb'][:, -1].float()

        gx = info_dict['goal'].to(self.device)
        genc: dict = {'pixels': gx}
        if getattr(wm, 'wants_proprio', False):
            # tolerate either key: the canonical h5 has no 'state' column, so
            # the goal proprio arrives as 'goal_proprio' there and as
            # 'goal_state' under the play-data config
            gpro = next((info_dict[k] for k in ('goal_proprio', 'goal_state')
                         if info_dict.get(k) is not None), None)
            if gpro is None:
                raise KeyError("proprio-variant WM: info_dict lacks "
                               "'goal_proprio'/'goal_state'")
            gpro = torch.as_tensor(np.asarray(gpro), dtype=torch.float32,
                                   device=self.device)
            gpro = gpro.reshape(gx.shape[0], -1)[:, -2:]
            genc['proprio'] = gpro.unsqueeze(1).expand(-1, gx.shape[1], -1)
        z_goal = wm.encode(genc)['emb'][:, -1].float()
        return z_now, z_goal

    # -- solve -------------------------------------------------------------
    def __call__(self, *args, **kwargs) -> dict:
        # WorldModelPolicy invokes solvers as callables; every concrete solver
        # supplies this shim itself (the Solver Protocol does not)
        return self.solve(*args, **kwargs)

    @torch.no_grad()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        t0 = time.time()
        z, zg = self._encode(info_dict)
        B = z.shape[0]

        # first block: the only one a purely reactive controller ever needs
        blocks = [self.actor(z, zg)]

        # how many blocks will actually be executed before the next replan
        need = min(self._recede, self._horizon)
        if need > 1:
            if self.fill == 'repeat':
                blocks += [blocks[0]] * (self._horizon - 1)
            else:
                wm = self._base()
                hs = int(getattr(wm, 'history_size', 3))
                z_hist = z.unsqueeze(1).expand(-1, hs, -1).contiguous()
                a_hist = torch.zeros(B, max(hs - 1, 1), self._action_dim,
                                     device=self.device)
                from stable_worldmodel.solver.lip import rollout_traj
                for _ in range(self._horizon - 1):
                    # committed rollout_traj is (wm, z_hist, a_hist, plan);
                    # its attention window is a fixed 3 frames
                    traj = rollout_traj(wm, z_hist, a_hist,
                                        torch.stack(blocks, 1))
                    blocks.append(self.actor(traj[:, -1], zg))
        else:
            # pad the unused tail so the returned shape matches the contract
            blocks += [torch.zeros_like(blocks[0])] * (self._horizon - 1)

        plan = torch.stack(blocks, dim=1)          # (B, H, a_dim)
        return {
            # key is 'actions' (plural) and must be CPU: the policy slices it
            # into a CPU warm-start buffer, exactly as it consumes CEM's output
            'actions': plan.detach().cpu(),
            'costs': [float('nan')] * B,           # no cost is evaluated
            'mean': [], 'var': [],
            'time': time.time() - t0,
        }
