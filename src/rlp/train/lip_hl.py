"""Tandem HIGH-LEVEL LIPv4 trainer: TD critic (dense fs1 cache, env-step
units) + PlannerNet actor over latent MACRO-actions imagined through a frozen
HWM high level (train_hwm.py checkpoint).

Adaptation of lip_ac.py (the schedamax-6k recipe) with:
- actions = macro-actions (a_dim = hwm.macro_dim, raw LayerNorm'd space, no
  z-scoring; ``core.planner.action_limit`` clamps it, approximately the paper's
  2/98-percentile bounds),
- imagination through the HWM F2 (1 step = hwm stride K env steps),
- solver-convention contexts everywhere: z-history = z0 repeated 3x, ZERO
  macro history (matches HWM training and the stateless HLIP solver exactly;
  no real waypoint history exists at plan time),
- goals: in-episode waypoint at t0 + d*K env steps (d in [1, max-delta]) or
  cross-episode with probability ``p_cross``,
- critic identical to lip_ac.py: n-step expectile-Huber quasimetric TD
  on the DENSE fs1 cache (td-cache-stride lesson: never train the value on a
  coarse cache), warm-started from ``init_value``, EMA teacher.

Checkpoint is a standard kind='lip4' actor + extra keys (hwm path, macro_dim,
hl_stride) consumed by HLIPSolver.
"""

import copy
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlp.config import dispatch, run_hydra
from rlp.core.planner import PlannerNet
from rlp.core.rollout import rollout_traj
from rlp.core.solver.lip import ValueFunction
from rlp.core.value import load_metric
from rlp.core.value.io import build_metric, save_metric
from rlp.core.value.learners.td import _expectile_loss
from rlp.core.value.samplers import NStepGoalSampler
from rlp.core.world_model.hwm import load_hwm
from rlp.data import LatentCache
from rlp.logging import logger

from .utils import cosine_interpolate

type ExpandBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _run(cfg: DictConfig) -> None:
    a = OmegaConf.merge(cfg, cfg.core.planner, cfg.core.value)
    a.embed_dim = a.embedding_dim
    a.iters = a.iterations
    a.amax = a.action_limit

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    # ------------------------------------------------------------ frozen high level
    hwm, blob = load_hwm(a.hwm, device=dev)
    K = blob["cfg"]["stride"]
    macro_dim = hwm.macro_dim
    logger.info(
        f"HWM {a.hwm}: stride {K} env steps/macro, macro_dim {macro_dim}, "
        f"p2 {[round(v, 2) for v in blob['l_p2'].tolist()]} "
        f"p98 {[round(v, 2) for v in blob['l_p98'].tolist()]} (amax {a.amax})"
    )

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
    logger.info(f"Actor episodes={len(ep_ids)} T={T_ep}")

    def sample(B: int) -> tuple[torch.Tensor, torch.Tensor]:
        zh: list[torch.Tensor] = []
        zg: list[torch.Tensor] = []
        for _ in range(B):
            e = int(ep_ids[rng.integers(len(ep_ids))])
            rows = ep_rows[e]
            t0 = int(rng.integers(0, T_ep - K))
            zh.append(z[rows[t0]])
            if rng.random() < a.p_cross:
                e2 = int(ep_ids[rng.integers(len(ep_ids))])
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t0 + d * K, T_ep - 1)]])
        z0 = torch.stack(zh)
        return z0, torch.stack(zg)

    # ------------------------------------------------------------ critic (same fs1 cache)
    if a.init_value:
        critic = load_metric(a.init_value, device=dev)
        assert critic.latent_dim == c.latent_dim, "init-value latent dim mismatch"
    else:
        critic = build_metric(
            "td",
            c.latent_dim,
            {
                "head": a.head,
                "hidden_dim": a.hidden_dim,
                "depth": a.depth,
                "embed_dim": a.embed_dim,
                "softplus": True,
                "symmetric": False,
            },
        ).to(dev)
    critic.train()
    teacher = copy.deepcopy(critic).to(dev)
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    teacher.eval()
    critic_fn = cast(ValueFunction, critic)
    teacher_fn = cast(ValueFunction, teacher)

    td_sampler = NStepGoalSampler(
        c,
        n_step=a.n_step,
        p_cross=a.td_p_cross,
        n_buckets=10,
        balanced=True,
        seed=a.seed,
        max_delta=a.td_max_delta,
    )
    c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)

    n_plan_env = a.horizon * K  # plan span in env steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan_env), 1.0
    else:
        plan_disc = a.gamma**n_plan_env
        plan_cost = (1.0 - plan_disc) / (1.0 - a.gamma)

    def critic_step(expand: ExpandBatch | None = None, tau: float | None = None, lr: float | None = None) -> float:
        tau = a.expectile if tau is None else tau
        if lr is not None:
            for pg in c_opt.param_groups:
                pg["lr"] = lr
        b = td_sampler.sample(a.td_batch)
        z_t, z_tn, z_g = b["z_t"].to(dev), b["z_tn"].to(dev), b["z_g"].to(dev)
        ne, reached, dist = (
            b["n_eff"].to(dev),
            b["reached"].to(dev),
            b["dist"].to(dev),
        )
        with torch.no_grad():
            d_next = teacher_fn(z_tn, z_g)
            if a.gamma >= 1.0:
                cost, disc = ne, torch.ones_like(ne)
            else:
                disc = a.gamma**ne
                cost = (1.0 - disc) / (1.0 - a.gamma)
            tgt = reached * dist + (1.0 - reached) * (cost + disc * d_next)
        loss = _expectile_loss(critic_fn(z_t, z_g) - tgt, tau, a.huber_beta)
        if expand is not None:
            z0e, zTe, zge = expand
            with torch.no_grad():
                tgt_e = plan_cost + plan_disc * teacher_fn(zTe, zge)
            loss = loss + a.expand_weight * _expectile_loss(critic_fn(z0e, zge) - tgt_e, tau, a.huber_beta)
        c_opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature here.
        c_opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), critic.parameters(), strict=True):
                tp.mul_(1.0 - a.ema_tau).add_(a.ema_tau * sp)
        return float(loss.item())

    # ------------------------------------------------------------ actor (v4)
    net = PlannerNet(
        z.shape[-1],
        horizon=a.horizon,
        a_dim=macro_dim,
        amax=a.amax,
        use_zg=False,
        use_gate=False,
        use_z0=False,
        use_grad=True,
    ).to(dev)
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=1e-5)

    def actor_step() -> tuple[float, float, ExpandBatch]:
        z0, zg = sample(a.batch)
        zh = z0.unsqueeze(1).expand(-1, 3, -1)  # solver convention
        ah = torch.zeros(a.batch, 2, macro_dim, device=dev)
        A = torch.zeros(a.batch, a.horizon, macro_dim, device=dev)
        e_path, zT = [], None
        for k in range(a.iters):
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(hwm, zh, ah, A_in)
                (gA,) = torch.autograd.grad(teacher_fn(traj[:, -1], zg).sum(), A_in)
            traj_f = traj.detach()
            E_feat = teacher_fn(traj_f[:, -1], zg).detach()
            A = net(A, gA.detach(), E_feat, z0, zg, traj_f, k=k)
            tr = rollout_traj(hwm, zh, ah, A)
            zT = tr[:, -1]
            e_path.append(teacher_fn(zT, zg).mean())
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        if zT is None:
            raise RuntimeError("actor produced no terminal latent")
        expand = (z0.detach(), zT.detach(), zg.detach())
        return float(e_path[0].item()), float(e_path[-1].item()), expand

    # ------------------------------------------------------------ schedule (lip_ac parity)
    pretrain = a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000)
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            logger.info(f"HLIP pretrain {i}/{pretrain}: td_loss={cl:.4f}")

    freeze_at = int(a.freeze_critic_frac * a.steps)
    expand = None
    for step in range(a.steps):
        critic_live = step < freeze_at
        if step == freeze_at:
            logger.info(f"HLIP step {step}: critic and teacher frozen")
        tau_s = (
            a.expectile
            if a.expectile_final is None
            else (a.expectile + (a.expectile_final - a.expectile) * min(step, freeze_at) / max(freeze_at, 1))
        )
        clr_s = cosine_interpolate(a.critic_lr, a.critic_lr_final, step, freeze_at)
        alr_s = cosine_interpolate(a.actor_lr, a.actor_lr_final, step, a.steps)
        for pg in a_opt.param_groups:
            pg["lr"] = alr_s
        cl = float("nan")
        if critic_live:
            for _ in range(a.critic_ratio):
                cl = critic_step(
                    expand if a.expand_weight > 0 else None,
                    tau=tau_s,
                    lr=clr_s,
                )
                expand = None
        e_first, e_final, expand = actor_step()
        if step % 500 == 0:
            logger.info(
                f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                f"td_loss {cl:.4f} tau {tau_s:.3f} alr {alr_s:.2e}"
            )

    # ------------------------------------------------------------ save
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint
    value_checkpoint = save_metric(teacher.cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory)
    logger.success(f"Saved teacher value to {value_checkpoint}")
    net.eval()
    torch.save(
        {
            "kind": "lip4",
            "feed": "none",
            "sd": net.cpu().state_dict(),
            "z_dim": z.shape[-1],
            "horizon": a.horizon,
            "iters": a.iters,
            "a_dim": macro_dim,
            "amax": a.amax,
            "use_gate": False,
            "use_zg": False,
            "use_z0": False,
            "use_grad": True,
            "value": str(value_checkpoint),
            "hwm": a.hwm,
            "hl_stride": K,
            "macro_dim": macro_dim,
        },
        planner_checkpoint,
    )
    logger.success(f"Saved HL learned planner to {planner_checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/lip_hl")


if __name__ == "__main__":
    main()
