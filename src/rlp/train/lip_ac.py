"""LIP-AC: train the TD value (critic) and the LIP planner (actor) in tandem.

The sequential pipeline (metric.py -> select TD by CEM -> lip.py)
selects the value by a zeroth-order criterion — CEM only needs correct *ranking*
of sampled plans — but LIP consumes the value's *gradient field* through the WM,
so the CEM-best TD need not be the best LIP teacher (observed on PushT, where a
rugged plan landscape reversed the method ordering). Here the two train jointly,
DDPG/TD3-style, with the planner as a K-step learned-optimizer actor:

  critic   d_phi(z, z_g): n-step expectile TD on the dense (fs1) latent cache.
           This TD loss is the ONLY gradient that reaches phi — the actor loss
           is computed through the frozen-parameter teacher, so the critic can
           never learn to "make plans look good" (the classic collapse).
  teacher  d_bar = EMA(d_phi) (Polyak, ``ema_tau``): serves TD bootstrap targets
           AND the actor (gradient features, refinement loss, and what we save
           for deployment — so training matches LIPSolver's plan-time value).
  actor    f_theta: exactly lip.py's objective, against the teacher.

Schedule: ``pretrain`` critic-only warmup (auto: 2000 if fresh, 0 if warm-started
via ``init_value``), then ``critic_ratio`` critic steps per actor step, critic+EMA
frozen after ``freeze_critic_frac`` of actor steps so the planner settles on a
stationary teacher (LIP is lr-sensitive; a moving target late in training makes
it chase noise).

Optional feedback loop (``expand_weight > 0``): the actor's own imagined terminal
latents become extra TD backups  d(z0,zg) <- H*fs + d_bar(z_T,zg)  — value
expansion on planner-visited states, shaping d exactly where the planner
travels. Low expectile makes this a one-sided (optimistic) bound absorber:
good plans tighten d, bad plans barely raise it. Off by default: it lets the
pair co-exploit WM errors, so compare against expand-weight 0 before trusting.

Outputs are written to the run's ``checkpoints/`` directory. The planner
checkpoint records the value-checkpoint path, so ``rlp.eval.world_model`` with
``solver=lip`` works unchanged and the same value can be CEM-evaluated for an
apples-to-apples comparison.
"""

import copy
from contextlib import suppress
from pathlib import Path

import h5py

with suppress(ImportError):
    import hdf5plugin  # noqa: F401  (registers HDF5 compression filters, e.g. cube h5)
from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlp.config import dispatch, run_hydra
from rlp.core.planner import PlannerNet, PlannerNetRec, PlannerNetV3
from rlp.core.rollout import rollout_traj
from rlp.core.solver.lip import ValueFunction
from rlp.core.value import load_metric
from rlp.core.value.io import build_metric, save_metric
from rlp.core.value.learners.td import _expectile_loss
from rlp.core.value.samplers import NStepGoalSampler
from rlp.core.world_model import load_pretrained
from rlp.core.world_model.protocols import LatentWorldModel
from rlp.data import LatentCache
from rlp.logging import logger

from .utils import cosine_interpolate

type ExpandBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _run(cfg: DictConfig) -> None:
    a = OmegaConf.merge(cfg, cfg.core.planner, cfg.core.value)
    if not isinstance(a, DictConfig):
        raise TypeError("merged planner/value configuration must be a mapping")
    a.embed_dim = a.embedding_dim
    aliases = {
        "arch": "architecture",
        "iters": "iterations",
        "amax": "action_limit",
        "s_dim": "recurrent_state_dim",
        "s0_mode": "recurrent_state_init",
        "rec_hidden": "recurrent_hidden_dim",
        "width": "transformer_width",
        "layers": "transformer_layers",
        "gd_init": "gradient_descent_init",
        "feat_norm": "feature_normalization",
        "iter_mode": "iteration_mode",
        "p0": "preconditioner_init",
        "pre_ln": "pre_layer_norm",
        "drop_zg": "drop_goal",
        "drop_z0": "drop_state",
        "drop_grad": "drop_gradient",
        "no_gate": "use_gate",
        "cond_mode": "conditioning_mode",
    }
    for old, new in aliases.items():
        a[old] = not a[new] if old == "no_gate" else a[new]
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    wm_module = load_pretrained(a.wm).to(dev).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)

    # ------------------------------------------------------------ actor data (fs5 + h5)
    c = LatentCache.load(a.cache)
    z = c.z.to(dev).float()
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    with h5py.File(a.h5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
        ep_len_h5 = h["ep_len"][:] if "ep_len" in h else None
    # nan-aware: some datasets (e.g. lewm-cube) pad episode-terminal steps with
    # NaN actions; those rows are never sampled (history blocks stop before the
    # terminal step) but must not poison the normalization stats
    amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    if a.action_stats_pin == "expert":
        # mixture actions z-scored with EXPERT stats so the actor operates in
        # the same action convention the eval un-scales with (eval-consistency).
        amu = np.array([0.010831, -0.003126, 0.002633, 0.000422, 0.15846])
        astd = np.array([0.2887, 0.392736, 0.641535, 0.391823, 0.249935])
        logger.info(f"LIP using fixed expert action statistics mean={amu.tolist()} std={astd.tolist()}")
    elif a.action_stats_pin and a.action_stats_pin.endswith(".json"):
        import json

        with Path(a.action_stats_pin).open() as f:
            _st = json.load(f)
        amu = np.asarray(_st["mean"], dtype=np.float64)
        astd = np.asarray(_st["std"], dtype=np.float64)
        assert amu.shape == astd.shape == (act.shape[-1],), (
            f"pinned stats dim {amu.shape} != action dim {act.shape[-1]}"
        )
        assert (astd > 1e-5).all(), f"pinned std degenerate: {astd.tolist()}"
        logger.info(f"LIP using action statistics from {a.action_stats_pin} mean={amu.tolist()} std={astd.tolist()}")
    elif a.action_stats_pin:
        raise ValueError(f"unknown action_stats_pin={a.action_stats_pin!r} (expected 'expert' or a *.json path)")
    act_n = ((act - amu) / astd).astype(np.float32)
    fs = 5  # primitive steps per action block
    a_dim = act.shape[-1] * fs

    def blocks(e: int, t: int) -> np.ndarray:
        offset = fs * t
        if ep_len_h5 is not None:
            offset = min(offset, max(0, int(ep_len_h5[e]) - fs))
        h0 = int(ep_off[e] + offset)
        return np.asarray(act_n[h0 : h0 + fs]).reshape(-1)

    def sample(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zh: list[torch.Tensor] = []
        ah: list[np.ndarray] = []
        zg: list[torch.Tensor] = []
        aref: list[np.ndarray] = []
        for _ in range(B):
            e = int(ep_ids[rng.integers(len(ep_ids))])
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
            # trust-region reference: the data's next-H real action blocks from t
            # (clamped at episode end) — where the WM knows the dynamics.
            # Clamp to L-2, the last FULL in-episode block: block L-1 starts at
            # the episode's final primitive step, so its 5-step read spills into
            # the next episode (silently wrong) and runs off the array end for
            # the dataset's last episode (ragged np.stack crash — hit on
            # puzzle 2026-07-23 with 1001-step episodes, e=999/t=197).
            aref.append(np.stack([blocks(e, min(t + k, L - 2)) for k in range(a.horizon)]))
            if rng.random() < a.p_cross:
                e2 = int(ep_ids[rng.integers(len(ep_ids))])
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, a.max_delta + 1))
                zg.append(z[rows[min(t + d, L - 1)]])
        return (
            torch.stack(zh),
            torch.from_numpy(np.stack(ah)).to(dev),
            torch.stack(zg),
            torch.from_numpy(np.stack(aref)).to(dev).float(),
        )

    # ------------------------------------------------------------ critic (fs1 cache)
    c_td = LatentCache.load(a.cache_td)
    if a.init_value:
        critic = load_metric(a.init_value, device=dev)
        assert critic.latent_dim == c_td.latent_dim, "init-value latent dim mismatch"
    else:
        critic = build_metric(
            "td",
            c_td.latent_dim,
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
        prm.requires_grad_(False)  # actor loss flows THROUGH, never INTO
    teacher.eval()
    critic_fn = cast(ValueFunction, critic)
    teacher_fn = cast(ValueFunction, teacher)

    td_sampler = NStepGoalSampler(
        c_td,
        n_step=a.n_step,
        p_cross=a.td_p_cross,
        n_buckets=10,
        balanced=True,
        seed=a.seed,
        max_delta=a.td_max_delta,
    )
    c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)

    n_plan = a.horizon * fs  # plan length in primitive steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan), 1.0
    else:
        plan_disc = a.gamma**n_plan
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
        if expand is not None:  # value expansion on planner rollouts
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

    # ------------------------------------------------------------ actor
    net: PlannerNet | PlannerNetRec | PlannerNetV3
    if a.arch == "traj":
        if (a.gd_init > 0 or a.feat_norm or a.cond_mode == "global") and a.goal_mode != "vonly":
            raise ValueError(
                "gradient_descent_init/feature_normalization/conditioning_mode=global need goal_mode=vonly"
            )
        net = PlannerNetV3(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            width=a.width,
            layers=a.layers,
            amax=a.amax,
            n_iters=a.iters,
            goal_mode=a.goal_mode,
            zero_init=a.zero_init,
            gd_init=a.gd_init,
            feat_norm=a.feat_norm,
            iter_mode=a.iter_mode,
            head_mode=a.head_mode,
            p0=a.p0,
            head_scale=a.head_scale,
            cond_mode=a.cond_mode,
            use_gate=not a.no_gate,
            pre_ln=a.pre_ln,
        ).to(dev)
    elif a.arch == "v4r":
        net = PlannerNetRec(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            hidden=a.rec_hidden,
            s_dim=a.s_dim,
            amax=a.amax,
            use_z0=False,
            use_zg=False,
            use_grad=not a.drop_grad,
            s0_mode=a.s0_mode,
            head_scale=a.head_scale,
        ).to(dev)
    elif a.arch == "v4":
        net = PlannerNet(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            feed=a.feed,
            amax=a.amax,
            use_zg=False,
            use_gate=False,
            use_z0=False,
            use_grad=not a.drop_grad,
            head_scale=a.head_scale,
        ).to(dev)
    else:
        net = PlannerNet(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            feed=a.feed,
            amax=a.amax,
            use_zg=not a.drop_zg,
            use_gate=not a.no_gate,
            use_z0=not a.drop_z0,
            use_grad=not a.drop_grad,
            head_scale=a.head_scale,
        ).to(dev)
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=1e-5)

    replay_buf: dict[str, torch.Tensor | None] = {
        "zh": None,
        "zg": None,
    }  # previous step's final imagined windows

    def actor_step() -> tuple[float, float, ExpandBatch]:
        zh, ah, zg, aref = sample(a.batch)
        replay_zh = replay_buf["zh"]
        replay_zg = replay_buf["zg"]
        if a.replay_prob > 0 and replay_zh is not None and replay_zg is not None:
            pick = torch.rand(a.batch, device=dev) < a.replay_prob
            idx = torch.nonzero(pick).squeeze(1)
            if idx.numel():
                take = torch.randint(0, replay_zh.shape[0], (idx.numel(),), device=dev)
                zh, ah, zg = zh.clone(), ah.clone(), zg.clone()
                zh[idx] = replay_zh[take]
                zg[idx] = replay_zg[take]
                ah[idx] = 0.0  # deployed replans query with zero action history
        z0 = zh[:, -1]
        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)
        s = net.init_state(a.batch, z0) if isinstance(net, PlannerNetRec) else None
        e_path: list[torch.Tensor] = []
        zT: torch.Tensor | None = None
        tr: torch.Tensor | None = None
        for k in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch 2.7 context-manager stub is untyped.
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(wm, zh, ah, A_in)
                (gA,) = torch.autograd.grad(teacher_fn(traj[:, -1], zg).sum(), A_in)
            # A_in holds A's values, so the grad pass's trajectory IS the feature
            # trajectory — reuse it instead of a third WM rollout (value-identical)
            traj_f = traj.detach()
            E_feat = teacher_fn(traj_f[:, -1], zg).detach()
            vtraj = None
            if a.goal_mode == "sep" or (a.goal_mode == "vonly" and a.cond_mode == "token"):
                vtraj = (
                    teacher_fn(
                        traj_f.reshape(-1, traj_f.shape[-1]),  # per-step V(z_t, zg)
                        zg.repeat_interleave(a.horizon, dim=0),
                    )
                    .view(a.batch, a.horizon)
                    .detach()
                )
            if isinstance(net, PlannerNetRec):
                if s is None:
                    raise RuntimeError("recurrent actor state is unavailable")
                A, s = net(A, gA.detach(), E_feat, z0, zg, s, traj_f, k=k, vtraj=vtraj)
            else:
                A = net(A, gA.detach(), E_feat, z0, zg, traj_f, k=k, vtraj=vtraj)
            tr = rollout_traj(wm, zh, ah, A)
            zT = tr[:, -1]
            e_path.append(teacher_fn(zT, zg).mean())
        if a.replay_prob > 0:
            if tr is None:
                raise RuntimeError("actor produced no rollout trajectory")
            replay_buf["zh"] = tr[:, -3:].detach()
            replay_buf["zg"] = zg.detach()
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        if a.bc_weight > 0:  # trust-region toward data actions
            loss = loss + a.bc_weight * ((A - aref) ** 2).mean()
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        if zT is None:
            raise RuntimeError("actor produced no terminal latent")
        expand = (z0.detach(), zT.detach(), zg.detach())
        return float(e_path[0].item()), float(e_path[-1].item()), expand

    # ------------------------------------------------------------ schedule
    pretrain = a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000)
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            logger.info(f"LIP-AC pretrain {i}/{pretrain}: td_loss={cl:.4f}")

    freeze_at = int(a.freeze_critic_frac * a.steps)
    expand = None
    for step in range(a.steps):
        critic_live = step < freeze_at
        if step == freeze_at:
            logger.info(f"LIP-AC step {step}: critic and teacher frozen for {a.steps - freeze_at} steps")
        # during-run schedules: teacher converges (lr decay) and sharpens
        # (expectile anneal) over the live phase; actor lr decays over all steps
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
                expand = None  # consume each actor batch once
        e_first, e_final, expand = actor_step()
        if step % 500 == 0:
            logger.info(
                f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}"
            )

    # ------------------------------------------------------------ save (teacher first;
    # the actor checkpoint references it, and it is what the actor optimized against)
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint
    value_checkpoint = save_metric(teacher.cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory)
    logger.success(f"Saved teacher value to {value_checkpoint}")
    net.eval()
    if a.arch == "traj":
        kind = "lip3"
    elif a.arch == "v4r":
        kind = "lip4r"
    elif a.arch == "v4":
        kind = "lip4"
    else:
        kind = "lip" if a.feed == "none" else "lip2"
    torch.save(
        {
            "kind": kind,
            "feed": a.feed,
            "sd": net.cpu().state_dict(),
            "z_dim": z.shape[-1],
            "horizon": a.horizon,
            "iters": a.iters,
            "a_dim": a_dim,
            "amax": a.amax,
            "width": a.width,
            "layers": a.layers,
            "s_dim": a.s_dim,
            "s0_mode": a.s0_mode,
            "hidden": a.rec_hidden,
            "goal_mode": a.goal_mode,
            "gd_init": a.gd_init,
            "feat_norm": a.feat_norm,
            "iter_mode": a.iter_mode,
            "head_mode": a.head_mode,
            "cond_mode": a.cond_mode,
            "use_gate": getattr(net, "use_gate", not a.no_gate),
            "use_zg": getattr(net, "use_zg", not a.drop_zg),
            "use_z0": getattr(net, "use_z0", not a.drop_z0),
            "use_grad": getattr(net, "use_grad", not a.drop_grad),
            "head_scale": a.head_scale,
            "pre_ln": a.pre_ln,
            "value": str(value_checkpoint),
        },
        planner_checkpoint,
    )
    logger.success(f"Saved learned planner to {planner_checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/lip_ac")


if __name__ == "__main__":
    main()
