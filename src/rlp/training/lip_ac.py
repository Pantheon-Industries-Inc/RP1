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

``expand_traj`` additionally reuses the actor's final refinement rollout as H
single-block backups  d(z_t,zg) <- fs + d_bar(z_{t+1},zg)  along the imagined
path (consecutive latents are one fs-step action block apart). The rollout is
already computed for the actor loss, so this densifies the critic's expansion
signal H-fold per actor batch without any extra world-model queries. Same
co-exploitation caveat as the endpoint term; requires ``expand_weight > 0``.

Outputs are written to the run's ``checkpoints/`` directory. The planner
checkpoint records the value-checkpoint path, so ``rlp.inference.evaluate`` with
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
from omegaconf import DictConfig

from rlp.core.planner import PlannerNet, PlannerNetRec, PlannerNetV3
from rlp.core.rollout import rollout_traj
from rlp.core.solver.lip import ValueFunction
from rlp.core.temporal import trajectory_value, window_pair, windowed_trajectory_value
from rlp.core.value import load_metric
from rlp.core.value.io import build_metric, save_metric
from rlp.core.value.learners.td import _expectile_loss
from rlp.core.value.samplers import NStepGoalSampler
from rlp.core.world_model import load_pretrained
from rlp.core.world_model.protocols import LatentWorldModel
from rlp.core.world_model.runtime import pick_device
from rlp.data import LatentCache
from rlp.utils.config import dispatch, phase_config, run_hydra
from rlp.utils.logging import logger

from .utils import cosine_interpolate

# (z0, imagined trajectory (B, H, D), zg) — the endpoint backup uses traj[:, -1];
# expand_traj additionally consumes every consecutive pair along the trajectory.
# With a windowed value (vframes > 1) the tuple instead carries pre-stacked
# (start window, terminal window, tiled goal), all 2-D.
type ExpandBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]



def _band_offset(rng, hi: int, bands=None) -> int:
    """Hindsight-goal offset in fs5 blocks. Legacy: uniform 1..hi. With ``bands``
    (one entry per deployment horizon, in blocks, e.g. [6, 21] for h25+h100)
    a band is drawn uniformly first and the offset uniformly within
    1..min(band, hi): every deployment horizon gets equal mass, so ONE actor
    serves h25 and h100 without the short horizon being 4x under-weighted.
    Equal-width buckets over 1..hi would NOT do this -- that is uniform again."""
    if hi < 1:
        return 1
    if not bands:
        return int(rng.integers(1, hi + 1))
    b = int(bands[int(rng.integers(len(bands)))])
    return int(rng.integers(1, min(b, hi) + 1))


def _parse_bands(v):
    if v is None or v is False or str(v).strip() in ("", "None", "null", "false"):
        return None
    return [int(float(x)) for x in str(v).replace(";", ",").split(",") if x.strip()]

def _run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "training", cfg.core.planner, cfg.core.value)
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
        "zproj": "projection_dim",
        "vscale": "value_scale",
        "pre_ln": "pre_layer_norm",
        "drop_zg": "drop_goal",
        "drop_z0": "drop_state",
        "drop_grad": "drop_gradient",
        "no_gate": "use_gate",
        "cond_mode": "conditioning_mode",
    }
    for old, new in aliases.items():
        a[old] = not a[new] if old == "no_gate" else a[new]
    if a.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
        raise ValueError(f"unsupported temporal objective: {a.temporal_objective}")
    if a.actor_only and not a.init_value:
        raise ValueError("actor_only=true requires init_value")
    if not a.actor_only and not a.cache_td:
        raise ValueError("cache_td is required unless actor_only=true")
    dev = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    wm_module = load_pretrained(a.wm).to(dev).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)

    c = LatentCache.load(a.cache, mmap=bool(a.cache_mmap))
    cap = a.get("max_episodes")
    if cap:
        # ids are e*P+k under phase multiplexing, so cap in ORIGINAL episodes
        c = c.first_episodes(int(cap) * c.phase_multiplex)
        logger.info(f"Data-volume cap: actor training on episodes [0, {int(cap)})")
    z = c.z.to(dev).float()
    eps = c.episodes()
    keys = [k for k in eps if len(eps[k]) > a.max_delta + 4]
    if not keys:
        longest = max((len(v) for v in eps.values()), default=0)
        raise ValueError(
            f"no episode in {a.cache} is longer than max_delta+4 = {a.max_delta + 4} blocks "
            f"(longest is {longest}); lower planner.max_delta or use a cache with longer episodes"
        )
    ep_rows = {e: np.asarray(eps[e]) for e in keys}
    ep_ids = np.array(keys)
    BANDS = _parse_bands(a.get("band_mix", None))
    if BANDS and max(BANDS) > a.max_delta:
        raise ValueError(f"band_mix {BANDS} exceeds max_delta {a.max_delta}")
    logger.info(
        f"actor goal band: max_delta={a.max_delta} blocks, "
        f"{'MIXTURE over deployment bands ' + str(BANDS) if BANDS else 'uniform'}, p_cross={a.p_cross}; "
        f"episodes kept {len(keys)}/{len(eps)} (len > max_delta+4)"
    )
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
    fs = a.frameskip
    a_dim = act.shape[-1] * fs

    # phase-multiplexed fs5 caches (tools/subsample_cache phases>1) store each
    # residue class of the stride as its own episode, id = e * P + k; the h5
    # is indexed by the ORIGINAL episode, and block t of phase k starts at
    # primitive step k + fs*t. Hand this trainer an fs1 cache instead and it
    # would read past the episode (h0 = ep_off + fs*t assumes block indices).
    phase_mult = c.phase_multiplex
    if phase_mult > 1:
        logger.info(f"LIP-AC actor cache is phase-multiplexed x{phase_mult} (every residue class of the stride)")

    def block_row(e: int, t: int) -> int:
        """h5 row of block ``t``'s first primitive step in (phase-multiplexed) episode ``e``."""
        src_e, phase = divmod(e, phase_mult) if phase_mult > 1 else (e, 0)
        offset = phase + fs * t
        if ep_len_h5 is not None:
            offset = min(offset, max(0, int(ep_len_h5[src_e]) - fs))
        return int(ep_off[src_e] + offset)

    def blocks(e: int, t: int) -> np.ndarray:
        h0 = block_row(e, t)
        return np.asarray(act_n[h0 : h0 + fs]).reshape(-1)

    def sample(B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """History windows, action history, goals -- and, for the grounding term, the
        real agent position (px) at the plan's start and the commanded displacement
        (px) of the primitive step before it (zeros when grounding is off)."""
        zh: list[torch.Tensor] = []
        ah: list[np.ndarray] = []
        zg: list[torch.Tensor] = []
        ag0 = np.zeros((B, 2), dtype=np.float32)
        u_prev = np.zeros((B, act.shape[-1]), dtype=np.float32)
        for i in range(B):
            e = int(ep_ids[rng.integers(len(ep_ids))])
            rows = ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            zh.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            ah.append(np.stack([blocks(e, t - 2), blocks(e, t - 1)]))
            if state_all is not None:
                h0 = block_row(e, t)
                ag0[i] = state_all[h0, :2]
                src_e = e // phase_mult if phase_mult > 1 else e
                if h0 - 1 >= int(ep_off[src_e]):
                    u_prev[i] = np.nan_to_num(act[h0 - 1]) * ACTION_SCALE
            if rng.random() < a.p_cross:
                e2 = int(ep_ids[rng.integers(len(ep_ids))])
                r2 = ep_rows[e2]
                zg.append(z[r2[rng.integers(len(r2))]])
            else:
                d = _band_offset(rng, min(a.max_delta, L - 1 - t) if BANDS else a.max_delta, BANDS)
                zg.append(z[rows[min(t + d, L - 1)]])
        return (
            torch.stack(zh),
            torch.from_numpy(np.stack(ah)).to(dev),
            torch.stack(zg),
            torch.from_numpy(ag0).to(dev),
            torch.from_numpy(u_prev).to(dev),
        )

    c_td = None if a.actor_only else LatentCache.load(a.cache_td, mmap=bool(a.cache_mmap))
    # ------------------------------------------------------------ physics grounding (PushT)
    # Calibrated on the UNCAPPED fs1 cache plus the dataset's state column, before
    # any data-volume cap: the probe and thresholds describe the encoder and the
    # environment, not the ablation. The module travels in the actor checkpoint so
    # the deployed energy is the trained one.
    grounding: GroundingPenalty | None = None
    state_all: np.ndarray | None = None
    if a.get("grounding"):
        if a.grounding != "pusht":
            raise ValueError(f"unsupported grounding {a.grounding!r} (expected 'pusht')")
        if c_td is None:
            raise ValueError("grounding needs the dense fs1 cache (cache_td) to fit the state probe")
        if not a.get("state_h5"):
            raise ValueError("grounding=pusht needs state_h5: the dataset h5 with its 'state' column")
        n_rows = int(c_td.z.shape[0])
        with h5py.File(a.state_h5, "r") as hs:
            st_epi = np.asarray(hs["episode_idx"][:n_rows]).reshape(-1)
            if len(st_epi) != n_rows or not np.array_equal(st_epi, c_td.episode_idx.numpy()):
                raise RuntimeError(f"{a.state_h5} rows do not align with the fs1 cache {a.cache_td}")
            state_all = np.asarray(hs["state"][:], dtype=np.float32)
        if state_all.shape[0] < act.shape[0]:
            raise RuntimeError(
                f"{a.state_h5} has fewer rows ({state_all.shape[0]}) than the action h5 ({act.shape[0]})"
            )
        grounding, ground_report = calibrate_pusht_grounding(
            c_td.z,
            state_all[:n_rows],
            act[:n_rows],
            c_td.episode_idx.numpy(),
            amu=amu,
            astd=astd,
            fs=fs,
            horizon=a.horizon,
            margin=a.get("ground_margin"),
            tau=float(a.ground_tau),
            ref=float(a.ground_ref),
            weight=float(a.ground_weight),
            deadzone=a.get("ground_deadzone"),
            quantile=float(a.ground_quantile),
            seed=int(a.seed),
        )
        grounding = grounding.to(dev)
        logger.info(
            "LIP-AC physics grounding calibrated: " + " ".join(f"{k}={v:.4g}" for k, v in ground_report.items())
        )
    if cap and c_td is not None:
        c_td = c_td.first_episodes(int(cap))
    td_near_frac = float(a.get("near_frac", 0.0) or 0.0)
    td_near_max = int(a.get("near_max", 3) or 3)
    if td_near_frac > 0:
        logger.info(f"LIP-AC co-critic near-goal oversampling: frac={td_near_frac} max={td_near_max} steps")
    base_dim = int(z.shape[-1] if c_td is None else c_td.latent_dim)
    if a.init_value:
        critic = load_metric(a.init_value, device=dev)
        value_dim = int(cast(int, critic.latent_dim))
        if value_dim % base_dim:
            raise ValueError(f"init-value width {value_dim} is not an integer multiple of cache latent dim {base_dim}")
        # m-frame window values declare latent_dim = m * D; any integer
        # multiple >= 1 is valid (do not restrict to the paper's {1, 3}).
        vframes = value_dim // base_dim
    else:
        if c_td is None:
            raise RuntimeError("critic cache unavailable")
        vframes = 1
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
    value_window_lag = None
    if vframes > 1:
        value_window_lag = fs if a.window_lag is None else int(a.window_lag)
        if value_window_lag != fs:
            raise ValueError(
                f"window lag {value_window_lag} != action block {fs}: consecutive imagined "
                "latents are one action block apart, so the deployed window would not match "
                "the trained one"
            )
        if a.expand_traj:
            raise ValueError("expand_traj is not supported with a windowed init_value (vframes > 1)")
        if a.goal_mode == "sep" or (a.goal_mode == "vonly" and a.cond_mode == "token"):
            raise ValueError("windowed init_value does not support per-step value conditioning (goal_mode=sep/vonly)")
        logger.info(f"LIP-AC consuming a {vframes}-frame window value (lag {value_window_lag})")
    critic.train()
    teacher = copy.deepcopy(critic).to(dev)
    for prm in teacher.parameters():
        prm.requires_grad_(False)  # actor loss flows THROUGH, never INTO
    teacher.eval()
    critic_fn = cast(ValueFunction, critic)
    teacher_fn = cast(ValueFunction, teacher)

    td_sampler = (
        None
        if c_td is None
        else NStepGoalSampler(
            c_td,
            n_step=a.n_step,
            p_cross=a.td_p_cross,
            n_buckets=a.td_n_buckets,
            balanced=True,
            seed=a.seed,
            max_delta=a.td_max_delta,
            near_frac=td_near_frac,
            near_max=td_near_max,
        )
    )
    # a parameter-free critic (learner=l2: the latent distance itself) can only
    # ever be frozen, so it gets no optimizer and no co-training phase
    critic_trainable = any(True for _ in critic.parameters())
    if not critic_trainable and not a.actor_only and a.freeze_critic_frac > 0:
        raise ValueError(
            f"critic {type(critic).__name__} has no parameters, so it cannot be co-trained; "
            "set planner.freeze_critic_frac=0"
        )
    c_opt = (
        None
        if a.actor_only or not critic_trainable
        else torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)
    )

    def _window_rows(indices: torch.Tensor, source: LatentCache | None = None) -> torch.Tensor:
        """Stack dense TD-cache rows into m-frame windows.

        Exactly the ``LatentCache.windowed`` construction: frames one action
        block (``fs`` primitive steps) apart, oldest first, clamped to the
        episode's first row at episode starts. ``source`` reads the latents
        from a row-aligned cache (the agent-displaced one) with the same
        episode bookkeeping.
        """
        if c_td is None:
            raise RuntimeError("window rows requested without the dense TD cache")
        src = c_td if source is None else source
        columns = []
        for k in range(vframes - 1, -1, -1):
            offset = k * fs
            j = indices - offset
            j = torch.where(c_td.step_idx[indices] < offset, indices - c_td.step_idx[indices], j)
            columns.append(src.z[j])
        return torch.cat(columns, dim=-1)

    n_plan = a.horizon * fs  # plan length in primitive steps
    if a.gamma >= 1.0:
        plan_cost, plan_disc = float(n_plan), 1.0
        block_cost, block_disc = float(fs), 1.0
    else:
        plan_disc = a.gamma**n_plan
        plan_cost = (1.0 - plan_disc) / (1.0 - a.gamma)
        block_disc = a.gamma**fs
        block_cost = (1.0 - block_disc) / (1.0 - a.gamma)

    def critic_step(expand: ExpandBatch | None = None, tau: float | None = None, lr: float | None = None) -> float:
        if td_sampler is None or c_opt is None:
            raise RuntimeError("critic_step called during actor-only training")
        tau = a.expectile if tau is None else tau
        if lr is not None:
            for pg in c_opt.param_groups:
                pg["lr"] = lr
        b = td_sampler.sample(a.td_batch)
        if vframes > 1:
            # window critics: every query is an m-frame stack rebuilt from the
            # dense cache; the goal side is the sampled goal frame's own window
            z_t = _window_rows(b["t_idx"]).to(dev)
            z_tn = _window_rows(b["tn_idx"]).to(dev)
            z_g = _window_rows(b["g_idx"]).to(dev)
        else:
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
            z0e, traje, zge = expand
            # windowed values receive pre-stacked endpoint windows (2-D);
            # single-frame values the full trajectory (endpoint = last frame)
            endpoint = traje if vframes > 1 else traje[:, -1]
            with torch.no_grad():
                tgt_e = plan_cost + plan_disc * teacher_fn(endpoint, zge)
            loss_e = _expectile_loss(critic_fn(z0e, zge) - tgt_e, tau, a.huber_beta)
            if a.expand_traj:
                # single-block backups along the reused refinement rollout:
                # consecutive imagined latents are one fs-step block apart
                Be, He, De = traje.shape
                src = torch.cat([z0e.unsqueeze(1), traje[:, :-1]], dim=1).reshape(-1, De)
                zg_rep = zge.repeat_interleave(He, dim=0)
                with torch.no_grad():
                    tgt_t = block_cost + block_disc * teacher_fn(traje.reshape(-1, De), zg_rep)
                loss_e = loss_e + _expectile_loss(critic_fn(src, zg_rep) - tgt_t, tau, a.huber_beta)
            loss = loss + a.expand_weight * loss_e
        c_opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature here.
        c_opt.step()
        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), critic.parameters(), strict=True):
                tp.mul_(1.0 - a.ema_tau).add_(a.ema_tau * sp)
        return float(loss.item())

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
            zproj=a.zproj,
            vscale=a.vscale,
            head_scale=a.head_scale,
            cond_mode=a.cond_mode,
            use_gate=not a.no_gate,
            pre_ln=a.pre_ln,
        ).to(dev)
    elif a.arch == "minimal_recurrent":
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
    elif a.arch == "minimal":
        net = PlannerNet(
            z.shape[-1],
            horizon=a.horizon,
            a_dim=a_dim,
            hidden=a.hidden_dim,
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
            hidden=a.hidden_dim,
            feed=a.feed,
            amax=a.amax,
            use_zg=not a.drop_zg,
            use_gate=not a.no_gate,
            use_z0=not a.drop_z0,
            use_grad=not a.drop_grad,
            head_scale=a.head_scale,
        ).to(dev)
    a_opt = torch.optim.AdamW(net.parameters(), lr=a.actor_lr, weight_decay=a.actor_weight_decay)

    replay_buf: dict[str, torch.Tensor | None] = {
        "zh": None,
        "zg": None,
        "agent": None,  # grounding: agent position at the end of the replayed rollout
        "u_prev": None,  # grounding: last commanded displacement of that rollout
    }  # previous step's final imagined windows
    ground_log: dict[str, float] = {}

    def actor_step() -> tuple[float, float, ExpandBatch]:
        zh, ah, zg, ag0, u_prev = sample(a.batch)
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
                replay_agent, replay_u = replay_buf["agent"], replay_buf["u_prev"]
                if grounding is not None and replay_agent is not None and replay_u is not None:
                    # the replayed start is the previous rollout's imagined end: the agent
                    # sits where that plan's commands took it (exact under the kinematics)
                    ag0, u_prev = ag0.clone(), u_prev.clone()
                    ag0[idx] = replay_agent[take]
                    u_prev[idx] = replay_u[take]
        z0 = zh[:, -1]

        def score_trajectory(trajectory: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
            """Deploy-matched trajectory score (m-frame windows when vframes > 1), plus
            the physics-grounding term when enabled -- the same energy the solver
            descends at deploy, so its gradient and value are the actor's inputs too."""
            if vframes > 1:
                energy = windowed_trajectory_value(teacher_fn, trajectory, zg, zh, vframes, a.temporal_objective)
            else:
                energy = trajectory_value(teacher_fn, trajectory, zg, z0, a.temporal_objective)
            if grounding is not None:
                energy = energy + grounding(z0, trajectory, plan, ag0, u_prev)
            return energy

        A = torch.zeros(a.batch, a.horizon, a_dim, device=dev)
        s = net.init_state(a.batch, z0) if isinstance(net, PlannerNetRec) else None
        e_path: list[torch.Tensor] = []
        zT: torch.Tensor | None = None
        tr: torch.Tensor | None = None
        rollout_action: torch.Tensor | None = None
        rollout_trajectory: torch.Tensor | None = None
        rollout_score: torch.Tensor | None = None
        if a.reuse_refinement_rollouts:
            rollout_action = A.detach().requires_grad_(True)
            rollout_trajectory = rollout_traj(wm, zh, ah, rollout_action)
            rollout_score = score_trajectory(rollout_trajectory, rollout_action)
        for k in range(a.iters):
            # gradient feature (detached — input to the learned rule, not the training path)
            if a.reuse_refinement_rollouts:
                if rollout_action is None or rollout_trajectory is None or rollout_score is None:
                    raise RuntimeError("refinement rollout was not initialized")
                (gA,) = torch.autograd.grad(rollout_score.sum(), rollout_action, retain_graph=k > 0)
                traj_f = rollout_trajectory.detach()
                E_feat = rollout_score.detach()
            else:
                with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch stub is untyped.
                    A_in = A.detach().requires_grad_(True)
                    traj = rollout_traj(wm, zh, ah, A_in)
                    score = score_trajectory(traj, A_in)
                    (gA,) = torch.autograd.grad(score.sum(), A_in)
                traj_f = traj.detach()
                E_feat = score.detach()
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
            score = score_trajectory(tr, A)
            e_path.append(score.mean())
            if a.reuse_refinement_rollouts and k + 1 < a.iters:
                rollout_action, rollout_trajectory, rollout_score = A, tr, score
        if a.replay_prob > 0:
            if tr is None:
                raise RuntimeError("actor produced no rollout trajectory")
            replay_buf["zh"] = tr[:, -3:].detach()
            replay_buf["zg"] = zg.detach()
            if grounding is not None:
                with torch.no_grad():
                    replay_buf["agent"] = grounding.agent_path(A.detach(), ag0, u_prev)[:, -1]
                    replay_buf["u_prev"] = grounding.commands(A.detach())[:, -1]
        if grounding is not None and tr is not None:
            with torch.no_grad():
                gt = grounding.terms(z0, tr.detach(), A.detach(), ag0, u_prev)
                ground_log["penalty"] = float(grounding.weight * gt["penalty"].mean())
                ground_log["disp_end"] = float(gt["displacement"][:, -1].mean())
                ground_log["unsupported_end"] = float(gt["unsupported"][:, -1].mean())
        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        if a.get("ac_weight", 0.0) > 0:
            # anti-constancy: penalize batch-level constancy of net plan
            # displacement, ||E_b[sum_t A]||^2 / E_b||sum_t A||^2 in [0,1].
            # An honest planner must vary its plan with the (z0, zg) task; a
            # world-model-exploit basin emits a near-constant plan (~0.8 vs
            # ~0.3 honest). Scale-free, rollout-free.
            _disp = A.sum(1)
            _const = _disp.mean(0).pow(2).sum() / (_disp.pow(2).sum(1).mean() + 1e-8)
            loss = loss + a.ac_weight * _const
        a_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        a_opt.step()
        if zT is None or tr is None:
            raise RuntimeError("actor produced no terminal latent")
        if vframes > 1:
            # stack the expansion tuple into the window value's input space:
            # start window from the real history, terminal window from the
            # rollout, goal tiled to match
            e0, eg = window_pair(zh, zg, vframes)
            eT, _ = window_pair(tr, zg, vframes)
            expand = (e0.detach(), eT.detach(), eg.detach())
        else:
            expand = (z0.detach(), tr.detach(), zg.detach())
        return float(e_path[0].item()), float(e_path[-1].item()), expand

    # ------------------------------------------------------------ checkpoint layout
    # Defined before the loop so periodic snapshots are structurally identical
    # to the final save. The value path is the run-deterministic location the
    # final ``save_metric`` writes to; snapshots reference the same teacher.
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint
    value_checkpoint = Path(str(a.run.directory)).resolve() / "checkpoints" / str(a.output.value_checkpoint)

    def deployable_planner(sd: dict[str, torch.Tensor]) -> dict[str, object]:
        """The full deployable planner payload around an actor state dict."""
        if a.arch == "traj":
            kind = "lip3"
        elif a.arch == "minimal_recurrent":
            kind = "lip4r"
        elif a.arch == "minimal":
            kind = "lip4"
        else:
            kind = "lip" if a.feed == "none" else "lip2"
        return {
            "kind": kind,
            "feed": a.feed,
            "sd": sd,
            "z_dim": z.shape[-1],
            "horizon": a.horizon,
            "iters": a.iters,
            "a_dim": a_dim,
            "amax": a.amax,
            "width": a.width,
            "layers": a.layers,
            "zproj": a.zproj,
            "s_dim": a.s_dim,
            "s0_mode": a.s0_mode,
            "hidden": a.rec_hidden if a.arch == "minimal_recurrent" else a.hidden_dim,
            "goal_mode": a.goal_mode,
            "zero_init": a.zero_init,
            "gd_init": a.gd_init,
            "feat_norm": a.feat_norm,
            "vscale": a.vscale,
            "iter_mode": a.iter_mode,
            "head_mode": a.head_mode,
            "p0": a.p0,
            "cond_mode": a.cond_mode,
            "use_gate": getattr(net, "use_gate", not a.no_gate),
            "use_zg": getattr(net, "use_zg", not a.drop_zg),
            "use_z0": getattr(net, "use_z0", not a.drop_z0),
            "use_grad": getattr(net, "use_grad", not a.drop_grad),
            "head_scale": a.head_scale,
            "pre_ln": a.pre_ln,
            "value": str(value_checkpoint),
            "temporal_objective": a.temporal_objective,
            "window_frames": vframes,
            "window_lag": value_window_lag,
            # physics grounding (probe + calibrated thresholds); the solver adds the
            # same term to its energy when present
            "grounding": grounding.export() if grounding is not None else None,
        }

    # ------------------------------------------------------------ schedule
    pretrain = 0 if a.actor_only else (a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000))
    for i in range(pretrain):
        cl = critic_step()
        if i % 500 == 0:
            logger.info(f"LIP-AC pretrain {i}/{pretrain}: td_loss={cl:.4f}")

    freeze_at = 0 if a.actor_only else int(a.freeze_critic_frac * a.steps)
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
        if a.ckpt_every and (step + 1) % int(a.ckpt_every) == 0 and (step + 1) < a.steps:
            # periodic deployable snapshot so a selection pass can early-stop
            # on held-out success rather than on the training objective
            snapshot = planner_checkpoint.with_name(f"{planner_checkpoint.stem}_step{step + 1}.pt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                deployable_planner({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}),
                snapshot,
            )
            logger.info(f"Saved planner snapshot at step {step + 1} to {snapshot}")
        if step % 500 == 0:
            ground_msg = (
                f" ground {ground_log['penalty']:.3f} disp_end {ground_log['disp_end']:.1f}px"
                f" unsupported_end {ground_log['unsupported_end']:.2f}"
                if ground_log
                else ""
            )
            logger.info(
                f"step {step}: E_final {e_final:.3f} E_first {e_first:.3f} "
                f"td_loss {cl:.4f} tau {tau_s:.3f} clr {clr_s:.2e} alr {alr_s:.2e}{ground_msg}"
            )

    # ------------------------------------------------------------ save (teacher first;
    # the actor checkpoint references it, and it is what the actor optimized against)
    saved_value = save_metric(teacher.cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory)
    logger.success(f"Saved teacher value to {saved_value}")
    net.eval()
    torch.save(deployable_planner(net.cpu().state_dict()), planner_checkpoint)
    logger.success(f"Saved learned planner to {planner_checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="training/lip_ac")


if __name__ == "__main__":
    main()
