"""Train ONE refiner against several frozen (world model, critic) pairs.

Every world model has its own latent space, so each source brings its own
latent cache, its own frozen critic and its own action statistics; only the
learned update rule f_theta is shared. Two uses:

* **Several world models of one environment** -- identical horizon and action
  width, so no padding is needed. This isolates "many landscapes" from "many
  tasks" and tests the reason to do this at all: the refiner's loss is the
  energy of an IMAGINED rollout, so it learns the errors of the particular
  world model it trains against (PUSHT_DIAG E1/E24). A rule that must work on
  several landscapes cannot spend its budget on any one landscape's holes.
* **Several environments** -- one checkpoint for TwoRoom / Reacher / OGBench.
  All four environments plan horizon 5, so the only mismatch is action width
  (TwoRoom / Reacher / PushT 10, OGBench cube 25). The plan is padded to the
  widest source and each source consumes its own prefix, which `v4` handles
  because the horizon is shared; `arch=traj` is needed only if a future source
  brings a different horizon. NOTE: deploying a PADDED checkpoint on a narrow
  environment still needs the solver to slice the plan -- same-width source
  sets (the 2-D trio) deploy unchanged today.

Scope: the critics are FROZEN (no co-training), which is the W1NEAR_FRZ recipe
and keeps this a strict subset of train/lip_ac -- one actor loop, no critic
optimiser, no value expansion, no replay.

Energy scales differ per source (steps-to-go in each environment's own time
units), so each source's loss is divided by an EMA of its own initial energy:
the shared rule optimises RELATIVE improvement and is not dominated by whichever
source has the largest raw energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import h5py
import numpy as np
import torch
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.core.planner import PlannerNet, PlannerNetV3
from rlp.core.rollout import rollout_traj
from rlp.core.solver.lip import ValueFunction
from rlp.core.temporal import trajectory_value, windowed_trajectory_value
from rlp.core.value import load_metric
from rlp.core.world_model import load_pretrained
from rlp.core.world_model.protocols import LatentWorldModel
from rlp.data import LatentCache
from rlp.logging import logger

from .utils import cosine_interpolate


@dataclass
class Source:
    """One (world model, cache, frozen critic) triple the shared rule trains on."""

    name: str
    wm: LatentWorldModel
    z: torch.Tensor
    ep_ids: np.ndarray
    ep_rows: dict[int, np.ndarray]
    act_n: np.ndarray
    ep_off: np.ndarray
    ep_len_h5: np.ndarray | None
    teacher_fn: ValueFunction
    vframes: int
    horizon: int
    a_dim: int
    e_ema: float = 0.0
    steps: int = field(default=0)

    def blocks(self, e: int, t: int, fs: int) -> np.ndarray:
        offset = fs * t
        if self.ep_len_h5 is not None:
            offset = min(offset, max(0, int(self.ep_len_h5[e]) - fs))
        h0 = int(self.ep_off[e] + offset)
        return np.asarray(self.act_n[h0 : h0 + fs]).reshape(-1)


def _load_source(spec: DictConfig, dev: str, fs: int, max_delta: int) -> Source:
    wm_module = load_pretrained(str(spec.wm)).to(dev).eval()
    wm_module.requires_grad_(False)
    cache = LatentCache.load(str(spec.cache), mmap=True)
    eps = cache.episodes()
    keys = [k for k in eps if len(eps[k]) > max_delta + 4]
    if not keys:
        longest = max((len(v) for v in eps.values()), default=0)
        raise ValueError(
            f"source {spec.name}: no episode in {spec.cache} longer than max_delta+4 = {max_delta + 4} "
            f"(longest {longest})"
        )
    with h5py.File(str(spec.h5), "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
        ep_len_h5 = h["ep_len"][:] if "ep_len" in h else None
    amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    act_n = ((act - amu) / astd).astype(np.float32)
    critic = load_metric(str(spec.value), device=dev)
    for prm in critic.parameters():
        prm.requires_grad_(False)
    critic.eval()
    base_dim = int(cache.latent_dim)
    vframes = int(cast(int, critic.latent_dim)) // base_dim
    src = Source(
        name=str(spec.name),
        wm=cast(LatentWorldModel, wm_module),
        z=cache.z.to(dev).float(),
        ep_ids=np.array(keys),
        ep_rows={e: np.asarray(eps[e]) for e in keys},
        act_n=act_n,
        ep_off=ep_off,
        ep_len_h5=ep_len_h5,
        teacher_fn=cast(ValueFunction, critic),
        vframes=vframes,
        horizon=int(spec.get("horizon", 5) or 5),
        a_dim=int(act.shape[-1]) * fs,
    )
    logger.info(
        f"source {src.name}: {len(src.ep_ids)} episodes, latent {base_dim}, vframes {vframes}, "
        f"horizon {src.horizon}, a_dim {src.a_dim}"
    )
    return src


def _run(cfg: DictConfig) -> None:
    a = cfg
    dev = str(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    fs = 5
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    sources = [_load_source(s, dev, fs, int(a.max_delta)) for s in a.sources]
    if not sources:
        raise ValueError("train/lip_multi needs at least one source")
    a_dim = max(s.a_dim for s in sources)
    horizon = max(s.horizon for s in sources)
    z_dim = int(sources[0].z.shape[-1])
    if any(int(s.z.shape[-1]) != z_dim for s in sources):
        raise ValueError("sources disagree on latent width; the shared rule conditions on z")
    mixed = len({(s.a_dim, s.horizon) for s in sources}) > 1
    # v4 flattens horizon x a_dim into one Linear, so a padded plan is fine as long
    # as the HORIZON matches (a narrow source just leaves the tail columns at zero);
    # only a variable horizon needs the tokenised rule.
    if len({s.horizon for s in sources}) > 1 and str(a.arch) != "traj":
        raise ValueError(
            f"sources disagree on horizon ({sorted({s.horizon for s in sources})}); architecture 'v4' "
            "flattens horizon x a_dim into one Linear and cannot take a variable horizon -- use arch=traj"
        )
    logger.info(f"shared rule: a_dim {a_dim} horizon {horizon} over {len(sources)} sources (mixed={mixed})")

    if str(a.arch) == "traj":
        net = PlannerNetV3(
            z_dim,
            horizon=horizon,
            a_dim=a_dim,
            amax=float(a.amax),
            width=int(a.width),
            layers=int(a.layers),
        ).to(dev)
    else:
        net = PlannerNet(  # type: ignore[assignment]
            z_dim,
            horizon=horizon,
            a_dim=a_dim,
            hidden=int(a.width),
            amax=float(a.amax),
            use_zg=False,
            use_z0=False,
            use_gate=True,
        ).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=float(a.actor_lr))

    def sample(src: Source, B: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zh, ah, zg = [], [], []
        for _ in range(B):
            e = int(src.ep_ids[rng.integers(len(src.ep_ids))])
            rows = src.ep_rows[e]
            L = len(rows)
            t = int(rng.integers(2, L - 2))
            zh.append(torch.stack([src.z[rows[t - 2]], src.z[rows[t - 1]], src.z[rows[t]]]))
            ah.append(np.stack([src.blocks(e, t - 2, fs), src.blocks(e, t - 1, fs)]))
            if rng.random() < float(a.p_cross):
                e2 = int(src.ep_ids[rng.integers(len(src.ep_ids))])
                r2 = src.ep_rows[e2]
                zg.append(src.z[r2[rng.integers(len(r2))]])
            else:
                d = int(rng.integers(1, int(a.max_delta) + 1))
                zg.append(src.z[rows[min(t + d, L - 1)]])
        return (
            torch.stack(zh),
            torch.from_numpy(np.stack(ah)).to(dev),
            torch.stack(zg),
        )

    def actor_step(src: Source) -> float:
        zh, ah, zg = sample(src, int(a.batch))
        z0 = zh[:, -1]

        def score(trajectory: torch.Tensor) -> torch.Tensor:
            if src.vframes > 1:
                return windowed_trajectory_value(
                    src.teacher_fn, trajectory, zg, zh, src.vframes, str(a.temporal_objective)
                )
            return trajectory_value(src.teacher_fn, trajectory, zg, z0, str(a.temporal_objective))

        # the shared rule always emits the padded plan; each source consumes its
        # own prefix, so unused action slots never reach a world model
        A = torch.zeros(int(a.batch), horizon, a_dim, device=dev)
        e_path: list[torch.Tensor] = []
        for k in range(int(a.iters)):
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                A_in = A.detach().requires_grad_(True)
                traj = rollout_traj(src.wm, zh, ah, A_in[:, : src.horizon, : src.a_dim])
                sc = score(traj)
                (gA,) = torch.autograd.grad(sc.sum(), A_in)
            A = net(A, gA.detach(), sc.detach(), z0, zg, traj.detach(), k=k)
            tr = rollout_traj(src.wm, zh, ah, A[:, : src.horizon, : src.a_dim])
            e_path.append(score(tr).mean())
        e0 = float(e_path[0].detach())
        src.e_ema = e0 if src.steps == 0 else 0.99 * src.e_ema + 0.01 * e0
        src.steps += 1
        scale = max(src.e_ema, 1e-6)
        loss = (e_path[-1] + float(a.mean_weight) * torch.stack(e_path).mean()) / scale
        if float(a.get("ac_weight", 0.0) or 0.0) > 0:
            disp = A.sum(1)
            const = disp.mean(0).pow(2).sum() / (disp.pow(2).sum(1).mean() + 1e-8)
            loss = loss + float(a.ac_weight) * const
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        return float(e_path[-1].detach())

    log_every = max(1, int(a.steps) // 20)
    for step in range(int(a.steps)):
        for pg in opt.param_groups:
            pg["lr"] = cosine_interpolate(float(a.actor_lr), a.get("actor_lr_final"), step, int(a.steps))
        src = sources[step % len(sources)]  # round robin: every source sees every lr phase
        e_final = actor_step(src)
        if step % log_every == 0:
            logger.info(f"multi step {step}/{a.steps} [{src.name}] E_final {e_final:.3f} (ema {src.e_ema:.3f})")

    out = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        "arch": str(a.arch),
        "a_dim": a_dim,
        "horizon": horizon,
        "amax": float(a.amax),
        "iterations": int(a.iters),
        "z_dim": z_dim,
        "sources": [s.name for s in sources],
        # per-source plan widths: a deployed solver slices the padded plan
        "source_shapes": {s.name: [s.horizon, s.a_dim] for s in sources},
    }
    path = str(cfg.run.checkpoints) + "/planner_multi.pt"
    torch.save(out, path)
    logger.success(f"Saved multi-source planner to {path} (sources {[s.name for s in sources]})")


def run() -> None:
    run_hydra(lambda cfg: dispatch(cfg), config_name="train/lip_multi")


__all__ = ["Source", "_run", "run"]
