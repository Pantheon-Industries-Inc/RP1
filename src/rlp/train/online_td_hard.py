"""Online TD on the **hard n100 regime**, comparable to the offline sweep.

The offline TD sweep trains an n-step HER **quasimetric** value on the *fixed*
all-trajectory cache; its best (n=20, gamma=0.99) reaches 52% on hard n100 at
paper CEM. The logged buffer under-covers the long cross-wall (doorway) states
the hard planner actually visits. **Online TD** continues that exact value while
collecting fresh trajectories on **hard high-distance goals** with the current
metric, appending them to the buffer (HER relabel), and taking the *same*
n-step quasimetric TD updates. The result is saved with its true quasimetric
arch so ``eval_hard.py`` scores it on the identical hard n100 manifest at paper
CEM -- a like-for-like online-vs-offline number.

This deliberately reuses the offline machinery (NStepGoalSampler, QuasimetricHead,
_expectile_loss) so the only difference from offline is the *data distribution*
(planner-induced hard states), isolating the online effect.

Example::

    pixi run train model=online_td_hard wm=lewm_tworoom \
        dataset=tworoom_pixels.lance seed_cache=caches/lewm_tworoom.pt \
        init_metric=metrics/atd_q_n20_g099 n_step=20 gamma=0.99 rounds=8 \
        output.checkpoint_prefix=online_seedbest save_rounds=2,4,8 device=cuda
"""

import copy
from typing import cast

import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Optimizer

from rlp.config import dispatch, run_hydra
from rlp.core.solver.lip import ValueFunction
from rlp.core.value import load_metric, save_metric
from rlp.core.value.head import QuasimetricHead
from rlp.core.value.learners.td import _expectile_loss
from rlp.core.value.samplers import NStepGoalSampler
from rlp.core.world_model.runtime import load_wm, pick_device
from rlp.data import LatentCache
from rlp.environment import World
from rlp.eval.hard import build_hard_manifest
from rlp.eval.trm import read_tworoom_geometry
from rlp.logging import logger

from .online_td import MPCSettings, ObsCodec, append_trajectories, collect_mpc, make_policy


def td_nstep_updates(
    value: nn.Module,
    target: nn.Module,
    opt: Optimizer,
    sampler: NStepGoalSampler,
    steps: int,
    gamma: float,
    expectile: float,
    tau: float,
    batch_size: int,
    device: str,
) -> float:
    """Same n-step distance-TD update as offline learners/td.fit (quasimetric).
    ``opt`` is persistent across rounds (no Adam-state reset thrash)."""
    value.train()
    value_fn = cast(ValueFunction, value)
    target_fn = cast(ValueFunction, target)
    last = 0.0
    for _ in range(steps):
        b = sampler.sample(batch_size)
        z_t, z_tn, z_g = (
            b["z_t"].to(device),
            b["z_tn"].to(device),
            b["z_g"].to(device),
        )
        ne, reached, dist = (
            b["n_eff"].to(device),
            b["reached"].to(device),
            b["dist"].to(device),
        )
        with torch.no_grad():
            d_next = target_fn(z_tn, z_g)
            if gamma >= 1.0:
                c, disc = ne, torch.ones_like(ne)
            else:
                disc = gamma**ne
                c = (1.0 - disc) / (1.0 - gamma)
            tgt = reached * dist + (1.0 - reached) * (c + disc * d_next)
        pred = value_fn(z_t, z_g)
        loss = _expectile_loss(pred - tgt, expectile, 1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature here.
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters(), strict=True):
                tp.mul_(1.0 - tau).add_(tau * sp)
        last = float(loss)
    value.eval()
    return last


def _run(cfg: DictConfig) -> None:
    args = OmegaConf.merge(cfg, cfg.core.solver)
    if not isinstance(args, DictConfig):
        raise TypeError("merged solver configuration must be a mapping")
    args.receding = args.receding_horizon
    args.collect_samples = args.num_samples
    args.collect_cem_steps = args.cem_steps
    args.collect_topk = args.topk

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    dataset = swm.data.load_dataset(args.dataset)
    codec = ObsCodec(wm, device)
    save_rounds = {int(x) for x in args.save_rounds.split(",") if x.strip()}

    # buffer seeded with all logged trajectories (same data offline TD saw)
    cache = LatentCache.load(args.seed_cache)
    next_ep = int(cache.episode_idx.max()) + 1
    logger.info(f"Seed buffer: {len(cache.z)} latents, next_episode={next_ep}")

    # value continues the offline metric (or fresh quasimetric)
    if args.init_metric:
        value = load_metric(args.init_metric, device=device)
        logger.info(f"Continuing offline value {args.init_metric}")
    else:
        value = QuasimetricHead(cache.latent_dim, hidden_dim=256, embed_dim=128, depth=2).to(device)
        logger.info("Initialized fresh quasimetric value")
    target = copy.deepcopy(value)
    for q in target.parameters():
        q.requires_grad_(False)

    # collection world (n_collect hard goals); geometry for the manifest
    n_col = args.n_collect
    world = World(
        args.env,
        num_envs=n_col,
        max_episode_steps=4 * args.collect_budget,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    geom = read_tworoom_geometry(world)
    door_points = geom["door_points"]
    pol_args = MPCSettings(
        collect_samples=int(args.collect_samples),
        collect_cem_steps=int(args.collect_cem_steps),
        collect_topk=int(args.collect_topk),
        horizon=int(args.horizon),
        receding=int(args.receding),
        action_block=int(args.action_block),
        seed=int(args.seed),
    )

    def checkpoint(rnd: int) -> None:
        out = save_metric(
            value.cpu(),
            run_name=f"{args.output.checkpoint_prefix}_{rnd}",
            cache_dir=args.run.directory,
        )
        value.to(device)
        logger.success(f"Saved online value to {out}")

    opt = torch.optim.AdamW(value.parameters(), lr=1e-3, weight_decay=1e-4)  # persistent across rounds
    for rnd in range(1, args.rounds + 1):
        # collect on a *fresh* hard manifest (rotating seed; disjoint from eval seed=1)
        cseed = 1000 + rnd
        cs, cg, _ = build_hard_manifest(dataset, n_col // 2, door_points, cseed, args.min_geo, args.max_geo)
        pol = make_policy(wm, value, pol_args, device, codec)
        trajs, cdone = collect_mpc(world, pol, codec, cs, cg, args.collect_budget)
        if args.only_successful:  # keep only goal-reaching (near-optimal) data
            trajs = [tr for tr, d in zip(trajs, cdone, strict=True) if bool(d)]
        cache, next_ep = append_trajectories(cache, trajs, codec, next_ep)
        sampler = NStepGoalSampler(
            cache,
            n_step=args.n_step,
            p_cross=args.p_cross,
            balanced=True,
            seed=args.seed + rnd,
        )
        loss = td_nstep_updates(
            value,
            target,
            opt,
            sampler,
            args.td_steps_per_round,
            args.gamma,
            args.expectile,
            args.tau,
            args.batch_size,
            device,
        )
        logger.success(
            f"round {rnd}: buffer={len(cache.z)} collect={100 * cdone.mean():.1f}% added={len(trajs)} loss={loss:.4f}"
        )
        if rnd in save_rounds or rnd == args.rounds:
            checkpoint(rnd)


def main() -> object:
    return run_hydra(dispatch, config_name="train/online_td_hard")


if __name__ == "__main__":
    main()
