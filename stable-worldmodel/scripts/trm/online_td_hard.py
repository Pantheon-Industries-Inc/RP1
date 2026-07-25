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

    python scripts/trm/online_td_hard.py --wm lewm_tworoom --dataset tworoom_pixels.lance \
        --seed-cache caches/lewm_tworoom.pt --init-metric metrics/atd_q_n20_g099.pt \
        --n-step 20 --gamma 0.99 --rounds 8 --save-prefix metrics/online_seedbest \
        --save-rounds 2,4,8 --device cuda
"""

import argparse
import copy
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.trm import LatentCache, load_metric, save_metric
from stable_worldmodel.trm.head import QuasimetricHead
from stable_worldmodel.trm.learners.td import _expectile_loss
from stable_worldmodel.trm.samplers import NStepGoalSampler

from _common import load_wm, pick_device
from eval_trm import read_tworoom_geometry
from eval_hard import build_hard_manifest
from online_td import ObsCodec, collect_mpc, append_trajectories, make_policy

QUASI_ARCH = {"head": "quasimetric", "hidden_dim": 256, "depth": 2, "embed_dim": 128}


def td_nstep_updates(value, target, opt, sampler, steps, gamma, expectile, tau, batch_size, device):
    """Same n-step distance-TD update as offline learners/td.fit (quasimetric).
    ``opt`` is persistent across rounds (no Adam-state reset thrash)."""
    value.train()
    last = 0.0
    for _ in range(steps):
        b = sampler.sample(batch_size)
        z_t, z_tn, z_g = b["z_t"].to(device), b["z_tn"].to(device), b["z_g"].to(device)
        ne, reached, dist = b["n_eff"].to(device), b["reached"].to(device), b["dist"].to(device)
        with torch.no_grad():
            d_next = target(z_tn, z_g)
            if gamma >= 1.0:
                c, disc = ne, torch.ones_like(ne)
            else:
                disc = gamma ** ne
                c = (1.0 - disc) / (1.0 - gamma)
            tgt = reached * dist + (1.0 - reached) * (c + disc * d_next)
        pred = value(z_t, z_g)
        loss = _expectile_loss(pred - tgt, expectile, 1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters()):
                tp.mul_(1.0 - tau).add_(tau * sp)
        last = float(loss)
    value.eval()
    return last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--env", default="swm/TwoRoom-v1")
    p.add_argument("--seed-cache", required=True, help="LatentCache to seed the buffer (all logged trajectories)")
    p.add_argument("--init-metric", default=None, help="offline quasimetric value to continue (None = fresh)")
    p.add_argument("--save-prefix", required=True, help="metrics/<name>; saves <name>_r{round}.pt")
    p.add_argument("--save-rounds", default="", help="comma list of round numbers to checkpoint (e.g. 2,4,8)")
    # TD (match offline)
    p.add_argument("--n-step", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=5e-3)
    p.add_argument("--p-cross", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--td-steps-per-round", type=int, default=1500)
    p.add_argument("--only-successful", action="store_true",
                   help="append ONLY goal-reaching collected trajectories (keep buffer near-optimal)")
    # collection on HARD goals (high-distance band)
    p.add_argument("--n-collect", type=int, default=100, help="hard collection goals/round (50 cross + 50 same)")
    p.add_argument("--min-geo", type=float, default=90.0)
    p.add_argument("--max-geo", type=float, default=180.0)
    p.add_argument("--collect-budget", type=int, default=120)
    p.add_argument("--collect-samples", type=int, default=60)
    p.add_argument("--collect-cem-steps", type=int, default=5)
    p.add_argument("--collect-topk", type=int, default=5)
    # planner geometry (match eval_hard for LeWM)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--receding", type=int, default=8)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    dataset = swm.data.load_dataset(args.dataset)
    codec = ObsCodec(wm, device)
    save_rounds = {int(x) for x in args.save_rounds.split(",") if x.strip()}

    # buffer seeded with all logged trajectories (same data offline TD saw)
    cache = LatentCache.load(args.seed_cache)
    next_ep = int(cache.episode_idx.max()) + 1
    logging.info(f"seed buffer: {len(cache.z)} latents, next_ep={next_ep}")

    # value continues the offline metric (or fresh quasimetric)
    if args.init_metric:
        value = load_metric(args.init_metric, device=device)
        logging.info(f"continuing offline value {args.init_metric}")
    else:
        value = QuasimetricHead(cache.latent_dim, hidden_dim=256, embed_dim=128, depth=2).to(device)
        logging.info("fresh quasimetric value")
    target = copy.deepcopy(value)
    for q in target.parameters():
        q.requires_grad_(False)

    # collection world (n_collect hard goals); geometry for the manifest
    n_col = args.n_collect
    world = swm.World(args.env, num_envs=n_col, max_episode_steps=4 * args.collect_budget,
                      image_shape=(224, 224), render_mode="rgb_array")
    geom = read_tworoom_geometry(world)
    door_points = geom["door_points"]
    pol_args = SimpleNamespace(collect_samples=args.collect_samples, collect_cem_steps=args.collect_cem_steps,
                               collect_topk=args.collect_topk, horizon=args.horizon, receding=args.receding,
                               action_block=args.action_block, seed=args.seed)

    def checkpoint(rnd):
        out = f"{args.save_prefix}_r{rnd}.pt"
        save_metric(value.cpu(), "td", cache.latent_dim, QUASI_ARCH, out)
        value.to(device)
        logging.success(f"saved online value -> {out}")

    opt = torch.optim.AdamW(value.parameters(), lr=1e-3, weight_decay=1e-4)  # persistent across rounds
    rng = np.random.default_rng(args.seed)
    for rnd in range(1, args.rounds + 1):
        # collect on a *fresh* hard manifest (rotating seed; disjoint from eval seed=1)
        cseed = 1000 + rnd
        cs, cg, _ = build_hard_manifest(dataset, n_col // 2, door_points, cseed, args.min_geo, args.max_geo)
        pol = make_policy(wm, value, pol_args, device, codec)
        trajs, cdone = collect_mpc(world, pol, codec, cs, cg, args.collect_budget)
        if args.only_successful:                       # keep only goal-reaching (near-optimal) data
            trajs = [tr for tr, d in zip(trajs, cdone) if bool(d)]
        cache, next_ep = append_trajectories(cache, trajs, codec, next_ep)
        sampler = NStepGoalSampler(cache, n_step=args.n_step, p_cross=args.p_cross,
                                   balanced=True, seed=args.seed + rnd)
        loss = td_nstep_updates(value, target, opt, sampler, args.td_steps_per_round, args.gamma,
                                args.expectile, args.tau, args.batch_size, device)
        logging.success(f"round {rnd}: buffer={len(cache.z)} collect={100*cdone.mean():.1f}% "
                        f"added={len(trajs)} loss={loss:.4f}")
        if rnd in save_rounds or rnd == args.rounds:
            checkpoint(rnd)


if __name__ == "__main__":
    main()
