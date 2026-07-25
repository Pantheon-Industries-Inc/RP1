"""Hard-n100 manifest evaluation (paper's broken-latent regime).

The paper's TwoRoom **hard n100** set = 50 same-room + 50 cross-wall goals in the
**high-distance bucket** (47/50 cross-wall require the doorway), where raw latent
distance is maximally misleading (LeWM latent ≈ 7%, TRM ≈ 97%). Short-range goals
(goal_offset along a trajectory) are too easy and hide the effect.

This evaluator samples high-distance (start, goal) *position* pairs with a 50/50
same-room/cross-wall split (geodesic via the doorway oracle), then scores every
terminal-cost condition on the SAME pairs via goal-conditioned MPC rollouts
(reusing the online_td rollout machinery). Works for LeWM (action_block 5) and
DINO-WM (action_block 1).

Example::

    python scripts/trm/eval_hard.py --wm lewm_tworoom --dataset tworoom_pixels.lance \
        --action-block 5 --horizon 8 --eval-budget 80 \
        --condition latent --condition regression=metrics/lewm_regression.pt \
        --condition offline_td=metrics/lewm_td.pt --condition online_td=metrics/lewm_online_td.pt \
        --condition shuffled=metrics/lewm_shuffled.pt --condition oracle --out results/lewm_hard.txt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tabulate import tabulate

import stable_worldmodel as swm
from stable_worldmodel.trm import MetricCost, load_metric
from stable_worldmodel.trm.latent_cache import _episode_col
from stable_worldmodel.trm.oracle import TwoRoomOracleCost, geodesic_distance

from _common import is_statewm, load_wm, pick_device
from eval_trm import parse_condition, build_pixel_transform, read_tworoom_geometry, LeanWorldModelPolicy
from online_td import ObsCodec, collect_mpc

WALL = 112.0


def build_hard_manifest(dataset, n_per_class, door_points, seed, min_geo, max_geo, n_cand=200000):
    """Sample high-distance (start, goal) proprio pairs: n_per_class same-room +
    n_per_class cross-wall, all with geodesic >= min_geo (high-distance bucket)."""
    pro = np.asarray(dataset.get_col_data("proprio")).reshape(-1, 2)[:, :2].astype(np.float32)
    rng = np.random.default_rng(seed)
    dp = torch.as_tensor(door_points, dtype=torch.float32)
    si = rng.integers(0, len(pro), n_cand)
    sj = rng.integers(0, len(pro), n_cand)
    a, b = pro[si], pro[sj]
    geo = geodesic_distance(torch.from_numpy(a), torch.from_numpy(b), dp).numpy()
    same = (a[:, 0] < WALL) == (b[:, 0] < WALL)
    cross = ~same
    # high-distance *band*: geodesic in [min_geo, max_geo] — far enough that the
    # wall dominates, but still reachable (verified by the oracle row).
    keep = (geo >= min_geo) & (geo <= max_geo)
    def pick(mask):
        idx = np.nonzero(mask & keep)[0]
        rng.shuffle(idx)                            # uniform within the band
        return idx[:n_per_class]
    cidx, sidx = pick(cross), pick(same)
    sel = np.concatenate([cidx, sidx])
    starts = np.concatenate([a[sel]]).astype(np.float32)
    goals = np.concatenate([b[sel]]).astype(np.float32)
    cls = (["cross"] * len(cidx)) + (["same"] * len(sidx))
    logging.info(f"hard manifest: {len(cidx)} cross-wall + {len(sidx)} same-room; "
                 f"geo[min/mean/max]={geo[sel].min():.0f}/{geo[sel].mean():.0f}/{geo[sel].max():.0f}")
    return starts, goals, np.array(cls)


def make_planner(wm, mode, metric, oracle_cost, args, device, codec):
    if mode == "oracle":
        model = oracle_cost
    elif mode == "latent":
        model = MetricCost(wm, None, mode="latent")
    else:
        model = MetricCost(wm, metric, mode=mode, lam=args.lam)
    solver = swm.solver.CEMSolver(model=model, num_samples=args.num_samples,
                                  n_steps=args.cem_steps, topk=args.topk, device=device, seed=args.seed,
                                  batch_size=args.solver_batch)  # batch ALL envs together (GPU, not serial)
    cfg = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding,
                         history_len=1, action_block=args.action_block, warm_start=True)
    transform = {} if codec.is_state else build_pixel_transform()
    return LeanWorldModelPolicy(solver=solver, config=cfg, process={}, transform=transform,
                                drop_pixels=codec.is_state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--env", default="swm/TwoRoom-v1")
    p.add_argument("--condition", action="append", default=[], dest="conditions")
    p.add_argument("--n-per-class", type=int, default=50)   # 50 cross + 50 same = n100
    p.add_argument("--min-geo", type=float, default=90.0)   # high-distance band lower (px)
    p.add_argument("--max-geo", type=float, default=180.0)  # high-distance band upper (px)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--receding", type=int, default=8)
    p.add_argument("--eval-budget", type=int, default=80)
    p.add_argument("--solver-batch", type=int, default=1024, help="CEM env-batch (>=n_envs => 1 batched GPU solve)")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--cem-steps", type=int, default=8)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    dataset = swm.data.load_dataset(args.dataset)
    codec = ObsCodec(wm, device)

    n = 2 * args.n_per_class
    world = swm.World(args.env, num_envs=n, max_episode_steps=4 * args.eval_budget,
                      image_shape=(224, 224), render_mode="rgb_array")
    geom = read_tworoom_geometry(world)
    oracle_cost = TwoRoomOracleCost(**geom)
    door_points = geom["door_points"]

    starts, goals, cls = build_hard_manifest(dataset, args.n_per_class, door_points, args.seed, args.min_geo, args.max_geo)

    rows = []
    for spec in args.conditions:
        name, mode, path = parse_condition(spec)
        metric = None if mode in ("latent", "oracle") else load_metric(path, device=device)
        policy = make_planner(wm, mode, metric, oracle_cost, args, device, codec)
        _, done = collect_mpc(world, policy, codec, starts, goals, args.eval_budget)
        cross_succ = 100.0 * done[cls == "cross"].mean()
        same_succ = 100.0 * done[cls == "same"].mean()
        overall = 100.0 * done.mean()
        logging.success(f"[{name}] n100={overall:.1f}%  (cross={cross_succ:.1f}  same={same_succ:.1f})")
        rows.append([name, mode, f"{overall:.1f}", f"{cross_succ:.1f}", f"{same_succ:.1f}"])

    table = tabulate(rows, headers=["condition", "mode", "n100%", "cross%", "same%"], tablefmt="github")
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(f"\n==== HARD n100 wm={args.wm} action_block={args.action_block} "
                    f"min_geo={args.min_geo} horizon={args.horizon} budget={args.eval_budget} ====\n")
            f.write(table + "\n")


if __name__ == "__main__":
    main()
