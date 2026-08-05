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

    pixi run eval model=hard wm=lewm_tworoom dataset=tworoom_pixels.lance \
        action_block=5 horizon=8 eval_budget=80 \
        'conditions=[latent,regression=metrics/lewm_regression,offline_td=metrics/lewm_td,oracle]'
"""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from stable_worldmodel.policy import BasePolicy
from tabulate import tabulate
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.value import MetricCost, load_metric
from rlp.core.value.oracle import TwoRoomOracleCost, geodesic_distance
from rlp.core.world_model.runtime import load_wm, pick_device
from rlp.data.protocols import Array, Dataset
from rlp.environment import World
from rlp.logging import log_multiline, logger
from rlp.train.online_td import ObsCodec, collect_mpc

from .trm import (
    LeanWorldModelPolicy,
    build_pixel_transform,
    parse_condition,
    read_tworoom_geometry,
)

WALL = 112.0


def build_hard_manifest(
    dataset: Dataset,
    n_per_class: int,
    door_points: Array,
    seed: int,
    min_geo: float,
    max_geo: float,
    n_cand: int = 200000,
) -> tuple[Array, Array, Array]:
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

    def pick(mask: Array) -> Array:
        idx = np.nonzero(mask & keep)[0]
        rng.shuffle(idx)  # uniform within the band
        return idx[:n_per_class]

    cidx, sidx = pick(cross), pick(same)
    sel = np.concatenate([cidx, sidx])
    starts = np.concatenate([a[sel]]).astype(np.float32)
    goals = np.concatenate([b[sel]]).astype(np.float32)
    cls = (["cross"] * len(cidx)) + (["same"] * len(sidx))
    logger.info(
        f"hard manifest: {len(cidx)} cross-wall + {len(sidx)} same-room; "
        f"geo[min/mean/max]={geo[sel].min():.0f}/{geo[sel].mean():.0f}/{geo[sel].max():.0f}"
    )
    return starts, goals, np.array(cls)


def make_planner(
    wm: nn.Module,
    mode: str,
    metric: nn.Module | None,
    oracle_cost: TwoRoomOracleCost,
    args: DictConfig,
    device: str,
    codec: ObsCodec,
) -> BasePolicy:
    model: TwoRoomOracleCost | MetricCost
    if mode == "oracle":
        model = oracle_cost
    elif mode == "latent":
        model = MetricCost(wm, None, mode="latent")
    else:
        model = MetricCost(wm, metric, mode=mode, lam=args.lam)
    solver = swm.solver.CEMSolver(
        model=model,
        num_samples=args.num_samples,
        n_steps=args.cem_steps,
        topk=args.topk,
        device=device,
        seed=args.seed,
        batch_size=args.solver_batch,
    )  # batch ALL envs together (GPU, not serial)
    cfg = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding,
        history_len=1,
        action_block=args.action_block,
        warm_start=True,
    )
    transform = {} if codec.is_state else build_pixel_transform()
    return LeanWorldModelPolicy(
        solver=solver,
        config=cfg,
        process={},
        transform=transform,
        drop_pixels=codec.is_state,
    )


def _run(cfg: DictConfig) -> None:
    args = cfg

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    dataset = swm.data.load_dataset(args.dataset)
    codec = ObsCodec(wm, device)

    n = 2 * args.n_per_class
    world = World(
        args.env,
        num_envs=n,
        max_episode_steps=4 * args.eval_budget,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )
    geom = read_tworoom_geometry(world)
    oracle_cost = TwoRoomOracleCost(**geom)
    door_points = geom["door_points"]

    starts, goals, cls = build_hard_manifest(
        dataset,
        args.n_per_class,
        door_points,
        args.seed,
        args.min_geo,
        args.max_geo,
    )

    rows: list[list[str]] = []
    for spec in args.conditions:
        name, mode, path = parse_condition(spec)
        if mode in ("latent", "oracle"):
            metric = None
        else:
            if path is None:
                raise ValueError(f"condition {spec!r} requires a metric checkpoint")
            metric = load_metric(path, device=device)
        policy = make_planner(wm, mode, metric, oracle_cost, args, device, codec)
        _, done = collect_mpc(world, policy, codec, starts, goals, args.eval_budget)
        cross_succ = 100.0 * done[cls == "cross"].mean()
        same_succ = 100.0 * done[cls == "same"].mean()
        overall = 100.0 * done.mean()
        logger.success(f"{name}: n100={overall:.1f}% cross={cross_succ:.1f}% same={same_succ:.1f}%")
        rows.append(
            [
                name,
                mode,
                f"{overall:.1f}",
                f"{cross_succ:.1f}",
                f"{same_succ:.1f}",
            ]
        )

    table = tabulate(
        rows,
        headers=["condition", "mode", "n100%", "cross%", "same%"],
        tablefmt="github",
    )
    log_multiline(table)
    results_path = Path(args.run.metrics) / args.output.filename
    results_path.write_text(table + "\n")
    logger.info(f"Hard-set results saved to {results_path}")


def main() -> object:
    return run_hydra(dispatch, config_name="eval/hard")


if __name__ == "__main__":
    main()
