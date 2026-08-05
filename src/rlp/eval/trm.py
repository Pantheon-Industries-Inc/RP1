"""Evaluate TRM terminal-cost conditions with CEM-MPC (CPU/MPS friendly).

For each *cost condition* we wrap the frozen world model in a ``MetricCost`` and
run the SAME CEM planner / sampler / budget, changing only the terminal cost --
isolating the planner-facing metric. Reports per-condition success rate, and
(when an oracle is available, i.e. TwoRoom) the Same-Candidate Selection Audit
(SCSA): Spearman / best-rank-percentile / regret vs the ground-truth task-state
oracle on a shared candidate set.

Conditions (``conditions=[NAME[=METRIC.pt],...]``):
    latent                      raw latent MSE baseline (the broken cost)
    trm_regression=PATH         replacement with the regression head
    trm_td=PATH                 replacement with the TD value
    trm_contrastive=PATH        replacement with the contrastive critic
    hybrid=PATH                 std(c_lat) + lambda*std(metric)
    shuffled=PATH               negative control (shuffled-label head)
    oracle                      ground-truth task-state planner (TwoRoom only)

Example::

    pixi run eval model=trm env=swm/TwoRoom-v1 wm=statewm_tworoom \
        dataset=tworoom_expert.lance num_eval=20 cross_wall=true \
        'conditions=[latent,trm_regression=metrics/reg.pt,trm_td=metrics/td.pt,oracle]' \
        scsa=true
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig
from tabulate import tabulate
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.solver.lip import EncoderWorldModel
from rlp.core.value import MetricCost, diagnostics, load_metric
from rlp.core.value.oracle import TwoRoomOracleCost
from rlp.core.world_model.runtime import is_statewm, load_wm, pick_device
from rlp.data.protocols import Dataset
from rlp.environment import World
from rlp.logging import log_multiline, logger


class TwoRoomGeometry(TypedDict):
    door_points: np.ndarray
    speed: float
    agent_radius: float
    door_half: float
    wall_thickness: int


type Condition = tuple[str, str, nn.Module | None, nn.Module | TwoRoomOracleCost]

WALL_CENTER = 112.0


class LeanWorldModelPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that optionally drops bulky image keys before planning.

    The solver expands *every* info tensor by ``num_samples``; for a
    state/proprio world model the ``pixels``/``goal`` images are unused and
    would blow up memory, so we strip them after preprocessing. For a pixel
    world model (LeWM) they are required, so set ``drop_pixels=False``.
    """

    drop_pixels: bool = True

    def _prepare_info(self, info_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        out = super()._prepare_info(info_dict)
        if self.drop_pixels:
            for k in list(out.keys()):
                if k.startswith("pixels") or k in ("goal", "goal_pixels"):
                    out.pop(k, None)
        return cast(dict[str, torch.Tensor], out)


def parse_condition(spec: str) -> tuple[str, str, str | None]:
    if "=" in spec:
        name, path = spec.split("=", 1)
    else:
        name, path = spec, None
    if name == "latent":
        mode = "latent"
    elif name == "oracle":
        mode = "oracle"
    elif name.startswith("hybrid"):
        mode = "hybrid"
    elif name.startswith("shuffled"):
        mode = "shuffled"
    else:
        mode = "replacement"
    return name, mode, path


def read_tworoom_geometry(world: World) -> TwoRoomGeometry:
    """Pull door points / speed / radius from a reset TwoRoom env."""
    env = world.envs.envs[0].unwrapped
    env.reset()
    wall_axis = int(env.wall_axis)
    n = int(env.num_doors)
    door_pts: list[list[float]] = []
    for i in range(n):
        c = float(env.door_positions[i])
        door_pts.append([WALL_CENTER, c] if wall_axis == 1 else [c, WALL_CENTER])
    speed = float(env.variation_space["agent"]["speed"].value.item())
    radius = float(env.variation_space["agent"]["radius"].value.item())
    door_half = float(env.door_sizes[0]) if n else 21.0
    return {
        "door_points": np.array(door_pts, dtype=np.float32),
        "speed": speed,
        "agent_radius": radius,
        "door_half": door_half,
        "wall_thickness": int(env.wall_thickness),
    }


def sample_eval_episodes(
    dataset: Dataset,
    num_eval: int,
    goal_offset: int,
    obs_key: str,
    seed: int,
    cross_wall: bool = False,
) -> tuple[list[int], list[int]]:
    """Pick (episode, start_step) pairs with enough room before the goal."""
    from rlp.data.latent_cache import _episode_col

    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    ep_ids = np.unique(ep)
    lengths = {int(e): int((ep == e).sum()) for e in ep_ids}

    candidates: list[tuple[int, int]] = []
    for e in ep_ids:
        L = lengths[int(e)]
        for s in range(0, L - goal_offset - 1):
            candidates.append((int(e), s))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    chosen_ep: list[int] = []
    chosen_start: list[int] = []
    for e, s in candidates:
        if len(chosen_ep) >= num_eval:
            break
        if cross_wall:
            rows = np.nonzero(ep == e)[0][np.argsort(st[ep == e])]
            start_pos = np.asarray(dataset.get_row_data([int(rows[s])])[obs_key]).reshape(-1)[:2]
            goal_pos = np.asarray(dataset.get_row_data([int(rows[s + goal_offset])])[obs_key]).reshape(-1)[:2]
            if (start_pos[0] < WALL_CENTER) == (goal_pos[0] < WALL_CENTER):
                continue
        chosen_ep.append(e)
        chosen_start.append(s)
    return chosen_ep, chosen_start


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_pixel_transform(img_size: int = 224) -> dict[str, Callable[[Any], Any]]:
    """ImageNet transform for pixels/goal (DINO-WM / pixel WMs). Actions stay raw
    to match how our WMs are trained, so no StandardScaler `process` is needed."""
    from torchvision.transforms import v2 as T

    img_tf = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.Resize(size=img_size),
        ]
    )
    return {"pixels": img_tf, "goal": img_tf}


def build_solver(model: nn.Module | TwoRoomOracleCost, args: DictConfig, device: str) -> swm.solver.CEMSolver:
    """Build the planning solver. ``cem`` is the planner used in the TRM paper;
    ``predictive_sampling`` is the MPC shooting method (sample N, pick argmin)."""
    name = args.solver
    if name == "cem":
        return swm.solver.CEMSolver(
            model=model,
            num_samples=args.num_samples,
            n_steps=args.cem_steps,
            topk=args.topk,
            var_scale=args.var_scale,
            device=device,
            seed=args.seed,
        )
    if name == "predictive_sampling":
        return swm.solver.PredictiveSamplingSolver(
            model=model,
            num_samples=args.num_samples,
            noise_scale=args.var_scale,
            device=device,
            seed=args.seed,
        )
    if name == "mppi":
        return swm.solver.MPPISolver(
            model=model,
            num_samples=args.num_samples,
            n_steps=args.cem_steps,
            topk=args.topk,
            var_scale=args.var_scale,
            device=device,
            seed=args.seed,
        )
    if name == "icem":
        return swm.solver.ICEMSolver(
            model=model,
            num_samples=args.num_samples,
            n_steps=args.cem_steps,
            topk=args.topk,
            var_scale=args.var_scale,
            device=device,
            seed=args.seed,
        )
    raise ValueError(f"unknown solver '{name}'")


def run_scsa(
    wm: nn.Module,
    conditions_built: Sequence[Condition],
    oracle_cost: TwoRoomOracleCost,
    dataset: Dataset,
    episodes: Sequence[int],
    starts: Sequence[int],
    goal_offset: int,
    obs_key: str,
    args: DictConfig,
    device: str,
) -> dict[str, dict[str, float]]:
    """Same-Candidate Selection Audit over sampled planning states."""
    from rlp.data.latent_cache import _episode_col

    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    H, A = args.horizon, 2
    per_metric: dict[str, list[dict[str, float]]] = {name: [] for name, _, _, _ in conditions_built if name != "oracle"}
    state_wm = cast(EncoderWorldModel, wm)

    for e, s in zip(episodes, starts, strict=True):
        rows = np.nonzero(ep == e)[0][np.argsort(st[ep == e])]
        start_pos = np.asarray(dataset.get_row_data([int(rows[s])])[obs_key]).reshape(-1)[:2]
        goal_pos = np.asarray(dataset.get_row_data([int(rows[s + goal_offset])])[obs_key]).reshape(-1)[:2]
        S = args.num_samples
        # shared candidate set (CEM-style gaussian proposals)
        cand = (
            torch.randn(
                1,
                S,
                H,
                A,
                generator=torch.Generator().manual_seed(int(e) * 1000 + s),
            )
            * args.var_scale
        )
        state = torch.tensor(start_pos, dtype=torch.float32).view(1, 1, 1, 2).expand(1, S, 1, 2)
        goal = torch.tensor(goal_pos, dtype=torch.float32).view(1, 1, 1, 2).expand(1, S, 1, 2)
        info_base = {
            state_wm.obs_key: state,
            state_wm.goal_key: goal,
            "state": state,
            "goal_state": goal,
        }
        # oracle terminal quality on the same candidates
        ocost = oracle_cost.get_cost(dict(info_base), cand).reshape(-1).cpu().numpy()
        for name, mode, metric, _ in conditions_built:
            if name == "oracle":
                continue
            mc = MetricCost(wm, metric, mode=mode, lam=args.lam)
            mcost = (
                mc.get_cost(
                    {k: v.to(device) for k, v in info_base.items()},
                    cand.to(device),
                )
                .reshape(-1)
                .cpu()
                .numpy()
            )
            per_metric[name].append(diagnostics.scsa_pointwise(mcost, ocost))
    return {name: diagnostics.scsa_aggregate(recs) for name, recs in per_metric.items()}


def _run(cfg: DictConfig) -> None:
    args = cfg

    device = pick_device(args.device)
    eval_budget = args.eval_budget or (2 * args.goal_offset)
    wm = load_wm(args.wm, device=device)
    if is_statewm(wm):
        wm_obs_key = getattr(wm, "obs_key", None)
        if not isinstance(wm_obs_key, str):
            raise TypeError("state world model obs_key must be a string")
        obs_key = wm_obs_key
    else:
        obs_key = "proprio"

    dataset = swm.data.load_dataset(args.dataset)
    episodes, starts = sample_eval_episodes(
        dataset,
        args.num_eval,
        args.goal_offset,
        obs_key,
        args.seed,
        args.cross_wall,
    )
    n = len(episodes)
    if n == 0:
        raise SystemExit("no eval episodes matched the criteria (try cross_wall=false)")
    logger.info(f"Evaluating episodes={n} cross_wall={args.cross_wall} device={device} budget={eval_budget}")

    if args.task == "cube":
        # OGBench cube: low-dim manipulation STATE, no rendering needed.
        world = World(
            args.env,
            num_envs=n,
            max_episode_steps=2 * eval_budget,
            env_type="single",
            ob_type="states",
            multiview=False,
            visualize_info=False,
            terminate_at_goal=True,
            add_pixels=False,
        )
        callables = [
            {
                "method": "set_state",
                "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}},
            },
            {
                "method": "set_target_pos",
                "args": {
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": "goal_privileged/block_0_pos"},
                    "target_quat": {"value": "goal_privileged/block_0_quat"},
                },
            },
        ]
        oracle_cost = None
    else:
        world = World(
            args.env,
            num_envs=n,
            max_episode_steps=2 * eval_budget,
            image_shape=(224, 224),
            render_mode="rgb_array",
        )
        callables = [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ]
        geom = read_tworoom_geometry(world)
        oracle_cost = TwoRoomOracleCost(**geom)

    # build conditions
    built: list[Condition] = []
    for spec in args.conditions:
        name, mode, path = parse_condition(str(spec))
        if mode == "oracle":
            if oracle_cost is None:
                logger.warning("Oracle condition requested but no oracle exists for this environment; skipping")
                continue
            built.append((name, mode, None, oracle_cost))
        elif mode == "latent":
            built.append((name, mode, None, MetricCost(wm, None, mode="latent")))
        else:
            if path is None:
                raise ValueError(f"condition {name!r} requires a metric checkpoint path")
            metric = load_metric(path, device=device)
            built.append(
                (
                    name,
                    mode,
                    metric,
                    MetricCost(wm, metric, mode=mode, lam=args.lam),
                )
            )

    config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding,
        history_len=1,
        action_block=args.action_block,
        warm_start=True,
    )

    state_wm = is_statewm(wm)
    transform: dict[str, Callable[[Any], Any]] = {} if state_wm else build_pixel_transform()
    process: dict[str, Any] = {}

    rows: list[list[str]] = []
    for name, mode, _metric, model in built:
        solver = build_solver(model, args, device)
        policy = LeanWorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            drop_pixels=state_wm,
        )
        world.set_policy(policy)
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=episodes,
            start_steps=starts,
            goal_offset=args.goal_offset,
            eval_budget=eval_budget,
            callables=callables,
        )
        logger.success(f"{name}: success_rate={metrics['success_rate']:.1f}%")
        rows.append([name, mode, f"{metrics['success_rate']:.1f}"])

    results_table = tabulate(rows, headers=["condition", "mode", "success%"], tablefmt="github")
    logger.info("MPC success rate for TRM conditions")
    log_multiline(results_table)

    scsa_rows: list[list[str]] = []
    if args.scsa and oracle_cost is not None:
        scsa = run_scsa(
            wm,
            built,
            oracle_cost,
            dataset,
            episodes,
            starts,
            args.goal_offset,
            obs_key,
            args,
            device,
        )
        for name, stats in scsa.items():
            scsa_rows.append(
                [
                    name,
                    f"{stats['spearman']:.3f}",
                    f"{stats['best_rank_pct']:.3f}",
                    f"{stats['regret']:.2f}",
                ]
            )
        scsa_table = tabulate(
            scsa_rows,
            headers=["metric", "spearman", "best_rank_pct", "regret"],
            tablefmt="github",
        )
        logger.info("SCSA against the task-state oracle")
        log_multiline(scsa_table)
    else:
        scsa_table = ""

    results_path = Path(args.run.metrics) / args.output.filename
    contents = results_table + "\n"
    if scsa_table:
        contents += "\n" + scsa_table + "\n"
    results_path.write_text(contents)
    logger.info(f"TRM results saved to {results_path}")


def main() -> object:
    return run_hydra(dispatch, config_name="eval/trm")


if __name__ == "__main__":
    main()
