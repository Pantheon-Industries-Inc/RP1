"""Evaluate TRM terminal-cost conditions with CEM-MPC (CPU/MPS friendly).

For each *cost condition* we wrap the frozen world model in a ``MetricCost`` and
run the SAME CEM planner / sampler / budget, changing only the terminal cost --
isolating the planner-facing metric. Reports per-condition success rate, and
(when an oracle is available, i.e. TwoRoom) the Same-Candidate Selection Audit
(SCSA): Spearman / best-rank-percentile / regret vs the ground-truth task-state
oracle on a shared candidate set.

Conditions (``--condition NAME[=METRIC.pt]``):
    latent                      raw latent MSE baseline (the broken cost)
    trm_regression=PATH         replacement with the regression head
    trm_td=PATH                 replacement with the TD value
    trm_contrastive=PATH        replacement with the contrastive critic
    hybrid=PATH                 std(c_lat) + lambda*std(metric)
    shuffled=PATH               negative control (shuffled-label head)
    oracle                      ground-truth task-state planner (TwoRoom only)

Example::

    python scripts/trm/eval_trm.py --env swm/TwoRoom-v1 --wm statewm_tworoom \
        --dataset tworoom_expert.lance --num-eval 20 --cross-wall \
        --condition latent --condition trm_regression=metrics/reg.pt \
        --condition trm_td=metrics/td.pt --condition trm_contrastive=metrics/con.pt \
        --condition oracle --scsa --out results/tworoom.txt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tabulate import tabulate

import stable_worldmodel as swm
from stable_worldmodel.trm import MetricCost, diagnostics, load_metric
from stable_worldmodel.trm.oracle import TwoRoomOracleCost, geodesic_distance

from _common import is_statewm, load_wm, pick_device

WALL_CENTER = 112.0


class LeanWorldModelPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy that optionally drops bulky image keys before planning.

    The solver expands *every* info tensor by ``num_samples``; for a
    state/proprio world model the ``pixels``/``goal`` images are unused and
    would blow up memory, so we strip them after preprocessing. For a pixel
    world model (LeWM) they are required, so set ``drop_pixels=False``.
    """

    drop_pixels: bool = True

    def _prepare_info(self, info_dict):
        out = super()._prepare_info(info_dict)
        if self.drop_pixels:
            for k in list(out.keys()):
                if k.startswith("pixels") or k in ("goal", "goal_pixels"):
                    out.pop(k, None)
        return out


def parse_condition(spec: str):
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


def read_tworoom_geometry(world):
    """Pull door points / speed / radius from a reset TwoRoom env."""
    env = world.envs.envs[0].unwrapped
    env.reset()
    wall_axis = int(env.wall_axis)
    n = int(env.num_doors)
    door_pts = []
    for i in range(n):
        c = float(env.door_positions[i])
        door_pts.append([WALL_CENTER, c] if wall_axis == 1 else [c, WALL_CENTER])
    speed = float(env.variation_space["agent"]["speed"].value.item())
    radius = float(env.variation_space["agent"]["radius"].value.item())
    door_half = float(env.door_sizes[0]) if n else 21.0
    return {
        "door_points": np.array(door_pts, dtype=np.float32),
        "speed": speed, "agent_radius": radius, "door_half": door_half,
        "wall_thickness": int(env.wall_thickness),
    }


def sample_eval_episodes(dataset, num_eval, goal_offset, obs_key, seed, cross_wall=False):
    """Pick (episode, start_step) pairs with enough room before the goal."""
    from stable_worldmodel.trm.latent_cache import _episode_col

    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    ep_ids = np.unique(ep)
    lengths = {int(e): int((ep == e).sum()) for e in ep_ids}

    candidates = []  # (episode, start_step)
    for e in ep_ids:
        L = lengths[int(e)]
        for s in range(0, L - goal_offset - 1):
            candidates.append((int(e), s))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    chosen_ep, chosen_start = [], []
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


def build_pixel_transform(img_size=224):
    """ImageNet transform for pixels/goal (DINO-WM / pixel WMs). Actions stay raw
    to match how our WMs are trained, so no StandardScaler `process` is needed."""
    from torchvision.transforms import v2 as T
    img_tf = T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), T.Resize(size=img_size),
    ])
    return {"pixels": img_tf, "goal": img_tf}


def build_solver(model, args, device):
    """Build the planning solver. ``cem`` is the planner used in the TRM paper;
    ``predictive_sampling`` is the MPC shooting method (sample N, pick argmin)."""
    name = args.solver
    if name == "cem":
        return swm.solver.CEMSolver(
            model=model, num_samples=args.num_samples, n_steps=args.cem_steps,
            topk=args.topk, var_scale=args.var_scale, device=device, seed=args.seed,
        )
    if name == "predictive_sampling":
        return swm.solver.PredictiveSamplingSolver(
            model=model, num_samples=args.num_samples, noise_scale=args.var_scale,
            device=device, seed=args.seed,
        )
    if name == "mppi":
        return swm.solver.MPPISolver(
            model=model, num_samples=args.num_samples, n_steps=args.cem_steps,
            topk=args.topk, var_scale=args.var_scale, device=device, seed=args.seed,
        )
    if name == "icem":
        return swm.solver.ICEMSolver(
            model=model, num_samples=args.num_samples, n_steps=args.cem_steps,
            topk=args.topk, var_scale=args.var_scale, device=device, seed=args.seed,
        )
    raise ValueError(f"unknown solver '{name}'")


def run_scsa(wm, conditions_built, oracle_cost, dataset, episodes, starts, goal_offset, obs_key, args, device):
    """Same-Candidate Selection Audit over sampled planning states."""
    from stable_worldmodel.trm.latent_cache import _episode_col

    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    rng = np.random.default_rng(args.seed + 7)
    H, A = args.horizon, 2
    per_metric = {name: [] for name, _, _, _ in conditions_built if name != "oracle"}

    for e, s in zip(episodes, starts):
        rows = np.nonzero(ep == e)[0][np.argsort(st[ep == e])]
        start_pos = np.asarray(dataset.get_row_data([int(rows[s])])[obs_key]).reshape(-1)[:2]
        goal_pos = np.asarray(dataset.get_row_data([int(rows[s + goal_offset])])[obs_key]).reshape(-1)[:2]
        S = args.num_samples
        # shared candidate set (CEM-style gaussian proposals)
        cand = torch.randn(1, S, H, A, generator=torch.Generator().manual_seed(int(e) * 1000 + s)) * args.var_scale
        state = torch.tensor(start_pos, dtype=torch.float32).view(1, 1, 1, 2).expand(1, S, 1, 2)
        goal = torch.tensor(goal_pos, dtype=torch.float32).view(1, 1, 1, 2).expand(1, S, 1, 2)
        info_base = {wm.obs_key: state, wm.goal_key: goal,
                     "state": state, "goal_state": goal}
        # oracle terminal quality on the same candidates
        ocost = oracle_cost.get_cost(dict(info_base), cand).reshape(-1).cpu().numpy()
        for name, mode, metric, _ in conditions_built:
            if name == "oracle":
                continue
            mc = MetricCost(wm, metric, mode=mode, lam=args.lam)
            mcost = mc.get_cost({k: v.to(device) for k, v in info_base.items()},
                                cand.to(device)).reshape(-1).cpu().numpy()
            per_metric[name].append(diagnostics.scsa_pointwise(mcost, ocost))
    return {name: diagnostics.scsa_aggregate(recs) for name, recs in per_metric.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="swm/TwoRoom-v1")
    p.add_argument("--wm", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--condition", action="append", default=[], dest="conditions")
    p.add_argument("--solver", default="cem", choices=["cem", "predictive_sampling", "mppi", "icem"])
    p.add_argument("--num-eval", type=int, default=20)
    p.add_argument("--goal-offset", type=int, default=20)
    p.add_argument("--eval-budget", type=int, default=None)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--receding", type=int, default=5)
    p.add_argument("--action-block", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--cem-steps", type=int, default=10)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--task", default="tworoom", choices=["tworoom", "cube"],
                   help="task wiring (env kwargs + callables + oracle availability)")
    p.add_argument("--cross-wall", action="store_true", help="select only cross-wall goals (hard manifest)")
    p.add_argument("--scsa", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = pick_device(args.device)
    eval_budget = args.eval_budget or (2 * args.goal_offset)
    wm = load_wm(args.wm, device=device)
    obs_key = wm.obs_key if is_statewm(wm) else "proprio"

    dataset = swm.data.load_dataset(args.dataset)
    episodes, starts = sample_eval_episodes(
        dataset, args.num_eval, args.goal_offset, obs_key, args.seed, args.cross_wall
    )
    n = len(episodes)
    if n == 0:
        raise SystemExit("no eval episodes matched the criteria (try without --cross-wall)")
    logging.info(f"evaluating {n} episodes (cross_wall={args.cross_wall}) on {device}; budget={eval_budget}")

    if args.task == "cube":
        # OGBench cube: low-dim manipulation STATE, no rendering needed.
        world = swm.World(args.env, num_envs=n, max_episode_steps=2 * eval_budget,
                          env_type="single", ob_type="states", multiview=False,
                          visualize_info=False, terminate_at_goal=True,
                          add_pixels=False)
        callables = [
            {"method": "set_state", "args": {"qpos": {"value": "qpos"},
                                             "qvel": {"value": "qvel"}}},
            {"method": "set_target_pos", "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged/block_0_pos"},
                "target_quat": {"value": "goal_privileged/block_0_quat"}}},
        ]
        oracle_cost = None
    else:
        world = swm.World(args.env, num_envs=n, max_episode_steps=2 * eval_budget,
                          image_shape=(224, 224), render_mode="rgb_array")
        callables = [
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ]
        geom = read_tworoom_geometry(world)
        oracle_cost = TwoRoomOracleCost(**geom)

    # build conditions
    built = []  # (name, mode, metric_module_or_None, planner_model)
    for spec in args.conditions:
        name, mode, path = parse_condition(spec)
        if mode == "oracle":
            if oracle_cost is None:
                logging.warning("oracle condition requested but no oracle for this env; skipping")
                continue
            built.append((name, mode, None, oracle_cost))
        elif mode == "latent":
            built.append((name, mode, None, MetricCost(wm, None, mode="latent")))
        else:
            metric = load_metric(path, device=device)
            built.append((name, mode, metric, MetricCost(wm, metric, mode=mode, lam=args.lam)))

    config = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding,
                            history_len=1, action_block=args.action_block, warm_start=True)

    state_wm = is_statewm(wm)
    transform, process = ({}, {}) if state_wm else (build_pixel_transform(), {})

    rows = []
    for name, mode, metric, model in built:
        solver = build_solver(model, args, device)
        policy = LeanWorldModelPolicy(solver=solver, config=config, process=process,
                                      transform=transform, drop_pixels=state_wm)
        world.set_policy(policy)
        metrics = world.evaluate(
            dataset=dataset, episodes_idx=episodes, start_steps=starts,
            goal_offset=args.goal_offset, eval_budget=eval_budget, callables=callables,
        )
        logging.success(f"[{name}] success_rate = {metrics['success_rate']:.1f}%")
        rows.append([name, mode, f"{metrics['success_rate']:.1f}"])

    print("\n=== MPC success rate (TRM conditions) ===")
    print(tabulate(rows, headers=["condition", "mode", "success%"], tablefmt="github"))

    scsa_rows = []
    if args.scsa and oracle_cost is not None:
        scsa = run_scsa(wm, built, oracle_cost, dataset, episodes, starts,
                        args.goal_offset, obs_key, args, device)
        for name, stats in scsa.items():
            scsa_rows.append([name, f"{stats['spearman']:.3f}", f"{stats['best_rank_pct']:.3f}",
                              f"{stats['regret']:.2f}"])
        print("\n=== SCSA (vs task-state oracle, shared candidates) ===")
        print(tabulate(scsa_rows, headers=["metric", "spearman", "best_rank_pct", "regret"], tablefmt="github"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(f"\n==== {args.env} wm={args.wm} n={n} cross_wall={args.cross_wall} "
                    f"horizon={args.horizon} samples={args.num_samples} ====\n")
            f.write(tabulate(rows, headers=["condition", "mode", "success%"], tablefmt="github") + "\n")
            if scsa_rows:
                f.write(tabulate(scsa_rows, headers=["metric", "spearman", "best_rank_pct", "regret"], tablefmt="github") + "\n")
        logging.info(f"results appended to {args.out}")


if __name__ == "__main__":
    main()
