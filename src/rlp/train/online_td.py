"""Online TD learning of the reachability metric (vs offline TD), same WM.

Offline TD trains the goal-conditioned temporal-distance value on a *fixed*
logged dataset; the MPC planner then visits states off that distribution, where
the value can be miscalibrated. **Online TD** closes the loop: each round we
(1) run MPC with the *current* value to collect fresh goal-conditioned
trajectories, (2) encode them with the frozen world model and append to a
growing buffer (hindsight goal relabeling via TransitionSampler), (3) take TD
updates, (4) evaluate MPC success. The metric is thus trained on exactly the
states the planner induces.

Works with both the low-dim state WM (records proprio) and DINO-WM (records
pixels + renders goal frames), so online vs offline TD can be compared on the
*same* world model. ``offline_baseline=true`` first trains TD on the logged buffer
only and reports it as the round "-1" reference.

Example (DINO-WM)::

    pixi run train model=online_td wm=dinowm_tworoom \
        dataset=tworoom_pixels.lance offline_baseline=true rounds=6 device=cuda
"""

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from stable_worldmodel.policy import BasePolicy
from tabulate import tabulate
from torch import nn

from rlp.config import dispatch, run_hydra
from rlp.core.solver.lip import EncoderWorldModel, ValueFunction
from rlp.core.value import MetricCost, PairwiseMetricHead, load_metric, save_metric
from rlp.core.value.learners.td import _expectile_loss
from rlp.core.value.samplers import TransitionSampler
from rlp.core.world_model.runtime import is_statewm, load_wm, pick_device
from rlp.data import LatentCache
from rlp.data.latent_cache import _episode_col
from rlp.data.protocols import Dataset
from rlp.environment import World
from rlp.logging import log_multiline, logger

WALL = 112.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class MPCSettings:
    collect_samples: int
    collect_cem_steps: int
    collect_topk: int
    horizon: int
    receding: int
    action_block: int
    seed: int


def sample_pairs(
    dataset: Dataset, n: int, goal_offset: int, seed: int, cross_wall: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    pro = np.asarray(dataset.get_col_data("proprio")).reshape(len(ep), -1)[:, :2]
    rng = np.random.default_rng(seed)
    order = {int(e): np.nonzero(ep == e)[0][np.argsort(st[ep == e])] for e in np.unique(ep)}
    cand = [(int(e), s) for e, rows in order.items() for s in range(0, len(rows) - goal_offset - 1)]
    rng.shuffle(cand)
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    for e, s in cand:
        if len(starts) >= n:
            break
        rows = order[e]
        sp, gp = pro[rows[s]], pro[rows[s + goal_offset]]
        if cross_wall and (sp[0] < WALL) == (gp[0] < WALL):
            continue
        starts.append(sp)
        goals.append(gp)
    return np.array(starts, np.float32), np.array(goals, np.float32)


class ObsCodec:
    """Encode collected observations into frozen-WM latents; build per-step
    planner info. Abstracts state (proprio) vs DINO-WM (pixels)."""

    def __init__(self, wm: nn.Module, device: str) -> None:
        if not callable(getattr(wm, "encode", None)):
            raise TypeError("observation codec requires a world model with encode()")
        self.wm = cast(EncoderWorldModel, wm)
        self.device = device
        self.is_state = is_statewm(wm)
        self.tf: Callable[[object], torch.Tensor] | None = None
        if not self.is_state:
            from torchvision.transforms import v2 as T

            self.tf = T.Compose(
                [
                    T.ToImage(),
                    T.ToDtype(torch.float32, scale=True),
                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    T.Resize(size=224),
                ]
            )

    @torch.no_grad()
    def encode(self, obs_seq: np.ndarray) -> torch.Tensor:
        """obs_seq: state -> (T,2) positions; dino -> (T,H,W,C) frames. -> (T,D)."""
        if self.is_state:
            obs_key = getattr(self.wm, "obs_key", None)
            if not isinstance(obs_key, str):
                raise TypeError("state world model obs_key must be a string")
            p = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
            return self.wm.encode({obs_key: p.unsqueeze(1)})["emb"][:, 0].cpu()
        # pixel WM (LeWM CLS / DINO-WM pooled): generic encode handles both
        if self.tf is None:
            raise RuntimeError("pixel transform is unavailable")
        imgs = torch.stack([self.tf(f) for f in obs_seq]).to(self.device)
        return self.wm.encode({"pixels": imgs.unsqueeze(1)})["emb"][:, 0].float().cpu()

    def render_goals(self, world: World, goals: np.ndarray) -> np.ndarray | None:
        if self.is_state:
            return None
        frames = []
        for i, env in enumerate(world.envs.envs):
            chw = env.unwrapped._render_frame(agent_pos=torch.as_tensor(goals[i], dtype=torch.float32))
            frames.append(chw.cpu().numpy().transpose(1, 2, 0))  # HWC uint8
        return np.stack(frames)

    def step_info(
        self,
        cur_obs: np.ndarray,
        cur_pos: np.ndarray,
        goals: np.ndarray,
        goal_imgs: np.ndarray | None,
        done: np.ndarray,
        needs_flush: np.ndarray,
    ) -> dict[str, Any]:
        # 'state'/'goal_state' (agent xy) are always provided so the task-state
        # oracle works; the WM ignores them.
        info: dict[str, Any] = {
            "state": torch.as_tensor(cur_pos, dtype=torch.float32)[:, None, :],
            "goal_state": torch.as_tensor(goals, dtype=torch.float32)[:, None, :],
            "terminated": done.copy(),
            "_needs_flush": needs_flush,
        }
        if self.is_state:
            info["proprio"] = torch.as_tensor(cur_obs, dtype=torch.float32)[:, None, :]
            info["goal_proprio"] = torch.as_tensor(goals, dtype=torch.float32)[:, None, :]
            return info
        n = len(cur_obs)
        if goal_imgs is None:
            raise RuntimeError("pixel world model requires rendered goal images")
        info["pixels"] = np.asarray(cur_obs)[:, None, ...]
        info["goal"] = goal_imgs[:, None, ...]
        # LeWM.get_cost pops 'action' from the goal dict; a dummy suffices
        # (rollout overwrites info['action'] with the candidate actions).
        info["action"] = torch.zeros(n, 1, 2)
        return info


def make_policy(wm: nn.Module, value: nn.Module, args: MPCSettings, device: str, codec: ObsCodec) -> BasePolicy:
    from rlp.eval.trm import LeanWorldModelPolicy, build_pixel_transform

    model = MetricCost(wm, value.eval(), mode="replacement")
    solver = swm.solver.CEMSolver(
        model=model,
        num_samples=args.collect_samples,
        n_steps=args.collect_cem_steps,
        topk=args.collect_topk,
        device=device,
        seed=args.seed,
    )
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


def collect_mpc(
    world: World,
    policy: BasePolicy,
    codec: ObsCodec,
    starts: np.ndarray,
    goals: np.ndarray,
    budget: int,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Goal-conditioned MPC rollout; returns per-env observation trajectories."""
    n = world.num_envs
    world.set_policy(policy)
    world.envs.reset()
    for i, env in enumerate(world.envs.envs):
        u = env.unwrapped
        u._set_state(starts[i])
        u._set_goal_state(goals[i])
    goal_imgs = codec.render_goals(world, goals)

    cur_pos = starts.copy()  # agent xy, tracked for the oracle / state WM
    if codec.is_state:
        cur = starts.copy()
        trajs: list[list[np.ndarray]] = [[starts[i].copy()] for i in range(n)]
    else:
        cur = np.stack(
            [
                world.envs.envs[i]
                .unwrapped._render_frame(agent_pos=torch.as_tensor(starts[i], dtype=torch.float32))
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                for i in range(n)
            ]
        )
        trajs = [[cur[i].copy()] for i in range(n)]

    done = np.zeros(n, bool)
    needs_flush = np.ones(n, bool)
    for _ in range(budget):
        info = codec.step_info(cur, cur_pos, goals, goal_imgs, done, needs_flush)
        needs_flush = np.zeros(n, bool)
        act = policy.get_action(info)
        _, _, term, trunc, infos = world.envs.step(act)
        cur_pos = np.asarray(infos["proprio"])[:, -1, :2]
        nxt = cur_pos if codec.is_state else np.asarray(infos["pixels"])[:, -1, ...]
        for i in range(n):
            if not done[i]:
                trajs[i].append(nxt[i].copy())
                if term[i]:
                    done[i] = True
        cur = nxt
        if done.all():
            break
    return trajs, done


def append_trajectories(
    cache: LatentCache | None,
    trajs: Sequence[Sequence[np.ndarray] | np.ndarray],
    codec: ObsCodec,
    next_ep: int,
) -> tuple[LatentCache, int]:
    zs: list[torch.Tensor] = []
    eps: list[np.ndarray] = []
    sts: list[np.ndarray] = []
    e = next_ep
    for tr in trajs:
        if len(tr) < 2:
            continue
        z = codec.encode(np.stack(list(tr)))
        zs.append(z)
        eps.append(np.full(len(tr), e, np.int64))
        sts.append(np.arange(len(tr), dtype=np.int64))
        e += 1
    if not zs:
        if cache is None:
            raise ValueError("cannot initialize latent cache from empty trajectories")
        return cache, e
    z = torch.cat(zs)
    ep = torch.from_numpy(np.concatenate(eps))
    stp = torch.from_numpy(np.concatenate(sts))
    if cache is None:
        return LatentCache(z=z, episode_idx=ep, step_idx=stp), e
    return LatentCache(
        z=torch.cat([cache.z, z]),
        episode_idx=torch.cat([cache.episode_idx, ep]),
        step_idx=torch.cat([cache.step_idx, stp]),
    ), e


def td_updates(
    value: nn.Module,
    target: nn.Module,
    sampler: TransitionSampler,
    steps: int,
    args: DictConfig,
    device: str,
) -> float:
    opt = torch.optim.AdamW(value.parameters(), lr=1e-3, weight_decay=1e-4)
    value.train()
    value_fn = cast(ValueFunction, value)
    target_fn = cast(ValueFunction, target)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        b = sampler.sample(args.batch_size)
        z_t, z_tp1, z_g, done = (
            b["z_t"].to(device),
            b["z_tp1"].to(device),
            b["z_g"].to(device),
            b["done"].to(device),
        )
        with torch.no_grad():
            tgt = done * 1.0 + (1 - done) * (1 + args.gamma * target_fn(z_tp1, z_g))
        loss = _expectile_loss(value_fn(z_t, z_g) - tgt, args.expectile, 1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature here.
        opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters(), strict=True):
                tp.mul_(1 - args.tau).add_(args.tau * sp)
    value.eval()
    return float(loss)


def eval_success(
    world: World,
    policy: BasePolicy,
    codec: ObsCodec,
    starts: np.ndarray,
    goals: np.ndarray,
    budget: int,
) -> float:
    _, done = collect_mpc(world, policy, codec, starts, goals, budget)
    return 100.0 * float(done.mean())


def logged_trajectories(dataset: Dataset, codec: ObsCodec) -> list[np.ndarray] | None:
    """Decode logged trajectories as obs sequences for the offline baseline buffer."""
    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    order = {int(e): np.nonzero(ep == e)[0][np.argsort(st[ep == e])] for e in np.unique(ep)}
    if codec.is_state:
        pro = np.asarray(dataset.get_col_data("proprio")).reshape(len(ep), -1)[:, :2]
        return [pro[rows] for rows in order.values()]
    return None  # for dino we seed the offline buffer from the precomputed feature cache instead


def _run(cfg: DictConfig) -> None:
    args = OmegaConf.merge(cfg, cfg.core.solver)
    if not isinstance(args, DictConfig):
        raise TypeError("merged solver configuration must be a mapping")
    args.receding = args.receding_horizon
    args.collect_samples = args.num_samples
    args.collect_cem_steps = args.cem_steps
    args.collect_topk = args.topk
    policy_settings = MPCSettings(
        collect_samples=int(args.collect_samples),
        collect_cem_steps=int(args.collect_cem_steps),
        collect_topk=int(args.collect_topk),
        horizon=int(args.horizon),
        receding=int(args.receding),
        action_block=int(args.action_block),
        seed=int(args.seed),
    )

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    dataset = swm.data.load_dataset(args.dataset)
    codec = ObsCodec(wm, device)
    latent_dim = codec.encode(
        np.zeros((1, 2), np.float32) if codec.is_state else np.zeros((1, 224, 224, 3), np.uint8)
    ).shape[-1]

    eval_starts, eval_goals = sample_pairs(dataset, args.n_eval, args.goal_offset, seed=999)
    n_envs = len(eval_starts)
    world = World(
        "swm/TwoRoom-v1",
        num_envs=n_envs,
        max_episode_steps=4 * args.eval_budget,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    if args.init_metric:
        value = load_metric(args.init_metric, device=device)
        logger.info(f"Initialized value from {args.init_metric}")
    else:
        value = PairwiseMetricHead(latent_dim).to(device)
    target = copy.deepcopy(value)
    for q in target.parameters():
        q.requires_grad_(False)

    rows: list[list[object]] = []
    cache: LatentCache | None = None
    next_ep = 0

    # seed the buffer with the logged data (so online rounds extend it)
    if args.offline_cache or codec.is_state:
        if codec.is_state:
            logged = logged_trajectories(dataset, codec)
            if logged is None:
                raise RuntimeError("state codec did not produce logged trajectories")
            cache, _ = append_trajectories(None, logged, codec, 0)
        else:
            cache = LatentCache.load(args.offline_cache)
        next_ep = int(cache.episode_idx.max()) + 1

    # ---- offline baseline row ----
    if args.init_metric:
        # the loaded metric IS the offline TD; eval it as-is (no extra training)
        pol = make_policy(wm, value, policy_settings, device, codec)
        succ = eval_success(world, pol, codec, eval_starts, eval_goals, args.eval_budget)
        logger.success(f"Offline TD loaded: evaluation_success={succ:.1f}%")
        rows.append(["offline", len(cache.z) if cache else 0, "-", f"{succ:.1f}", "-"])
    elif args.offline_baseline and cache is not None:
        sampler = TransitionSampler(cache, p_random_goal=0.1, seed=args.seed)
        loss = td_updates(value, target, sampler, args.offline_steps, args, device)
        pol = make_policy(wm, value, policy_settings, device, codec)
        succ = eval_success(world, pol, codec, eval_starts, eval_goals, args.eval_budget)
        logger.success(f"Offline TD: buffer={len(cache.z)} evaluation_success={succ:.1f}% loss={loss:.4f}")
        rows.append(["offline", len(cache.z), "-", f"{succ:.1f}", f"{loss:.4f}"])

    # ---- online rounds: collect MPC data -> append -> TD -> eval ----
    rng = np.random.default_rng(args.seed + 1)
    for rnd in range(args.rounds):
        cs, cg = sample_pairs(dataset, n_envs, args.goal_offset, seed=int(rng.integers(1_000_000)))
        pol = make_policy(wm, value, policy_settings, device, codec)
        trajs, cdone = collect_mpc(world, pol, codec, cs, cg, args.eval_budget)
        cache, next_ep = append_trajectories(cache, trajs, codec, next_ep)
        sampler = TransitionSampler(cache, p_random_goal=0.1, seed=args.seed + rnd)
        loss = td_updates(value, target, sampler, args.td_steps_per_round, args, device)
        pol = make_policy(wm, value, policy_settings, device, codec)
        succ = eval_success(world, pol, codec, eval_starts, eval_goals, args.eval_budget)
        logger.success(
            f"round {rnd}: buffer={len(cache.z)} collect={100 * cdone.mean():.1f}% "
            f"eval_succ={succ:.1f}% loss={loss:.4f}"
        )
        rows.append(
            [
                str(rnd),
                len(cache.z),
                f"{100 * cdone.mean():.1f}",
                f"{succ:.1f}",
                f"{loss:.4f}",
            ]
        )

    table = tabulate(
        rows,
        headers=["round", "buffer", "collect%", "eval%", "td_loss"],
        tablefmt="github",
    )
    log_multiline(table)
    checkpoint = save_metric(value.cpu(), run_name=args.output.checkpoint, cache_dir=args.run.directory)
    logger.success(f"Saved online-TD value to {checkpoint}")
    results = Path(args.run.metrics) / args.output.filename
    results.write_text(table + "\n")
    logger.info(f"Saved online-TD results to {results}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/online_td")


if __name__ == "__main__":
    main()
