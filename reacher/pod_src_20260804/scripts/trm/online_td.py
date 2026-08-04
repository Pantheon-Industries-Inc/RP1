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
*same* world model. ``--offline-baseline`` first trains TD on the logged buffer
only and reports it as the round "-1" reference.

Example (DINO-WM)::

    python scripts/trm/online_td.py --wm dinowm_tworoom --dataset tworoom_pixels.lance \
        --offline-baseline --rounds 6 --device cuda --out results/online_td_dino.txt
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from loguru import logger as logging
from tabulate import tabulate

import stable_worldmodel as swm
from stable_worldmodel.trm import LatentCache, MetricCost, PairwiseMetricHead, load_metric, save_metric
from stable_worldmodel.trm.latent_cache import _episode_col
from stable_worldmodel.trm.learners.td import _expectile_loss
from stable_worldmodel.trm.samplers import TransitionSampler

from _common import is_statewm, load_wm, pick_device

WALL = 112.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sample_pairs(dataset, n, goal_offset, seed, cross_wall=True):
    ep = _episode_col(dataset).reshape(-1)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    pro = np.asarray(dataset.get_col_data("proprio")).reshape(len(ep), -1)[:, :2]
    rng = np.random.default_rng(seed)
    order = {int(e): np.nonzero(ep == e)[0][np.argsort(st[ep == e])] for e in np.unique(ep)}
    cand = [(int(e), s) for e, rows in order.items() for s in range(0, len(rows) - goal_offset - 1)]
    rng.shuffle(cand)
    starts, goals = [], []
    for e, s in cand:
        if len(starts) >= n:
            break
        rows = order[e]
        sp, gp = pro[rows[s]], pro[rows[s + goal_offset]]
        if cross_wall and (sp[0] < WALL) == (gp[0] < WALL):
            continue
        starts.append(sp); goals.append(gp)
    return np.array(starts, np.float32), np.array(goals, np.float32)


class ObsCodec:
    """Encode collected observations into frozen-WM latents; build per-step
    planner info. Abstracts state (proprio) vs DINO-WM (pixels)."""

    def __init__(self, wm, device):
        self.wm = wm
        self.device = device
        self.is_state = is_statewm(wm)
        if not self.is_state:
            from torchvision.transforms import v2 as T
            self.tf = T.Compose([
                T.ToImage(), T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), T.Resize(size=224),
            ])

    @torch.no_grad()
    def encode(self, obs_seq: np.ndarray) -> torch.Tensor:
        """obs_seq: state -> (T,2) positions; dino -> (T,H,W,C) frames. -> (T,D)."""
        if self.is_state:
            p = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
            return self.wm.encode({self.wm.obs_key: p.unsqueeze(1)})["emb"][:, 0].cpu()
        # pixel WM (LeWM CLS / DINO-WM pooled): generic encode handles both
        imgs = torch.stack([self.tf(f) for f in obs_seq]).to(self.device)  # (T,C,H,W)
        return self.wm.encode({"pixels": imgs.unsqueeze(1)})["emb"][:, 0].float().cpu()

    def render_goals(self, world, goals):
        if self.is_state:
            return None
        frames = []
        for i, env in enumerate(world.envs.envs):
            chw = env.unwrapped._render_frame(agent_pos=torch.as_tensor(goals[i], dtype=torch.float32))
            frames.append(chw.cpu().numpy().transpose(1, 2, 0))  # HWC uint8
        return np.stack(frames)

    def step_info(self, cur_obs, cur_pos, goals, goal_imgs, done, needs_flush):
        # 'state'/'goal_state' (agent xy) are always provided so the task-state
        # oracle works; the WM ignores them.
        info = {
            "state": torch.as_tensor(cur_pos, dtype=torch.float32)[:, None, :],
            "goal_state": torch.as_tensor(goals, dtype=torch.float32)[:, None, :],
            "terminated": done.copy(), "_needs_flush": needs_flush,
        }
        if self.is_state:
            info["proprio"] = torch.as_tensor(cur_obs, dtype=torch.float32)[:, None, :]
            info["goal_proprio"] = torch.as_tensor(goals, dtype=torch.float32)[:, None, :]
            return info
        n = len(cur_obs)
        info["pixels"] = np.asarray(cur_obs)[:, None, ...]
        info["goal"] = goal_imgs[:, None, ...]
        # LeWM.get_cost pops 'action' from the goal dict; a dummy suffices
        # (rollout overwrites info['action'] with the candidate actions).
        info["action"] = torch.zeros(n, 1, 2)
        return info


def make_policy(wm, value, args, device, codec):
    from eval_trm import LeanWorldModelPolicy, build_pixel_transform
    model = MetricCost(wm, value.eval(), mode="replacement")
    solver = swm.solver.CEMSolver(model=model, num_samples=args.collect_samples,
                                  n_steps=args.collect_cem_steps, topk=args.collect_topk,
                                  device=device, seed=args.seed)
    cfg = swm.PlanConfig(horizon=args.horizon, receding_horizon=args.receding,
                         history_len=1, action_block=args.action_block, warm_start=True)
    transform = {} if codec.is_state else build_pixel_transform()
    return LeanWorldModelPolicy(solver=solver, config=cfg, process={}, transform=transform,
                                drop_pixels=codec.is_state)


def collect_mpc(world, policy, codec, starts, goals, budget):
    """Goal-conditioned MPC rollout; returns per-env observation trajectories."""
    n = world.num_envs
    world.set_policy(policy)
    world.envs.reset()
    for i, env in enumerate(world.envs.envs):
        u = env.unwrapped
        u._set_state(starts[i]); u._set_goal_state(goals[i])
    goal_imgs = codec.render_goals(world, goals)

    cur_pos = starts.copy()  # agent xy, tracked for the oracle / state WM
    if codec.is_state:
        cur = starts.copy()
        trajs = [[starts[i].copy()] for i in range(n)]
    else:
        cur = np.stack([
            world.envs.envs[i].unwrapped._render_frame(
                agent_pos=torch.as_tensor(starts[i], dtype=torch.float32)).cpu().numpy().transpose(1, 2, 0)
            for i in range(n)])
        trajs = [[cur[i].copy()] for i in range(n)]

    done = np.zeros(n, bool)
    needs_flush = np.ones(n, bool)
    for t in range(budget):
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


def append_trajectories(cache, trajs, codec, next_ep):
    zs, eps, sts = [], [], []
    e = next_ep
    for tr in trajs:
        if len(tr) < 2:
            continue
        z = codec.encode(np.stack(tr))
        zs.append(z)
        eps.append(np.full(len(tr), e, np.int64)); sts.append(np.arange(len(tr), dtype=np.int64))
        e += 1
    if not zs:
        return cache, e
    z = torch.cat(zs); ep = torch.from_numpy(np.concatenate(eps)); stp = torch.from_numpy(np.concatenate(sts))
    if cache is None:
        return LatentCache(z=z, episode_idx=ep, step_idx=stp), e
    return LatentCache(z=torch.cat([cache.z, z]), episode_idx=torch.cat([cache.episode_idx, ep]),
                       step_idx=torch.cat([cache.step_idx, stp])), e


def td_updates(value, target, sampler, steps, args, device):
    opt = torch.optim.AdamW(value.parameters(), lr=1e-3, weight_decay=1e-4)
    value.train()
    loss = torch.tensor(0.0)
    for _ in range(steps):
        b = sampler.sample(args.batch_size)
        z_t, z_tp1, z_g, done = (b["z_t"].to(device), b["z_tp1"].to(device),
                                 b["z_g"].to(device), b["done"].to(device))
        with torch.no_grad():
            tgt = done * 1.0 + (1 - done) * (1 + args.gamma * target(z_tp1, z_g))
        loss = _expectile_loss(value(z_t, z_g) - tgt, args.expectile, 1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), value.parameters()):
                tp.mul_(1 - args.tau).add_(args.tau * sp)
    value.eval()
    return float(loss)


def eval_success(world, policy, codec, starts, goals, budget):
    _, done = collect_mpc(world, policy, codec, starts, goals, budget)
    return 100.0 * float(done.mean())


def logged_trajectories(dataset, codec):
    """Decode logged trajectories as obs sequences for the offline baseline buffer."""
    ep = _episode_col(dataset).reshape(-1); st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    order = {int(e): np.nonzero(ep == e)[0][np.argsort(st[ep == e])] for e in np.unique(ep)}
    if codec.is_state:
        pro = np.asarray(dataset.get_col_data("proprio")).reshape(len(ep), -1)[:, :2]
        return [pro[rows] for rows in order.values()]
    return None  # for dino we seed the offline buffer from the precomputed feature cache instead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--offline-cache", default=None, help="precomputed feature cache for DINO offline seed")
    p.add_argument("--offline-baseline", action="store_true")
    p.add_argument("--init-metric", default=None, help="load this TD value as the starting point (= offline metric)")
    p.add_argument("--save-value", default=None, help="save the final online-TD value head here")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--td-steps-per-round", type=int, default=1500)
    p.add_argument("--offline-steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--goal-offset", type=int, default=20)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--receding", type=int, default=20)
    p.add_argument("--action-block", type=int, default=1, help="frameskip in MPC (LeWM uses 5)")
    p.add_argument("--collect-samples", type=int, default=120)
    p.add_argument("--collect-cem-steps", type=int, default=6)
    p.add_argument("--collect-topk", type=int, default=15)
    p.add_argument("--eval-budget", type=int, default=35)
    p.add_argument("--n-eval", type=int, default=40)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    wm = load_wm(args.wm, device=device)
    dataset = swm.data.load_dataset(args.dataset)
    codec = ObsCodec(wm, device)
    latent_dim = codec.encode(
        (np.zeros((1, 2), np.float32) if codec.is_state else np.zeros((1, 224, 224, 3), np.uint8))
    ).shape[-1]

    eval_starts, eval_goals = sample_pairs(dataset, args.n_eval, args.goal_offset, seed=999)
    n_envs = len(eval_starts)
    world = swm.World("swm/TwoRoom-v1", num_envs=n_envs, max_episode_steps=4 * args.eval_budget,
                      image_shape=(224, 224), render_mode="rgb_array")

    if args.init_metric:
        value = load_metric(args.init_metric, device=device)
        logging.info(f"initialised value from {args.init_metric}")
    else:
        value = PairwiseMetricHead(latent_dim).to(device)
    target = copy.deepcopy(value)
    for q in target.parameters():
        q.requires_grad_(False)

    rows = []
    cache, next_ep = None, 0

    # seed the buffer with the logged data (so online rounds extend it)
    if args.offline_cache or codec.is_state:
        if codec.is_state:
            cache, _ = append_trajectories(None, logged_trajectories(dataset, codec), codec, 0)
        else:
            cache = LatentCache.load(args.offline_cache)
        next_ep = int(cache.episode_idx.max()) + 1

    # ---- offline baseline row ----
    if args.init_metric:
        # the loaded metric IS the offline TD; eval it as-is (no extra training)
        pol = make_policy(wm, value, args, device, codec)
        succ = eval_success(world, pol, codec, eval_starts, eval_goals, args.eval_budget)
        logging.success(f"[offline-TD(loaded)] eval_succ={succ:.1f}%")
        rows.append(["offline", len(cache.z) if cache else 0, "-", f"{succ:.1f}", "-"])
    elif args.offline_baseline and cache is not None:
        sampler = TransitionSampler(cache, p_random_goal=0.1, seed=args.seed)
        loss = td_updates(value, target, sampler, args.offline_steps, args, device)
        pol = make_policy(wm, value, args, device, codec)
        succ = eval_success(world, pol, codec, eval_starts, eval_goals, args.eval_budget)
        logging.success(f"[offline-TD] buffer={len(cache.z)} eval_succ={succ:.1f}% loss={loss:.4f}")
        rows.append(["offline", len(cache.z), "-", f"{succ:.1f}", f"{loss:.4f}"])

    # ---- online rounds: collect MPC data -> append -> TD -> eval ----
    rng = np.random.default_rng(args.seed + 1)
    for rnd in range(args.rounds):
        cs, cg = sample_pairs(dataset, n_envs, args.goal_offset, seed=int(rng.integers(1e6)))
        pol = make_policy(wm, value, args, device, codec)
        trajs, cdone = collect_mpc(world, pol, codec, cs, cg, args.eval_budget)
        cache, next_ep = append_trajectories(cache, trajs, codec, next_ep)
        sampler = TransitionSampler(cache, p_random_goal=0.1, seed=args.seed + rnd)
        loss = td_updates(value, target, sampler, args.td_steps_per_round, args, device)
        pol = make_policy(wm, value, args, device, codec)
        succ = eval_success(world, pol, codec, eval_starts, eval_goals, args.eval_budget)
        logging.success(f"round {rnd}: buffer={len(cache.z)} collect={100*cdone.mean():.1f}% "
                        f"eval_succ={succ:.1f}% loss={loss:.4f}")
        rows.append([str(rnd), len(cache.z), f"{100*cdone.mean():.1f}", f"{succ:.1f}", f"{loss:.4f}"])

    table = tabulate(rows, headers=["round", "buffer", "collect%", "eval%", "td_loss"], tablefmt="github")
    print(table)
    if args.save_value:
        save_metric(value.cpu(), "td", value.latent_dim,
                    {"hidden_dim": 256, "depth": 2, "softplus": True, "symmetric": False},
                    args.save_value)
        logging.success(f"saved online-TD value -> {args.save_value}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(f"\n==== ONLINE TD wm={args.wm} rounds={args.rounds} offline_baseline={args.offline_baseline} ====\n")
            f.write(table + "\n")


if __name__ == "__main__":
    main()
