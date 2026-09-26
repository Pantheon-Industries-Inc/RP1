"""rp1-AC: train the TD value (critic) and the rp1 planner (actor) in tandem.

The planner consumes the value's *gradient field* through the world model, so
the two train jointly, DDPG/TD3-style, with the planner as a K-step
learned-optimizer actor:

  critic   d_phi(z, z_g): n-step expectile TD on the dense (fs1) latent cache.
           This TD loss is the only gradient that reaches phi; the actor loss
           goes through the frozen-parameter teacher, so the critic cannot learn
           to make plans look good.
  teacher  d_bar = EMA(d_phi) (Polyak, ``ema_tau``): serves the TD bootstrap
           targets and the actor, and is the value saved for deployment, so
           training matches the solver's plan-time value.
  actor    f_theta: refines a zero plan for K iterations against the teacher.

Schedule: ``pretrain`` critic-only warmup (auto: 2000 if fresh, 0 if warm-started
via ``init_value``), then ``critic_ratio`` critic steps per actor step, critic and
EMA frozen after ``freeze_critic_frac`` of actor steps so the planner settles on
a stationary teacher.

``expand_weight > 0`` turns the actor's imagined terminal latents into extra TD
backups  d(z0,zg) <- H*fs + d_bar(z_T,zg)  -- value expansion on the states the
planner visits. A low expectile makes this a one-sided bound: good plans
tighten d, bad plans barely raise it. It lets the pair co-exploit world-model
errors, so compare against ``expand_weight=0``. ``expand_traj`` adds the H
single-block backups  d(z_t,zg) <- fs + d_bar(z_{t+1},zg)  along the same
rollout, at no extra world-model cost.

``grounding=pusht`` adds the physics-grounding penalty of
:mod:`rp1.core.agent.value.grounding` to the energy the planner is trained on;
the calibrated term travels in the planner checkpoint, so the solver deploys
the same energy.

The planner checkpoint records the value's path, so the solver finds the
teacher it was trained against.
"""

import copy
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import h5py

with suppress(ImportError):
    import hdf5plugin  # noqa: F401  (registers HDF5 compression filters, e.g. cube h5)
from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from rp1.core.agent.planner import PlannerNet
from rp1.core.agent.value import build_metric
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, window_pair, windowed_trajectory_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.data import LatentCache
from rp1.data.base import load_action_stats
from rp1.training.harness.checkpointing import load_metric, load_pretrained, save_metric
from rp1.training.harness.schedule import cosine_interpolate
from rp1.training.phases.agent.grounding import Grounding
from rp1.training.phases.agent.learners.td import expectile_loss
from rp1.training.phases.agent.samplers import NStepGoalSampler
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger

# (z0, imagined trajectory (B, H, D), zg) -- the endpoint backup uses traj[:, -1];
# expand_traj additionally consumes every consecutive pair along the trajectory.
# With a windowed value (frames > 1) the tuple instead carries pre-stacked
# (start window, terminal window, tiled goal), all 2-D.
type ExpandBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ActionBlocks:
    """Normalized action blocks of ``frameskip`` primitive steps, read from an action h5.

    Episodes are those of the block cache; with ``phases > 1`` (a phase-multiplexed
    cache) episode ``e * phases + k`` is source episode ``e`` starting at step ``k``.
    """

    def __init__(self, path: str, frameskip: int, stats: str | None, phases: int) -> None:
        with h5py.File(path, "r") as file:
            self.actions = file["action"][:]
            self.offsets = file["ep_offset"][:]
            self.lengths = file["ep_len"][:] if "ep_len" in file else None
        # nan-aware: some datasets pad episode-terminal steps with NaN actions;
        # those rows are never sampled but must not poison the statistics
        mean, std = np.nanmean(self.actions, 0), np.nanstd(self.actions, 0) + 1e-6
        if stats is not None:
            mean, std = load_action_stats(stats)
            if mean.shape != (self.actions.shape[-1],):
                raise ValueError(f"action statistics have shape {mean.shape}; actions have {self.actions.shape[-1]}")
            logger.info(f"Actions normalized with statistics from {stats}")
        self.mean, self.std = mean, std
        self.normalized = ((self.actions - mean) / std).astype(np.float32)
        self.frameskip = frameskip
        self.phases = phases
        self.dim = self.actions.shape[-1] * frameskip

    def row(self, episode: int, block: int) -> int:
        """The h5 row of the block's first primitive step."""
        source, phase = divmod(episode, self.phases)
        offset = phase + self.frameskip * block
        if self.lengths is not None:
            offset = min(offset, max(0, int(self.lengths[source]) - self.frameskip))
        return int(self.offsets[source] + offset)

    def first_row(self, episode: int) -> int:
        return int(self.offsets[episode // self.phases])

    def __call__(self, episode: int, block: int) -> np.ndarray:
        start = self.row(episode, block)
        return np.asarray(self.normalized[start : start + self.frameskip]).reshape(-1)


@dataclass(frozen=True)
class Problems:
    """A batch of planning problems; ``rows`` are the action-h5 rows where each plan starts."""

    histories: torch.Tensor
    actions: torch.Tensor
    goals: torch.Tensor
    rows: np.ndarray
    has_previous: np.ndarray


class PlanningTasks:
    """Three-frame latent histories, their two action blocks, and a goal latent.

    In-episode goals are ``1..max_delta`` blocks ahead; with ``band_mix`` a band is
    drawn uniformly first and the offset uniformly within it, so each deployment
    horizon gets equal mass instead of the long ones dominating.
    """

    def __init__(
        self,
        cache: LatentCache,
        blocks: ActionBlocks,
        *,
        max_delta: int,
        band_mix: list[int] | None,
        p_cross: float,
        rng: np.random.Generator,
        device: str,
    ) -> None:
        if band_mix and max(band_mix) > max_delta:
            raise ValueError(f"band_mix {band_mix} exceeds max_delta {max_delta}")
        self.z = cache.z.to(device).float()
        episodes = cache.episodes()
        keys = [key for key in episodes if len(episodes[key]) > max_delta + 4]
        if not keys:
            longest = max((len(rows) for rows in episodes.values()), default=0)
            raise ValueError(
                f"no episode is longer than max_delta+4 = {max_delta + 4} blocks "
                f"(longest is {longest}); lower planner.max_delta or use a cache with longer episodes"
            )
        self.rows = {key: np.asarray(episodes[key]) for key in keys}
        self.episodes = np.array(keys)
        self.blocks = blocks
        self.max_delta = max_delta
        self.band_mix = band_mix
        self.p_cross = p_cross
        self.rng = rng
        self.device = device

    def _offset(self, remaining: int) -> int:
        if not self.band_mix:
            return int(self.rng.integers(1, self.max_delta + 1))
        high = min(self.max_delta, remaining)
        if high < 1:
            return 1
        band = self.band_mix[int(self.rng.integers(len(self.band_mix)))]
        return int(self.rng.integers(1, min(band, high) + 1))

    def sample(self, batch: int) -> Problems:
        z, rng = self.z, self.rng
        histories: list[torch.Tensor] = []
        actions: list[np.ndarray] = []
        goals: list[torch.Tensor] = []
        starts = np.empty(batch, dtype=np.int64)
        has_previous = np.empty(batch, dtype=bool)
        for i in range(batch):
            episode = int(self.episodes[rng.integers(len(self.episodes))])
            rows = self.rows[episode]
            length = len(rows)
            t = int(rng.integers(2, length - 2))
            histories.append(torch.stack([z[rows[t - 2]], z[rows[t - 1]], z[rows[t]]]))
            actions.append(np.stack([self.blocks(episode, t - 2), self.blocks(episode, t - 1)]))
            starts[i] = self.blocks.row(episode, t)
            has_previous[i] = starts[i] > self.blocks.first_row(episode)
            if rng.random() < self.p_cross:
                other = self.rows[int(self.episodes[rng.integers(len(self.episodes))])]
                goals.append(z[other[rng.integers(len(other))]])
            else:
                delta = self._offset(length - 1 - t)
                goals.append(z[rows[min(t + delta, length - 1)]])
        return Problems(
            torch.stack(histories),
            torch.from_numpy(np.stack(actions)).to(self.device),
            torch.stack(goals),
            starts,
            has_previous,
        )


@dataclass(frozen=True)
class Returns:
    """Discounted cost and continuation factor of a full plan and of one action block."""

    plan_cost: float
    plan_discount: float
    block_cost: float
    block_discount: float

    @classmethod
    def of(cls, gamma: float, horizon: int, frameskip: int) -> "Returns":
        if gamma >= 1.0:
            return cls(float(horizon * frameskip), 1.0, float(frameskip), 1.0)
        plan_discount, block_discount = gamma ** (horizon * frameskip), gamma**frameskip
        return cls(
            (1.0 - plan_discount) / (1.0 - gamma), plan_discount, (1.0 - block_discount) / (1.0 - gamma), block_discount
        )


class Critic:
    """The TD critic, its EMA teacher, and the optimizer that trains the critic."""

    def __init__(self, a: DictConfig, module: nn.Module, cache: LatentCache | None, frames: int, device: str) -> None:
        module.train()
        self.module = module
        self.teacher = copy.deepcopy(module).to(device)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)  # the actor loss flows through the teacher, never into it
        self.teacher.eval()
        self.cache = cache
        self.frames = frames
        self.a = a
        self.device = device
        self.returns = Returns.of(a.gamma, a.horizon, a.frameskip)
        self.sampler = (
            None
            if cache is None
            else NStepGoalSampler(
                cache,
                n_step=a.n_step,
                p_cross=a.td_p_cross,
                n_buckets=a.td_n_buckets,
                balanced=True,
                seed=a.seed,
                max_delta=a.td_max_delta,
                near_frac=a.near_frac,
                near_max=a.near_max,
            )
        )
        # a parameter-free value (the latent L2 distance) can only ever be frozen
        trainable = any(True for _ in module.parameters())
        if cache is not None and not trainable and a.freeze_critic_frac > 0:
            raise ValueError(f"{type(module).__name__} has no parameters to co-train; set freeze_critic_frac=0")
        self.optimizer = (
            None
            if cache is None or not trainable
            else torch.optim.AdamW(module.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)
        )

    @property
    def value(self) -> ValueFunction:
        return cast(ValueFunction, self.module)

    @property
    def target(self) -> ValueFunction:
        return cast(ValueFunction, self.teacher)

    def windows(self, indices: torch.Tensor) -> torch.Tensor:
        """Dense-cache rows stacked into windows of ``frames`` latents one action block apart.

        The same construction as ``LatentCache.windowed``: oldest first, clamped
        to the episode's first row at episode starts.
        """
        if self.cache is None:
            raise RuntimeError("window rows requested without the dense TD cache")
        step = self.cache.step_idx[indices]
        columns = []
        for k in range(self.frames - 1, -1, -1):
            offset = k * self.a.frameskip
            rows = torch.where(step < offset, indices - step, indices - offset)
            columns.append(self.cache.z[rows])
        return torch.cat(columns, dim=-1)

    def step(self, expand: ExpandBatch | None, tau: float, lr: float) -> float:
        if self.sampler is None or self.optimizer is None:
            raise RuntimeError("critic step during actor-only training")
        a, device = self.a, self.device
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        batch = self.sampler.sample(a.td_batch)
        if self.frames > 1:
            # every query is a window rebuilt from the dense cache; the goal side
            # is the sampled goal frame's own window
            z_t, z_tn, z_g = (self.windows(batch[key]).to(device) for key in ("t_idx", "tn_idx", "g_idx"))
        else:
            z_t, z_tn, z_g = batch["z_t"].to(device), batch["z_tn"].to(device), batch["z_g"].to(device)
        steps, reached, distance = batch["n_eff"].to(device), batch["reached"].to(device), batch["dist"].to(device)
        with torch.no_grad():
            bootstrap = self.target(z_tn, z_g)
            if a.gamma >= 1.0:
                cost, discount = steps, torch.ones_like(steps)
            else:
                discount = a.gamma**steps
                cost = (1.0 - discount) / (1.0 - a.gamma)
            target = reached * distance + (1.0 - reached) * (cost + discount * bootstrap)
        loss = expectile_loss(self.value(z_t, z_g) - target, tau, a.huber_beta)
        if expand is not None:
            loss = loss + a.expand_weight * self._expansion_loss(expand, tau)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # PyTorch 2.7 Tensor.backward lacks a typed signature here.
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.teacher.parameters(), self.module.parameters(), strict=True):
                target_parameter.mul_(1.0 - a.ema_tau).add_(a.ema_tau * parameter)
        return float(loss.item())

    def _expansion_loss(self, expand: ExpandBatch, tau: float) -> torch.Tensor:
        """Value-expansion backups on the actor's imagined rollouts."""
        a, returns = self.a, self.returns
        start, trajectory, goal = expand
        # windowed values receive pre-stacked endpoint windows (2-D)
        endpoint = trajectory if self.frames > 1 else trajectory[:, -1]
        with torch.no_grad():
            target = returns.plan_cost + returns.plan_discount * self.target(endpoint, goal)
        loss = expectile_loss(self.value(start, goal) - target, tau, a.huber_beta)
        if a.expand_traj:
            _, horizon, dim = trajectory.shape
            sources = torch.cat([start.unsqueeze(1), trajectory[:, :-1]], dim=1).reshape(-1, dim)
            goals = goal.repeat_interleave(horizon, dim=0)
            with torch.no_grad():
                targets = returns.block_cost + returns.block_discount * self.target(trajectory.reshape(-1, dim), goals)
            loss = loss + expectile_loss(self.value(sources, goals) - targets, tau, a.huber_beta)
        return loss


def build_critic(a: DictConfig, latent_dim: int, cache: LatentCache | None, device: str) -> tuple[nn.Module, int]:
    """The initial critic and the number of frames its windows stack.

    A warm-started value may take a window of ``m`` latents (width ``m * D``).
    """
    if a.init_value:
        critic = load_metric(a.init_value, device=device)
        value_dim = int(cast(int, critic.latent_dim))
        if value_dim % latent_dim:
            raise ValueError(
                f"init-value width {value_dim} is not an integer multiple of cache latent dim {latent_dim}"
            )
        return critic, value_dim // latent_dim
    if cache is None:
        raise RuntimeError("critic cache unavailable")
    architecture = {
        "head": a.head,
        "hidden_dim": a.hidden_dim,
        "depth": a.depth,
        "embed_dim": a.embedding_dim,
        "softplus": True,
        "symmetric": False,
        "sym_frac": a.sym_frac,
        "num_components": a.num_components,
        "alpha_init": a.alpha_init,
        "scale": a.scale,
    }
    return build_metric("td", cache.latent_dim, architecture).to(device), 1


def window_lag(a: DictConfig, frames: int) -> int | None:
    """Frame spacing of a windowed value, which must be one action block."""
    if frames == 1:
        return None
    lag = a.frameskip if a.window_lag is None else int(a.window_lag)
    if lag != a.frameskip:
        raise ValueError(
            f"window lag {lag} != action block {a.frameskip}: consecutive imagined latents are one action "
            "block apart, so the deployed window would not match the trained one"
        )
    if a.expand_traj:
        raise ValueError("expand_traj is not supported with a windowed init_value")
    logger.info(f"rp1-AC consuming a {frames}-frame window value (lag {lag})")
    return lag


class Actor:
    """The planner network, its optimizer, and the replay of its own imagined histories."""

    def __init__(self, a: DictConfig, action_dim: int, grounding: Grounding | None, device: str) -> None:
        self.net = PlannerNet(
            horizon=a.horizon,
            action_dim=action_dim,
            hidden_dim=a.hidden_dim,
            action_limit=a.action_limit,
            head_scale=a.head_scale,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=a.actor_lr, weight_decay=a.actor_weight_decay)
        self.a = a
        self.action_dim = action_dim
        self.grounding = grounding
        self.device = device
        # histories and goals; with grounding also the agent position and last command
        self.replay: tuple[torch.Tensor, ...] | None = None
        self.grounding_log: dict[str, float] = {}

    def _replayed(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        """Swap a ``replay_prob`` share of the batch for the previous step's imagined endpoints.

        ``batch`` is histories, actions, goals and, with grounding, the anchors.
        """
        if self.a.replay_prob <= 0 or self.replay is None:
            return batch
        picked = torch.nonzero(torch.rand(self.a.batch, device=self.device) < self.a.replay_prob).squeeze(1)
        if not picked.numel():
            return batch
        take = torch.randint(0, self.replay[0].shape[0], (picked.numel(),), device=self.device)
        batch = [tensor.clone() for tensor in batch]
        batch[1][picked] = 0.0  # deployed replans query with zero action history
        for index, replayed in zip([0, *range(2, len(batch))], self.replay, strict=True):
            batch[index][picked] = replayed[take]
        return batch

    def step(
        self, tasks: PlanningTasks, wm: LatentWorldModel, teacher: ValueFunction, frames: int
    ) -> tuple[float, float, ExpandBatch]:
        a, grounding = self.a, self.grounding
        problems = tasks.sample(a.batch)
        batch = [problems.histories, problems.actions, problems.goals]
        if grounding is not None:
            batch.extend(grounding.anchors(problems.rows, problems.has_previous))
        histories, actions, goals, *anchors = self._replayed(batch)
        start = histories[:, -1]

        def score(trajectory: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
            if frames > 1:
                energy = windowed_trajectory_value(teacher, trajectory, goals, histories, frames, a.temporal_objective)
            else:
                energy = trajectory_value(teacher, trajectory, goals, start, a.temporal_objective)
            if grounding is not None:
                energy = energy + grounding.penalty(start, trajectory, plan, *anchors)
            return energy

        plan = torch.zeros(a.batch, a.horizon, self.action_dim, device=self.device)
        energies: list[torch.Tensor] = []
        trajectory: torch.Tensor | None = None
        if a.reuse_refinement_rollouts:
            reused_plan = plan.detach().requires_grad_(True)
            reused_score = score(rollout_traj(wm, histories, actions, reused_plan), reused_plan)
        for k in range(a.iterations):
            # the gradient is an input to the learned rule, not part of the training path
            if a.reuse_refinement_rollouts:
                (gradient,) = torch.autograd.grad(reused_score.sum(), reused_plan, retain_graph=k > 0)
                energy = reused_score.detach()
            else:
                with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch stub is untyped.
                    probe = plan.detach().requires_grad_(True)
                    probe_score = score(rollout_traj(wm, histories, actions, probe), probe)
                    (gradient,) = torch.autograd.grad(probe_score.sum(), probe)
                energy = probe_score.detach()
            plan = self.net(plan, gradient.detach(), energy)
            trajectory = rollout_traj(wm, histories, actions, plan)
            plan_score = score(trajectory, plan)
            energies.append(plan_score.mean())
            if a.reuse_refinement_rollouts and k + 1 < a.iterations:
                reused_plan, reused_score = plan, plan_score
        if trajectory is None:
            raise RuntimeError("the planner ran no refinement iteration")
        if a.replay_prob > 0:
            self.replay = (trajectory[:, -3:].detach(), goals.detach())
            if grounding is not None:
                with torch.no_grad():
                    # the replayed start is this rollout's end: the agent sits where the plan took it
                    end = grounding.penalty.agent_path(plan.detach(), *anchors)[:, -1]
                    self.replay += (end, grounding.penalty.commands(plan.detach())[:, -1])
        if grounding is not None:
            with torch.no_grad():
                terms = grounding.penalty.terms(start, trajectory.detach(), plan.detach(), *anchors)
            self.grounding_log = {
                "penalty": float(grounding.penalty.weight * terms["penalty"].mean()),
                "displacement": float(terms["displacement"][:, -1].mean()),
                "unsupported": float(terms["unsupported"][:, -1].mean()),
            }

        loss = energies[-1] + a.mean_weight * torch.stack(energies).mean()
        if a.ac_weight > 0:
            # anti-constancy: the batch-level constancy of the net plan displacement,
            # ||E_b[sum_t A]||^2 / E_b||sum_t A||^2 in [0, 1]. A planner that exploits
            # the world model emits a near-constant plan whatever the task.
            displacement = plan.sum(1)
            constancy = displacement.mean(0).pow(2).sum() / (displacement.pow(2).sum(1).mean() + 1e-8)
            loss = loss + a.ac_weight * constancy
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()

        if frames > 1:
            # the expansion tuple in the window value's input space: start window from
            # the real history, terminal window from the rollout, goal tiled to match
            start_window, goal_window = window_pair(histories, goals, frames)
            end_window, _ = window_pair(trajectory, goals, frames)
            expand = (start_window.detach(), end_window.detach(), goal_window.detach())
        else:
            expand = (start.detach(), trajectory.detach(), goals.detach())
        return float(energies[0].item()), float(energies[-1].item()), expand


def planner_payload(
    a: DictConfig,
    state_dict: dict[str, torch.Tensor],
    action_dim: int,
    value: Path,
    frames: int,
    lag: int | None,
    grounding: Grounding | None,
) -> dict[str, object]:
    """The deployable planner checkpoint around an actor state dict."""
    return {
        "state_dict": state_dict,
        "horizon": a.horizon,
        "iterations": a.iterations,
        "action_dim": action_dim,
        "action_limit": a.action_limit,
        "hidden_dim": a.hidden_dim,
        "head_scale": a.head_scale,
        "value": str(value),
        "temporal_objective": a.temporal_objective,
        "window_frames": frames,
        "window_lag": lag,
        "grounding": None if grounding is None else grounding.penalty.export(),
    }


def run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "training", cfg.core.agent.planner, cfg.core.agent.value)
    if a.temporal_objective not in {"terminal", "tel-exact", "tel-stopprev"}:
        raise ValueError(f"unsupported temporal objective: {a.temporal_objective}")
    if a.actor_only and not a.init_value:
        raise ValueError("actor_only=true requires init_value")
    if not a.actor_only and not a.cache_td:
        raise ValueError("cache_td is required unless actor_only=true")
    device = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    wm_module = load_pretrained(a.wm).to(device).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)

    cache = LatentCache.load(a.cache, mmap=bool(a.cache_mmap)).first_episodes(a.max_episodes)
    if cache.phase_multiplex > 1:
        logger.info(f"Planner cache is phase-multiplexed x{cache.phase_multiplex}")
    blocks = ActionBlocks(a.h5, a.frameskip, a.action_stats, cache.phase_multiplex)
    band_mix = None if a.band_mix is None else [int(band) for band in a.band_mix]
    tasks = PlanningTasks(
        cache, blocks, max_delta=a.max_delta, band_mix=band_mix, p_cross=a.p_cross, rng=rng, device=device
    )
    dense = None if a.actor_only else LatentCache.load(a.cache_td, mmap=bool(a.cache_mmap))
    grounding = None
    if a.grounding is not None:
        if dense is None:
            raise ValueError("grounding needs the dense cache (cache_td) to fit its state probe")
        grounding = Grounding(a, dense, blocks.actions, blocks.mean, blocks.std, device)
    if dense is not None:
        dense = dense.first_episodes(a.max_episodes)
    if a.near_frac > 0:
        logger.info(f"rp1-AC co-critic near-goal oversampling: frac={a.near_frac} max={a.near_max} steps")
    module, frames = build_critic(a, int(tasks.z.shape[-1] if dense is None else dense.latent_dim), dense, device)
    lag = window_lag(a, frames)
    critic = Critic(a, module, dense, frames, device)
    actor = Actor(a, blocks.dim, grounding, device)

    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint
    # the teacher is saved next to the planner, so checkpoints refer to it by name
    value_checkpoint = Path(str(a.output.value_checkpoint))

    pretrain = 0 if a.actor_only else (a.pretrain if a.pretrain >= 0 else (0 if a.init_value else 2000))
    for i in range(pretrain):
        loss = critic.step(None, a.expectile, a.critic_lr)
        if i % 500 == 0:
            logger.info(f"rp1-AC pretrain {i}/{pretrain}: td_loss={loss:.4f}")

    freeze_at = 0 if a.actor_only else int(a.freeze_critic_frac * a.steps)
    expand: ExpandBatch | None = None
    for step in range(a.steps):
        if step == freeze_at:
            logger.info(f"rp1-AC step {step}: critic and teacher frozen for {a.steps - freeze_at} steps")
        # the teacher converges (lr decay) and sharpens (expectile anneal) over the
        # live phase; the actor lr decays over all steps
        tau = (
            a.expectile
            if a.expectile_final is None
            else a.expectile + (a.expectile_final - a.expectile) * min(step, freeze_at) / max(freeze_at, 1)
        )
        critic_lr = cosine_interpolate(a.critic_lr, a.critic_lr_final, step, freeze_at)
        actor_lr = cosine_interpolate(a.actor_lr, a.actor_lr_final, step, a.steps)
        for group in actor.optimizer.param_groups:
            group["lr"] = actor_lr
        td_loss = float("nan")
        if step < freeze_at:
            for _ in range(a.critic_ratio):
                td_loss = critic.step(expand if a.expand_weight > 0 else None, tau, critic_lr)
                expand = None  # each actor batch is consumed once
        first, final, expand = actor.step(tasks, wm, critic.target, frames)
        if a.ckpt_every and (step + 1) % int(a.ckpt_every) == 0 and (step + 1) < a.steps:
            snapshot = planner_checkpoint.with_name(f"{planner_checkpoint.stem}_step{step + 1}.pt")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            state = {key: value.detach().cpu().clone() for key, value in actor.net.state_dict().items()}
            torch.save(planner_payload(a, state, blocks.dim, value_checkpoint, frames, lag, grounding), snapshot)
            logger.info(f"Saved planner snapshot at step {step + 1} to {snapshot}")
        if step % 500 == 0:
            grounded = "".join(f" ground_{key} {value:.3f}" for key, value in actor.grounding_log.items())
            logger.info(
                f"step {step}: E_final {final:.3f} E_first {first:.3f} "
                f"td_loss {td_loss:.4f} tau {tau:.3f} clr {critic_lr:.2e} alr {actor_lr:.2e}{grounded}"
            )

    # the teacher first: the planner checkpoint references it
    saved_value = save_metric(critic.teacher.cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory)
    logger.success(f"Saved teacher value to {saved_value}")
    actor.net.eval()
    torch.save(
        planner_payload(a, actor.net.cpu().state_dict(), blocks.dim, value_checkpoint, frames, lag, grounding),
        planner_checkpoint,
    )
    logger.success(f"Saved learned planner to {planner_checkpoint}")
