"""Train one planner against several frozen (world model, value) pairs.

Each source brings its own world model, block cache, action h5 and frozen value;
only the learned update rule is shared. The planner's loss is the energy of an
imagined rollout, so trained on one world model it learns that model's errors;
a rule that must work on several energy landscapes cannot spend its capacity on
the holes of any one of them. Sources are either several world models of one
environment or several environments with the same horizon and action width.

The values stay frozen, so this is the actor half of :mod:`rp1_ac` with no
critic, value expansion or replay. Energies are in each environment's own time
units, so every source's loss is divided by a running mean of its initial
energy, and the shared rule optimizes relative improvement.

Sources are taken round-robin, and one deployable checkpoint is written per
source: the same weights, each next to the value it deploys with.

Example::

    pixi run posttrain --config-name phases/agent/rp1_multi \\
        'training.sources=[{name: tworoom, wm: ..., cache: ..., h5: ..., value: ..., max_delta: 10}, ...]'
"""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import DictConfig

from rp1.core.agent.planner import PlannerNet
from rp1.core.agent.value.temporal import ValueFunction, trajectory_value, windowed_trajectory_value
from rp1.core.world_model.base import LatentWorldModel
from rp1.core.world_model.rollout import rollout_traj
from rp1.data import LatentCache
from rp1.training.harness.checkpointing import load_metric, load_pretrained
from rp1.training.harness.schedule import cosine_interpolate
from rp1.training.phases.agent.rp1_ac import ActionBlocks, PlanningTasks, planner_payload
from rp1.utils.config import phase_config
from rp1.utils.device import pick_device
from rp1.utils.logging import logger


@dataclass
class Source:
    """One frozen (world model, value) pair and the planning problems drawn for it."""

    name: str
    wm: LatentWorldModel
    tasks: PlanningTasks
    value: ValueFunction
    value_path: str
    frames: int
    action_dim: int
    energy_scale: float = 0.0
    steps: int = 0

    def score(
        self, trajectory: torch.Tensor, histories: torch.Tensor, goals: torch.Tensor, objective: str
    ) -> torch.Tensor:
        if self.frames > 1:
            return windowed_trajectory_value(self.value, trajectory, goals, histories, self.frames, objective)
        return trajectory_value(self.value, trajectory, goals, histories[:, -1], objective)


def load_source(spec: DictConfig, a: DictConfig, rng: np.random.Generator, device: str) -> Source:
    wm = load_pretrained(str(spec.wm)).to(device).eval()
    wm.requires_grad_(False)
    cache = LatentCache.load(str(spec.cache), mmap=True)
    blocks = ActionBlocks(str(spec.h5), a.frameskip, None, cache.phase_multiplex)
    # episode length is a property of the environment, so a source may bring its own goal band
    max_delta = a.max_delta if spec.max_delta is None else int(spec.max_delta)
    tasks = PlanningTasks(cache, blocks, max_delta=max_delta, band_mix=None, p_cross=a.p_cross, rng=rng, device=device)
    value = load_metric(str(spec.value), device=device)
    value.requires_grad_(False)
    source = Source(
        name=str(spec.name),
        wm=cast(LatentWorldModel, wm),
        tasks=tasks,
        value=cast(ValueFunction, value),
        value_path=str(Path(str(spec.value)).expanduser().resolve()),
        frames=int(cast(int, value.latent_dim)) // cache.latent_dim,
        action_dim=blocks.dim,
    )
    logger.info(
        f"Source {source.name}: {len(tasks.episodes)} episodes, latent {cache.latent_dim}, "
        f"{source.frames} value frame(s), action width {source.action_dim}, max_delta {max_delta}"
    )
    return source


def run(cfg: DictConfig) -> None:
    a = phase_config(cfg, "training", cfg.core.agent.planner)
    device = pick_device(a.device)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)

    sources = [load_source(spec, a, rng, device) for spec in a.sources]
    if not sources:
        raise ValueError("rp1_multi needs at least one source")
    widths = {source.action_dim for source in sources}
    if len(widths) > 1:
        raise ValueError(f"sources disagree on action width {sorted(widths)}; one planner deploys one width")
    action_dim = widths.pop()
    net = PlannerNet(
        horizon=a.horizon,
        action_dim=action_dim,
        hidden_dim=a.hidden_dim,
        action_limit=a.action_limit,
        head_scale=a.head_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.actor_lr)

    def step(source: Source) -> float:
        problems = source.tasks.sample(a.batch)
        histories, actions, goals = problems.histories, problems.actions, problems.goals
        plan = torch.zeros(a.batch, a.horizon, action_dim, device=device)
        energies: list[torch.Tensor] = []
        for _ in range(a.iterations):
            with torch.enable_grad():  # type: ignore[no-untyped-call]  # PyTorch stub is untyped.
                probe = plan.detach().requires_grad_(True)
                energy = source.score(
                    rollout_traj(source.wm, histories, actions, probe), histories, goals, a.temporal_objective
                )
                (gradient,) = torch.autograd.grad(energy.sum(), probe)
            plan = net(plan, gradient.detach(), energy.detach())
            trajectory = rollout_traj(source.wm, histories, actions, plan)
            energies.append(source.score(trajectory, histories, goals, a.temporal_objective).mean())
        initial = float(energies[0].detach())
        source.energy_scale = initial if source.steps == 0 else 0.99 * source.energy_scale + 0.01 * initial
        source.steps += 1
        loss = (energies[-1] + a.mean_weight * torch.stack(energies).mean()) / max(source.energy_scale, 1e-6)
        if a.ac_weight > 0:
            displacement = plan.sum(1)
            constancy = displacement.mean(0).pow(2).sum() / (displacement.pow(2).sum(1).mean() + 1e-8)
            loss = loss + a.ac_weight * constancy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        return float(energies[-1].detach())

    log_every = max(1, a.steps // 20)
    for index in range(a.steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_interpolate(a.actor_lr, a.actor_lr_final, index, a.steps)
        source = sources[index % len(sources)]  # round robin: every source sees every learning-rate phase
        final = step(source)
        if index % log_every == 0:
            logger.info(
                f"step {index}/{a.steps} [{source.name}] E_final {final:.3f} (initial ~{source.energy_scale:.3f})"
            )

    state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    directory = Path(a.run.checkpoints)
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        lag = a.frameskip if source.frames > 1 else None
        path = directory / f"planner_{source.name}.pt"
        torch.save(planner_payload(a, state, action_dim, Path(source.value_path), source.frames, lag, None), path)
        logger.success(f"Saved the shared planner for {source.name} to {path}")


__all__ = ["Source", "load_source", "run"]
