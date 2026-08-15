"""Train the Imagination-Based Planner against a frozen goal-conditioned critic.

IBP (Pascanu et al., 2017, arXiv:1707.06170) learns *how to plan*: a manager
chooses at every step whether to act or to imagine — and from which node of an
imagination tree — while a controller proposes the actions and an LSTM memory
aggregates the imagined steps into the plan context conditioning both.  The
agent is trained to plan *economically*: its objective is the sum of a task
loss and a resource loss charging for imagination.

This trainer fits those pieces in the setting the repository's own learned
planner is trained in — frozen pretrained world model, frozen quasimetric
critic supplied through ``init_value``, latent windows from the offline cache:

    task loss    = V( z_T( plan ), z_g )            (controller + memory)
    manager gain = -( V( z_T( plan ), z_g ) + tau * imagination steps )

The controller and memory are trained by pathwise gradients through the frozen
differentiable world model.  The manager's routes are discrete and
non-differentiable, so it is trained by REINFORCE against the negative total
loss with an entropy bonus, exactly as the paper does; the routes are constants
to the controller's gradient, as the paper also specifies.

**Deliberate deviations from the paper**, for the same reason DMPO's port has
them (see docs/ibp/README_ibp.md):

1. The imagination is this repository's *frozen pretrained latent world model*,
   not an interaction network learned on-line.  IBP's model loss therefore has
   no term here — nothing about the model is trained.
2. The controller is trained by pathwise gradients rather than the paper's
   policy gradient, because the world model is differentiable.  This is the
   convention RLP and the DMPO port use, and the reason the rows are
   comparable.
3. Costs are the goal-conditioned critic, not a task reward; ``tau`` is
   expressed in the same units, so the trade-off the paper studies is intact.

Example (TwoRoom on the tracked LeJEPA base, after ``model=rlp skip=[planner]``
has produced the caches and ``value_td``)::

    pixi run train model=ibp \
        wm=assets/core/world_model/lejepa_tworoom \
        cache=$RLP_DATA_HOME/caches/tworoom_fs5.pt \
        h5=$RLP_DATA_HOME/caches/tworoom_actions.h5 \
        init_value=logs/<date>/<time>/checkpoints/value_td
"""

from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig, OmegaConf

from rlp.config import dispatch, run_hydra
from rlp.core.planner.ibp import IBPNet
from rlp.core.rollout import rollout_traj
from rlp.core.temporal import ValueFunction, trajectory_value
from rlp.core.value import load_metric
from rlp.core.value.io import save_metric
from rlp.core.world_model import load_pretrained
from rlp.core.world_model.protocols import LatentWorldModel
from rlp.logging import logger

from .utils import cosine_interpolate
from .windows import WindowSampler


def _device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _run(cfg: DictConfig) -> None:
    a = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), cfg.core.planner)
    if not isinstance(a, DictConfig):
        raise TypeError("merged planner configuration must be a mapping")
    a["amax"] = a["action_limit"]
    if not a.init_value:
        raise ValueError("model=ibp trains against a frozen critic: pass init_value=<value_td>")
    dev = _device(str(a.device))
    torch.manual_seed(a.seed)

    wm_module = load_pretrained(a.wm).to(dev).eval()
    wm_module.requires_grad_(False)
    wm = cast(LatentWorldModel, wm_module)

    sampler = WindowSampler(
        cache=a.cache,
        h5=a.h5,
        horizon=a.horizon,
        max_delta=a.max_delta,
        p_cross=a.p_cross,
        frameskip=a.frameskip,
        device=dev,
        mmap=bool(a.cache_mmap),
        seed=a.seed,
    )

    critic = load_metric(a.init_value, device=dev)
    critic_dim = int(cast(int, critic.latent_dim))
    if critic_dim != sampler.latent_dim:
        # A tree node is one imagined frame; a window critic has no window to
        # read at the interior nodes (the solver rejects these too).
        raise ValueError(f"IBP needs a single-frame critic: critic dim {critic_dim} != latent dim {sampler.latent_dim}")
    critic.eval()
    critic.requires_grad_(False)
    value_module = cast(ValueFunction, critic)

    net = IBPNet(
        z_dim=sampler.latent_dim,
        a_dim=sampler.a_dim,
        horizon=a.horizon,
        hidden=a.hidden,
        memory=a.memory,
        amax=a.amax,
        max_imagine=a.max_imagine,
        init_scale=a.init_scale,
    ).to(dev)

    # The paper's per-piece learning rates: controller 3e-4, manager 1e-4.
    manager_parameters = list(net.manager.parameters())
    manager_ids = {id(p) for p in manager_parameters}
    other_parameters = [p for p in net.parameters() if id(p) not in manager_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": a.actor_lr},
            {"params": manager_parameters, "lr": a.manager_lr},
        ],
        weight_decay=a.weight_decay,
    )

    def score(z_hist: torch.Tensor, a_hist: torch.Tensor, z_goal: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """Differentiable cost of one plan per problem."""
        trajectory = rollout_traj(wm, z_hist, a_hist, plan)
        return trajectory_value(value_module, trajectory, z_goal, z_hist[:, -1], "terminal")

    baseline = torch.zeros((), device=dev)

    def step(index: int) -> tuple[float, float, float]:
        nonlocal baseline
        batch = sampler.sample(a.batch)
        z_hist, a_hist, z_goal = batch.z_hist, batch.a_hist, batch.z_goal

        def imagine(z_window: torch.Tensor, a_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return rollout_traj(wm, z_window, a_window, action[:, None])[:, -1]

        def node_value(z: torch.Tensor) -> torch.Tensor:
            return value_module(z, z_goal)

        result = net.search(z_hist, a_hist, z_goal, imagine, node_value, sample=True)
        # The executed plan is re-scored through a fresh differentiable unroll:
        # this is the task loss the controller and memory are fit against.
        task = score(z_hist, a_hist, z_goal, result.plan)
        resource = a.tau * result.imagined
        # REINFORCE: the manager's reward is the negative total loss, as in the
        # paper; the controller treats the routes as constants.
        gain = -(task.detach() + resource)
        advantage = gain - baseline
        baseline = a.baseline_decay * baseline + (1 - a.baseline_decay) * gain.mean().detach()
        manager_loss = -(result.log_prob * advantage.detach()).mean() - a.entropy_weight * result.entropy.mean()
        loss = task.mean() + manager_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
        optimizer.step()
        return float(task.mean().item()), float(result.imagined.mean().item()), float(result.unrolls.mean().item())

    for i in range(a.steps):
        lr = cosine_interpolate(a.actor_lr, a.actor_lr_final, i, a.steps)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * (a.manager_lr / a.actor_lr)
        task_cost, imagined, unrolls = step(i)
        if i % 100 == 0:
            logger.info(f"step {i}: E_task {task_cost:.3f} imagined {imagined:.2f} unrolls {unrolls:.2f} lr {lr:.2e}")

    value_checkpoint = save_metric(critic.cpu(), run_name=a.output.value_checkpoint, cache_dir=a.run.directory)
    logger.success(f"Copied the frozen critic to {value_checkpoint}")
    planner_checkpoint = Path(a.run.checkpoints) / a.output.planner_checkpoint
    net.eval()
    torch.save(
        {
            "kind": "ibp",
            "sd": net.cpu().state_dict(),
            "z_dim": sampler.latent_dim,
            "horizon": int(a.horizon),
            "a_dim": sampler.a_dim,
            "hidden": int(a.hidden),
            "memory": int(a.memory),
            "amax": float(a.amax),
            "max_imagine": int(a.max_imagine),
            "tau": float(a.tau),
            "value": str(value_checkpoint),
            "value_context": 1,
        },
        planner_checkpoint,
    )
    logger.success(f"Saved the IBP planner to {planner_checkpoint}")


def main() -> object:
    return run_hydra(dispatch, config_name="train/ibp")


if __name__ == "__main__":
    main()
