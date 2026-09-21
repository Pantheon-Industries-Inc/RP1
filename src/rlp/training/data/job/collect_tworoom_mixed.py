"""Collect a mixed TwoRoom dataset: expert + random exploration.

The random episodes make the agent collide with the central wall, so a learned
world model can model the wall. This is what surfaces the paper's phenomenon:
with an accurate WM, blocked plans yield terminals that stay on the wrong side
of the wall, where raw latent (Euclidean) distance mis-ranks candidates but a
reachability-trained TRM metric does not.

Example::

    pixi run prepare job=collect_tworoom_mixed \
        preparation.expert=300 preparation.random=300 preparation.out=tworoom_mixed.lance
"""

from pathlib import Path

import numpy as np
import stable_worldmodel as swm
from omegaconf import DictConfig
from stable_worldmodel.envs.two_room import ExpertPolicy
from stable_worldmodel.policy import RandomPolicy

from rlp.environment import World
from rlp.utils.config import dispatch, phase_config, run_hydra
from rlp.utils.logging import logger


def _run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")

    path = Path(swm.data.utils.get_cache_dir()) / "datasets" / args.out
    rng = np.random.default_rng(args.seed)

    world = World(
        "swm/TwoRoom-v1",
        num_envs=args.num_envs,
        max_episode_steps=args.max_steps,
        image_shape=(224, 224),
        render_mode="rgb_array",
    )

    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))
    world.collect(path, episodes=args.expert, seed=rng.integers(0, 1_000_000).item())

    world.set_policy(RandomPolicy())
    world.collect(path, episodes=args.random, seed=rng.integers(0, 1_000_000).item())

    logger.success(f"Collected mixed TwoRoom dataset expert={args.expert} random={args.random} path={path}")


def main() -> object:
    return run_hydra(dispatch, config_name="training/data/job/collect_tworoom_mixed")


if __name__ == "__main__":
    main()
