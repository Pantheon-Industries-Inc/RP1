"""Collect a mixed TwoRoom dataset: expert + random exploration.

The random episodes make the agent collide with the central wall, so a learned
world model can model the wall. This is what surfaces the paper's phenomenon:
with an accurate WM, blocked plans yield terminals that stay on the wrong side
of the wall, where raw latent (Euclidean) distance mis-ranks candidates but a
reachability-trained TRM metric does not.

Example::

    python scripts/data/collect_tworooms_mixed.py --expert 300 --random 300 \
        --out tworoom_mixed.lance
"""

import argparse
from pathlib import Path

import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy
from stable_worldmodel.policy import RandomPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert", type=int, default=300)
    p.add_argument("--random", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--out", default="tworoom_mixed.lance")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    path = Path(swm.data.utils.get_cache_dir()) / "datasets" / args.out
    rng = np.random.default_rng(args.seed)

    world = swm.World("swm/TwoRoom-v1", num_envs=args.num_envs,
                      max_episode_steps=args.max_steps, image_shape=(224, 224),
                      render_mode="rgb_array")

    # expert episodes (goal-reaching coverage)
    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))
    world.collect(path, episodes=args.expert, seed=rng.integers(0, 1_000_000).item())

    # random episodes (wall-collision coverage); appended to the same lance
    world.set_policy(RandomPolicy())
    world.collect(path, episodes=args.random, seed=rng.integers(0, 1_000_000).item())

    logging.success(f"🎉 mixed TwoRoom dataset ({args.expert} expert + {args.random} random) -> {path}")


if __name__ == "__main__":
    main()
