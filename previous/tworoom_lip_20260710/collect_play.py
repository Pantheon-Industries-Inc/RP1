"""Collect a tworoom play dataset locally (mirrors scripts/data/collect_tworooms.py).

Same expert as the repo collector: ExpertPolicy(action_noise=2.0,
action_repeat_prob=0.05), default variations (agent/target position random).
"""

import argparse
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    world = swm.World(
        "swm/TwoRoom-v1",
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=100,
        render_mode="rgb_array",
    )
    world.set_policy(ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05))
    world.collect(Path(args.out), episodes=args.episodes, seed=args.seed)
    print(f"collected {args.episodes} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
