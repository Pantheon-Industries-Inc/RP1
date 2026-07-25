"""Collect a PushT play dataset (mirrors the documented pusht_expert recipe).

Uses the env's built-in WeakPolicy (keeps the agent near the block so
interactions are frequent) — the same collection policy family the PushT
LeWM checkpoint was trained on.
"""

import argparse
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--dist-constraint", type=int, default=100)
    args = p.parse_args()

    world = swm.World(
        "swm/PushT-v1",
        num_envs=args.num_envs,
        image_shape=(224, 224),
        max_episode_steps=args.max_steps,
        render_mode="rgb_array",
    )
    world.set_policy(WeakPolicy(dist_constraint=args.dist_constraint, seed=args.seed))
    world.collect(Path(args.out), episodes=args.episodes, seed=args.seed)
    print(f"collected {args.episodes} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
