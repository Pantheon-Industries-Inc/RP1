"""Render the environment illustration figure used in the paper.

Renders one frame per evaluation environment straight from the registered
Stable-WM envs and lays them out as a labelled panel.

    python scripts/make_env_figure.py --out docs/figures/environments.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import stable_worldmodel  # noqa: F401  (registers the swm/* envs)

# (label, env id, reset seed, extra gym.make kwargs)
PANELS = (
    ("Reacher", "swm/ReacherDMControl-v0", 0, {}),
    ("OGBench Cube", "swm/OGBCube-v0", 2, {"ob_type": "pixels", "width": 224, "height": 224}),
    ("TwoRoom", "swm/TwoRoom-v1", 0, {}),
)


def render(env_name: str, seed: int, **kwargs) -> np.ndarray:
    env = gym.make(env_name, render_mode="rgb_array", **kwargs)
    env.reset(seed=seed)
    frame = np.asarray(env.render())
    env.close()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/figures/environments.png"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    frames = [(label, render(env_name, seed, **kwargs)) for label, env_name, seed, kwargs in PANELS]

    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3.4))
    for ax, (label, frame) in zip(axes, frames, strict=True):
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("0.6")
            spine.set_linewidth(0.8)
        ax.set_title(label, fontsize=13, family="serif", y=-0.16)

    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.99, bottom=0.09)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
