#!/usr/bin/env python3
"""Smoke-test loading all six OGBench cube world-model checkpoints."""

from __future__ import annotations

from pathlib import Path

import stable_worldmodel as swm


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = [
    "models/ogbench_cube_single_dino/weights_step_100000.pt",
    "models/ogbench_cube_double_dino/weights_step_100000.pt",
    "models/ogbench_cube_triple_dino/weights_step_100000.pt",
    "models/ogbench_cube_single_lewm/weights_step_40000.pt",
    "models/ogbench_cube_double_lewm/weights_step_40000.pt",
    "models/ogbench_cube_triple_lewm/weights_step_40000.pt",
]


def main() -> None:
    for rel_path in CHECKPOINTS:
        checkpoint = ROOT / rel_path
        model = swm.wm.utils.load_pretrained(str(checkpoint))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK {rel_path}: {type(model).__name__} ({n_params:,} params)")


if __name__ == "__main__":
    main()
