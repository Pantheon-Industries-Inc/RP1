from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

from rp1.core.agent.value import build_metric
from rp1.training.harness.checkpointing import load_planner, save_metric
from rp1.utils.config import compose_config, dispatch


def test_one_planner_is_written_for_every_source(agent_data: Any, cube_world_model: Path, tmp_path: Path) -> None:
    value = save_metric(build_metric("l2", 192, {}), run_name="l2", cache_dir=tmp_path)
    source = f"wm: {cube_world_model}, cache: {agent_data.blocks}, h5: {agent_data.actions}, value: {value}"
    cfg = compose_config(
        Path("training"),
        "phases/agent/rp1_multi",
        [
            f"training.sources=[{{name: a, {source}, max_delta: null}}, {{name: b, {source}, max_delta: 2}}]",
            "training.steps=2",
            "training.batch=4",
            "training.max_delta=3",
            "runtime.device=cpu",
        ],
    )
    with open_dict(cfg):
        cfg.run = OmegaConf.create({"directory": str(tmp_path), "checkpoints": str(tmp_path / "checkpoints")})
    dispatch(cfg)

    first, second = (load_planner(str(tmp_path / "checkpoints" / f"planner_{name}.pt"), None) for name in "ab")
    for key, tensor in first.payload["state_dict"].items():
        assert torch.equal(tensor, second.payload["state_dict"][key])
    assert first.payload["action_dim"] == 25
