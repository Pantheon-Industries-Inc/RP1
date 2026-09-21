from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

from rp1.training.harness.checkpointing import load_planner
from rp1.utils.config import compose_config, dispatch


def test_training_writes_a_deployable_planner(agent_data: Any, cube_world_model: Path, tmp_path: Path) -> None:
    data = agent_data
    cfg = compose_config(
        Path("training"),
        "phases/agent/rp1_ac",
        [
            f"training.cache={data.blocks}",
            f"training.cache_td={data.dense}",
            f"training.h5={data.actions}",
            f"training.wm={cube_world_model}",
            "training.steps=2",
            "training.batch=4",
            "training.td_batch=8",
            "training.pretrain=1",
            "training.max_delta=3",
            "training.expand_weight=1.0",
            "training.replay_prob=0.5",
            "runtime.device=cpu",
        ],
    )
    with open_dict(cfg):
        cfg.run = OmegaConf.create({"directory": str(tmp_path), "checkpoints": str(tmp_path / "checkpoints")})
    (tmp_path / "checkpoints").mkdir()
    dispatch(cfg)

    checkpoint = load_planner(str(tmp_path / "checkpoints" / "planner.pt"), None)
    assert checkpoint.payload["iterations"] == cfg.core.agent.planner.iterations
    state = torch.randn(2, 192)
    assert checkpoint.value(state, state).shape == (2,)
