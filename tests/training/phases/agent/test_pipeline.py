from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from rp1.data import LatentCache
from rp1.training.phases.agent import pipeline as pipeline
from rp1.utils.config import run_hydra


def test_pipeline_composes_with_no_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rp1",
            "training.wm=unused",
            "training.dataset=unused",
            f"training.cache_directory={tmp_path}/caches",
            "training.stages=[]",
        ],
    )
    run_hydra(lambda cfg: pipeline.run(cfg), config_dir="training", config_name="posttrain")


def test_pipeline_stage_executes_inside_parent_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = LatentCache(
        z=torch.randn(20, 4),
        episode_idx=torch.arange(20, dtype=torch.int64) // 10,
        step_idx=torch.arange(20, dtype=torch.int64) % 10,
    )
    inp = tmp_path / "fs1.pt"
    out = tmp_path / "fs2.pt"
    cache.save(inp)

    def task(cfg: DictConfig) -> None:
        pipeline._stage(
            cfg,
            1,
            "subsample",
            "training/data/job/subsample_cache",
            **{"preparation.inp": str(inp), "preparation.out": str(out), "preparation.frameskip": 2},
        )
        pipeline._stage(
            cfg,
            2,
            "value",
            "training/phases/agent/metric",
            **{
                "training.cache": str(inp),
                "training.learner": "td",
                "training.steps": 2,
                "training.batch_size": 8,
                "training.n_step": 2,
                "runtime.device": "cpu",
            },
            **{"output.checkpoint": "value_td"},
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rp1", "training.wm=unused", "training.dataset=unused"])
    run_hydra(task, config_dir="training", config_name="posttrain")

    sub = LatentCache.load(str(out), mmap=False)
    assert len(sub.z) == 10  # every 2nd frame of two 10-step episodes
    runs = sorted((tmp_path / "logs").rglob("checkpoints/value_td"))
    assert runs, "value stage saved no checkpoint"


def test_value_architecture_knobs_route_onto_the_value_group(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`value.depth` / `value.eikonal_weight` configure the value GROUP, not the trainer.

    train/metric merges `cfg.core.value` onto its flat config, so a bare
    `eikonal_weight=` override would be dropped on the floor and the gradient
    penalty would silently stay off.
    """
    seen: dict[str, dict[str, object]] = {}

    def capture(parent: DictConfig, index: int, name: str, config_name: str, **values: object) -> object:
        seen[name] = values
        return None

    monkeypatch.setattr(rlp_pipeline, "_stage", capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rlp",
            "wm=unused",
            "dataset=unused",
            f"cache_directory={tmp_path}/caches",
            "skip=[cache,subsample,actions,planner]",
            "value.eikonal_weight=0.5",
            "value.depth=1",
        ],
    )
    run_hydra(lambda cfg: rlp_pipeline._run(cfg), config_name="train/rlp")
    value = seen["value"]
    assert value["core.value.eikonal_weight"] == 0.5
    assert value["core.value.depth"] == 1
    assert "eikonal_weight" not in value
    assert "depth" not in value


def test_eikonal_weight_is_absent_when_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, dict[str, object]] = {}

    def capture(parent: DictConfig, index: int, name: str, config_name: str, **values: object) -> object:
        seen[name] = values
        return None

    monkeypatch.setattr(rlp_pipeline, "_stage", capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rlp",
            "wm=unused",
            "dataset=unused",
            f"cache_directory={tmp_path}/caches",
            "skip=[cache,subsample,actions,planner]",
        ],
    )
    run_hydra(lambda cfg: rlp_pipeline._run(cfg), config_name="train/rlp")
    assert "core.value.eikonal_weight" not in seen["value"]
