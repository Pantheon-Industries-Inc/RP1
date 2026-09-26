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


def _captured_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *overrides: str) -> dict[str, dict[str, object]]:
    seen: dict[str, dict[str, object]] = {}

    def capture(parent: DictConfig, index: int, name: str, config_name: str, **values: object) -> object:
        seen[name] = values
        return None

    monkeypatch.setattr(pipeline, "_stage", capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rp1",
            "training.wm=unused",
            "training.dataset=unused",
            f"training.cache_directory={tmp_path}/caches",
            *overrides,
        ],
    )
    run_hydra(lambda cfg: pipeline.run(cfg), config_dir="training", config_name="posttrain")
    return seen


def test_architecture_overrides_configure_the_value_and_planner_groups(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _captured_stages(
        monkeypatch,
        tmp_path,
        "training.stages=[value,planner]",
        "training.value.eikonal_weight=0.5",
        "training.value.depth=1",
        "training.value.save_every=100",
        "training.planner.iterations=4",
    )
    value, planner = seen["value"], seen["planner"]
    assert value["core.agent.value.eikonal_weight"] == 0.5
    assert value["core.agent.value.depth"] == 1
    assert value["training.save_every"] == 100
    assert "training.eikonal_weight" not in value
    assert planner["core.agent.planner.iterations"] == 4
    assert planner["core.agent.planner.action_limit"] == 2.5


def test_unset_overrides_are_not_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _captured_stages(monkeypatch, tmp_path, "training.stages=[value]")
    assert "core.agent.value.eikonal_weight" not in seen["value"]
    assert seen["value"]["training.learner"] == "td"


def test_a_teacher_replaces_the_value_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    teacher = tmp_path / "value_td_step100"
    teacher.mkdir()
    seen = _captured_stages(
        monkeypatch, tmp_path, "training.stages=[planner]", f"training.teacher={teacher}", "training.actor_phases=5"
    )
    assert seen["planner"]["training.init_value"] == str(teacher)
    assert str(seen["planner"]["training.cache"]).endswith("rp1_fs5p5.pt")
    with pytest.raises(SystemExit):
        _captured_stages(monkeypatch, tmp_path, f"training.teacher={teacher}")
