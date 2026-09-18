"""Mechanics tests for the composed RLP replication pipeline (train/rlp)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from rlp.config import run_hydra
from rlp.data import LatentCache
from rlp.train import rlp as rlp_pipeline


def test_pipeline_composes_and_skips_all_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rlp",
            "wm=unused",
            "dataset=unused",
            f"cache_directory={tmp_path}/caches",
            "skip=[cache,subsample,actions,value,planner]",
        ],
    )
    run_hydra(lambda cfg: rlp_pipeline._run(cfg), config_name="train/rlp")


def test_pipeline_stage_executes_inside_parent_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A real stage must compose and dispatch after run_hydra closed its Hydra context."""
    cache = LatentCache(
        z=torch.randn(20, 4),
        episode_idx=torch.arange(20, dtype=torch.int64) // 10,
        step_idx=torch.arange(20, dtype=torch.int64) % 10,
    )
    inp = tmp_path / "fs1.pt"
    out = tmp_path / "fs2.pt"
    cache.save(inp)

    def task(cfg: DictConfig) -> None:
        rlp_pipeline._stage(cfg, 1, "subsample", "tools/subsample_cache", inp=str(inp), out=str(out), frameskip=2)
        rlp_pipeline._stage(
            cfg,
            2,
            "value",
            "train/metric",
            cache=str(inp),
            learner="td",
            steps=2,
            batch_size=8,
            n_step=2,
            device="cpu",
            **{"output.checkpoint": "value_td"},
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "wm=unused", "dataset=unused"])
    run_hydra(task, config_name="train/rlp")

    sub = LatentCache.load(str(out))
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
