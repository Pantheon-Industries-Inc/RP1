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

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlp", "wm=unused", "dataset=unused"])
    run_hydra(task, config_name="train/rlp")

    sub = LatentCache.load(str(out))
    assert len(sub.z) == 10  # every 2nd frame of two 10-step episodes
