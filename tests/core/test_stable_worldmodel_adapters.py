"""Contract tests for RLP's adapters around the PyPI Stable World Model."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stable_worldmodel
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from rlp.core.agent.policy import NoMovePolicy
from rlp.core.agent.value import LatentGoalCost
from rlp.core.agent.value.base import TensorInfo
from rlp.environment.world import _resize_images_like_env
from rlp.training.harness import checkpointing as checkpoint_module


def test_stable_worldmodel_comes_from_pinned_distribution() -> None:
    package_path = Path(stable_worldmodel.__file__).resolve()
    assert importlib.metadata.version("stable-worldmodel") == "0.1.1"
    assert "thirdparty" not in package_path.parts


def test_checkpoint_adapter_accepts_plain_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_save(model: nn.Module, *, run_name: str, config: object, **kwargs: Any) -> None:
        captured.update(model=model, run_name=run_name, config=config, kwargs=kwargs)

    monkeypatch.setattr(checkpoint_module, "_save_pretrained", fake_save)
    model = nn.Linear(2, 2)
    checkpoint_module.save_pretrained(
        model,
        run_name="test",
        config={"_target_": "torch.nn.Identity"},
        filename="weights.pt",
    )
    assert captured["model"] is model
    assert OmegaConf.is_config(captured["config"])
    assert captured["kwargs"] == {"filename": "weights.pt"}


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_checkpoint_loader_resolves_existing_relative_paths(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_load(name: str, cache_dir: str | None = None, extra_args: object = None) -> nn.Module:
        captured.update(name=name, cache_dir=cache_dir, extra_args=extra_args)
        return nn.Identity()

    checkpoint = tmp_path / "checkpoint"
    if kind == "directory":
        checkpoint.mkdir()
    else:
        checkpoint.write_bytes(b"checkpoint")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checkpoint_module, "_load_pretrained", fake_load)

    model = checkpoint_module.load_pretrained(Path("checkpoint"), cache_dir="cache", extra_args={"value": 1})

    assert isinstance(model, nn.Identity)
    assert captured == {
        "name": str(checkpoint.resolve()),
        "cache_dir": "cache",
        "extra_args": {"value": 1},
    }


def test_checkpoint_loader_preserves_remote_or_missing_names(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_load(name: str, cache_dir: str | None = None, extra_args: object = None) -> nn.Module:
        del cache_dir, extra_args
        captured.append(name)
        return nn.Identity()

    monkeypatch.setattr(checkpoint_module, "_load_pretrained", fake_load)
    checkpoint_module.load_pretrained("owner/model")
    checkpoint_module.load_pretrained("missing-checkpoint.pt")

    assert captured == ["owner/model", "missing-checkpoint.pt"]


def test_latent_goal_cost_broadcasts_candidates_and_caches_goal() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encode_calls = 0

        def encode(self, goal: TensorInfo) -> dict[str, torch.Tensor]:
            self.encode_calls += 1
            return {"emb": goal["pixels"].float()}

        def rollout(self, info: TensorInfo, actions: torch.Tensor) -> None:
            batch, candidates = actions.shape[:2]
            info["predicted_emb"] = torch.zeros(batch, candidates, 1, 3)

    model = Model()
    cost = LatentGoalCost(model)
    info = {"goal": torch.ones(2, 1, 3)}
    actions = torch.zeros(2, 4, 5, 2)
    first = cost.get_cost(info, actions)
    second = cost.get_cost(info, actions)
    assert first.shape == (2, 4)
    assert torch.equal(first, torch.full((2, 4), 3.0))
    assert torch.equal(second, first)
    assert model.encode_calls == 1


def test_no_move_policy_zeroes_the_action() -> None:
    class ActionSpace:
        def sample(self) -> np.ndarray:
            return np.array([1.0, -2.0], dtype=np.float32)

    class Env:
        def __init__(self) -> None:
            self.action_space = ActionSpace()

    policy = NoMovePolicy()
    policy.set_env(Env())
    assert np.array_equal(policy.get_action({}), np.zeros(2, dtype=np.float32))


def test_dataset_images_are_resized_to_environment_shape() -> None:
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    env_pixels = np.zeros((2, 1, 16, 12, 3), dtype=np.uint8)
    resized = _resize_images_like_env(images, env_pixels)
    assert resized.shape == (2, 16, 12, 3)
    assert resized.dtype == np.uint8


def test_hydra_configs_compose() -> None:
    config_root = Path("configs").resolve()
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = compose(config_name="inference/benchmark/lewm", overrides=["core/solver=adam"])
    assert cfg.environment.env_name == "swm/OGBCube-v0"
    assert cfg.core.solver._target_ == "rlp.core.agent.solver.GradientSolver"

    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = compose(config_name="training/lewm", overrides=["training/data=tworoom_lewm"])
    assert cfg.core.world_model.architecture._target_ == "stable_worldmodel.wm.lewm.LeWM"
