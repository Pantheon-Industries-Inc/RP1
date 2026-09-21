"""Tests for shared training callbacks, transforms, and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from torchvision.transforms import v2

from rlp.core.agent.value.temporal import trajectory_value
from rlp.data import LatentCache
from rlp.training.harness import callbacks
from rlp.training.harness.callbacks import NonFiniteGradientGuard, PortableCheckpointCallback
from rlp.training.harness.schedule import cosine_interpolate, sample_windows
from rlp.training.harness.transforms import nested_clip, nested_resize


@dataclass
class TrainerDouble:
    global_step: int
    current_epoch: int = 0
    max_epochs: int = 1
    is_global_zero: bool = True


class ModuleDouble(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Identity()


def test_cosine_interpolate_endpoints_and_disabled_final() -> None:
    assert cosine_interpolate(1.0, 0.0, 0, 10) == pytest.approx(1.0)
    assert cosine_interpolate(1.0, 0.0, 10, 10) == pytest.approx(0.0)
    assert cosine_interpolate(1.0, None, 5, 10) == 1.0


def test_sample_windows_aligns_values_and_actions() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    actions = values + 100
    episodes = {0: np.arange(5), 1: np.arange(5, 10)}
    sampled_values, sampled_actions = sample_windows(
        episodes, values, actions, window=3, batch_size=8, rng=np.random.default_rng(0)
    )
    assert sampled_values.shape == sampled_actions.shape == (8, 3, 2)
    assert torch.equal(sampled_actions, sampled_values + 100)
    assert torch.equal(sampled_values[:, 1:, 0], sampled_values[:, :-1, 0] + 2)


def test_latent_cache_windowing_is_causal_and_episode_local() -> None:
    cache = LatentCache(
        z=torch.arange(10, dtype=torch.float32).view(5, 2),
        episode_idx=torch.tensor([0, 0, 0, 1, 1]),
        step_idx=torch.tensor([0, 1, 2, 0, 1]),
    )
    windowed = cache.windowed(frames=3, lag=1)
    assert windowed.z.shape == (5, 6)
    assert torch.equal(windowed.z[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(windowed.z[2], torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(windowed.z[3], torch.tensor([6.0, 7.0, 6.0, 7.0, 6.0, 7.0]))


def test_tel_exact_preserves_terminal_gradient_and_adds_start_baseline() -> None:
    def value(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return ((state - goal) ** 2).sum(dim=-1)

    start = torch.tensor([[2.0]], requires_grad=True)
    trajectory = torch.tensor([[[1.5], [1.0], [0.5]]], requires_grad=True)
    goal = torch.zeros(1, 1)
    terminal = trajectory_value(value, trajectory, goal, start, "terminal")
    telescoping = trajectory_value(value, trajectory, goal, start, "tel-exact")
    terminal_gradient = torch.autograd.grad(terminal.sum(), trajectory, retain_graph=True)[0]
    telescoping_gradient = torch.autograd.grad(telescoping.sum(), trajectory)[0]
    assert torch.equal(terminal_gradient, telescoping_gradient)
    assert torch.equal(telescoping, terminal - value(start, goal))


def test_nested_dependency_transforms_match_torchvision() -> None:
    image = torch.rand(3, 12, 10)
    sample = {"nested": {"image": image.clone(), "values": torch.tensor([-20.0, 2.0, 30.0])}}
    resized = nested_resize(6, "nested.image", "nested.image")(sample)
    assert torch.allclose(resized["nested"]["image"], v2.Resize(6)(image))
    clipped = nested_clip(10, "nested.values", "nested.values")(sample)
    assert torch.equal(clipped["nested"]["values"], torch.tensor([-10.0, 2.0, 10.0]))


def test_nonfinite_gradient_guard_skips_and_aborts() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    trainer = cast(Trainer, TrainerDouble(global_step=7))
    guard = NonFiniteGradientGuard(max_skipped=1)
    parameter.grad = torch.tensor(float("nan"))
    guard.on_before_optimizer_step(trainer, cast(LightningModule, torch.nn.Identity()), optimizer)
    observed_grad: torch.Tensor | None = parameter.grad
    assert observed_grad is None
    assert guard.skipped == 1
    parameter.grad = torch.tensor(float("inf"))
    with pytest.raises(ValueError, match="more than 1"):
        guard.on_before_optimizer_step(trainer, cast(LightningModule, torch.nn.Identity()), optimizer)


def test_portable_checkpoint_callback_cadence_and_rank(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved: list[tuple[torch.nn.Module, str, object, str, str | Path]] = []

    def fake_save(
        model: torch.nn.Module,
        *,
        run_name: str,
        config: object,
        filename: str,
        cache_dir: str | Path,
        **kwargs: Any,
    ) -> None:
        del kwargs
        saved.append((model, run_name, config, filename, cache_dir))

    monkeypatch.setattr(callbacks, "save_pretrained", fake_save)
    callback = PortableCheckpointCallback(
        "wm",
        {"_target_": "torch.nn.Identity"},
        tmp_path,
        epoch_interval=1,
        step_interval=5,
    )
    module = ModuleDouble()
    trainer_double = TrainerDouble(global_step=5)
    trainer = cast(Trainer, trainer_double)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    callback.on_train_epoch_end(trainer, module)
    assert [item[3] for item in saved] == ["weights_step_5.pt", "weights_epoch_1.pt"]

    trainer_double.global_step = 10
    trainer_double.is_global_zero = False
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert len(saved) == 2
