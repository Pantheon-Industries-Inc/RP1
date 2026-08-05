"""Unit tests for the TRM package (head, samplers, cost, oracle, diagnostics, io)."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from rlp.core.value import (
    IQEHead,
    MetricCost,
    PairwiseMetricHead,
    QuasimetricHead,
    diagnostics,
    learners,
    load_metric,
    oracle,
    save_metric,
)
from rlp.core.value.io import PretrainedMetric
from rlp.core.value.learners.contrastive import ContrastiveCritic
from rlp.core.value.learners.regression import RegressionConfig
from rlp.core.value.protocols import TensorInfo
from rlp.core.value.samplers import (
    BalancedHorizonPairSampler,
    GeometricFutureSampler,
    TransitionSampler,
)
from rlp.data import LatentCache

MetricHead = PairwiseMetricHead | QuasimetricHead | IQEHead | ContrastiveCritic


class _PlanningStub(torch.nn.Module):
    """Minimal base implementing the planning-cost contract used by MetricCost."""

    def get_cost(self, info: TensorInfo, actions: torch.Tensor) -> torch.Tensor:
        del info
        return torch.zeros(actions.shape[:2])

    def criterion(self, info: TensorInfo) -> torch.Tensor:
        return torch.zeros(next(iter(info.values())).shape[:2])


def _toy_cache(D: int = 6, E: int = 20, L: int = 25, seed: int = 0) -> LatentCache:
    g = np.random.default_rng(seed)
    zs: list[np.ndarray] = []
    ep: list[int] = []
    st: list[int] = []
    for e in range(E):
        base = g.standard_normal(D)
        for t in range(L):
            zs.append(base + 0.1 * t + 0.01 * g.standard_normal(D))
            ep.append(e)
            st.append(t)
    return LatentCache(torch.tensor(np.array(zs)).float(), torch.tensor(ep), torch.tensor(st))


def test_head_shape_and_nonneg() -> None:
    head = PairwiseMetricHead(8)
    z = torch.randn(4, 8)
    out = head(z, z + 0.1)
    assert out.shape == (4,)
    assert (out >= 0).all()  # softplus output


def test_head_symmetric_flag() -> None:
    head = PairwiseMetricHead(8, symmetric=True)
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(head(a, b), head(b, a), atol=1e-6)


def test_balanced_sampler_full_horizon() -> None:
    cache = _toy_cache()
    s = BalancedHorizonPairSampler(cache, n_buckets=5)
    batch = s.sample(2000)
    deltas = batch["label"].numpy()
    assert deltas.min() >= 1
    assert deltas.max() >= 15  # reaches long separations (full horizon)


def test_max_delta_cap() -> None:
    cache = _toy_cache()
    s = BalancedHorizonPairSampler(cache, max_delta=5)
    assert s.sample(500)["label"].numpy().max() <= 5


def test_transition_sampler_done_flag() -> None:
    cache = _toy_cache()
    s = TransitionSampler(cache)
    b = s.sample(256)
    assert set(np.unique(b["done"].numpy())).issubset({0.0, 1.0})
    for k in ("z_t", "z_tp1", "z_g"):
        assert b[k].shape == (256, cache.latent_dim)


def test_geometric_future_sampler() -> None:
    cache = _toy_cache()
    b = GeometricFutureSampler(cache, gamma=0.95).sample(128)
    assert b["z_s"].shape == b["z_g"].shape == (128, cache.latent_dim)


def test_geodesic_same_vs_cross() -> None:
    agent = torch.tensor([[50.0, 100.0]])
    same = torch.tensor([[80.0, 100.0]])
    cross = torch.tensor([[180.0, 100.0]])
    doors = torch.tensor([[112.0, 112.0]])
    d_same = float(oracle.geodesic_distance(agent, same, doors))
    d_cross = float(oracle.geodesic_distance(agent, cross, doors))
    assert abs(d_same - 30.0) < 1e-3  # straight line, same room
    assert d_cross > 130.0  # must detour through door


def test_rollout_dynamics_wall_blocks() -> None:
    # push straight right into the wall, far from the door -> stays on left side
    pos0 = torch.tensor([[50.0, 30.0]])
    acts = torch.ones(1, 10, 2) * torch.tensor([1.0, 0.0])  # move +x
    doors = torch.tensor([[112.0, 180.0]])  # door far away (y=180)
    term = oracle.rollout_dynamics(pos0, acts, speed=10.0, door_points=doors, door_half=10.0)
    assert float(term[0, 0]) < 112.0  # blocked by wall


def test_metric_cost_modes_shapes() -> None:
    cache = _toy_cache()
    head = learners.regression.fit(cache, RegressionConfig(steps=50, batch_size=128, scale=25.0), "cpu")

    class _IdentityWM(torch.nn.Module):
        """Minimal base WM: latent == state, populates predicted_emb/goal_emb."""

        def get_cost(self, info: TensorInfo, acts: torch.Tensor) -> torch.Tensor:
            del acts
            z = info["state"][:, :, -1, :]  # (B,S,D) use last history frame as terminal
            info["predicted_emb"] = z.unsqueeze(2)  # (B,S,1,D)
            info["goal_emb"] = info["goal_state"][:, 0]  # (B,T,D)
            g = info["goal_emb"][:, -1].unsqueeze(1)
            return ((z - g) ** 2).sum(-1)

        def criterion(self, info: TensorInfo) -> torch.Tensor:
            p = info["predicted_emb"][..., -1, :]
            g = info["goal_emb"][..., -1, :].unsqueeze(1)
            return ((p - g) ** 2).sum(-1)

    wm = _IdentityWM()
    B, S, D = 3, 7, cache.latent_dim
    info = {
        "state": torch.randn(B, S, 1, D),
        "goal_state": torch.randn(B, S, 1, D),
    }
    acts = torch.randn(B, S, 4, 2)
    for mode in ("latent", "replacement", "hybrid"):
        metric = None if mode == "latent" else head
        cost = MetricCost(wm, metric, mode=mode).get_cost(dict(info), acts)
        assert cost.shape == (B, S)
        assert torch.isfinite(cost).all()


@torch.no_grad()
def test_metric_cost_reconstructs_context_and_uses_pessimistic_ensemble() -> None:
    class RecordingMetric(torch.nn.Module):
        def __init__(self, latent_dim: int, value: float) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.value = value
            self.last_shapes: tuple[torch.Size, torch.Size] | None = None

        def cost(self, pred: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
            self.last_shapes = (pred.shape, goal.shape)
            return torch.full(pred.shape[:-1], self.value)

    predicted = torch.randn(2, 5, 3, 4)
    goal = torch.randn(2, 1, 4)
    info = {"predicted_emb": predicted, "goal_emb": goal}

    for context in (1, 2, 3):
        metric = RecordingMetric(4 * context, 1.0)
        cost = MetricCost(_PlanningStub(), metric)
        result = cost._metric_terminal_cost(info)
        assert result.shape == (2, 5)
        assert metric.last_shapes == ((2, 5, 4 * context), (2, 5, 4 * context))

    low = RecordingMetric(4, 1.0)
    high = RecordingMetric(4, 3.0)
    ensemble = MetricCost(_PlanningStub(), low, metrics=[low, high])
    assert torch.equal(ensemble._metric_terminal_cost(info), torch.full((2, 5), 3.0))


def test_metric_cost_rejects_unavailable_context() -> None:
    metric = PairwiseMetricHead(12)
    cost = MetricCost(_PlanningStub(), metric)
    info = {"predicted_emb": torch.randn(2, 4, 2, 4), "goal_emb": torch.randn(2, 1, 4)}
    with pytest.raises(ValueError, match="requires 3 predicted frames"):
        cost._metric_terminal_cost(info)


def test_scsa_perfect_and_anti() -> None:
    perfect = diagnostics.scsa_pointwise([0.1, 0.2, 0.3, 0.9], [0.1, 0.2, 0.3, 0.9])
    assert perfect["spearman"] > 0.99
    assert perfect["regret"] == 0.0
    anti = diagnostics.scsa_pointwise([0.9, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.9])
    assert anti["spearman"] < -0.5


def test_scsa_spearman_handles_ties_and_degenerate_inputs() -> None:
    tied = diagnostics.scsa_pointwise([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
    assert tied["spearman"] == pytest.approx(0.8660254, rel=1e-5)
    assert np.isnan(diagnostics._spearman(np.array([1.0]), np.array([1.0])))
    assert np.isnan(diagnostics._spearman(np.ones(3), np.arange(3)))


@pytest.mark.parametrize(
    "head",
    [
        PairwiseMetricHead(8, hidden_dim=64, symmetric=True, scale=25.0),
        QuasimetricHead(8, hidden_dim=64, embed_dim=16, sym_frac=0.25),
        IQEHead(8, hidden_dim=64, embed_dim=16, num_components=4),
        ContrastiveCritic(8, hidden_dim=64, rep_dim=16),
    ],
)
def test_metric_io_roundtrip(tmp_path: Path, head: MetricHead) -> None:
    path = save_metric(head, run_name=type(head).__name__, cache_dir=tmp_path)
    assert path == tmp_path / "checkpoints" / type(head).__name__
    assert (path / "weights.pt").is_file()
    assert (path / "config.json").is_file()
    loaded = load_metric(path)
    assert type(loaded) is type(head)
    loaded_metric = cast(PretrainedMetric, loaded)
    assert loaded_metric.latent_dim == head.latent_dim
    assert not loaded.training
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    assert torch.allclose(head.cost(a, b), loaded_metric.cost(a, b), atol=1e-5)
    from_weights = load_metric(path / "weights.pt")
    from_weights_metric = cast(PretrainedMetric, from_weights)
    assert torch.allclose(head.cost(a, b), from_weights_metric.cost(a, b), atol=1e-5)


def test_metric_loader_rejects_legacy_blob(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": {}}, path)
    with pytest.raises(ValueError, match="Unsupported metric checkpoint"):
        load_metric(path)
