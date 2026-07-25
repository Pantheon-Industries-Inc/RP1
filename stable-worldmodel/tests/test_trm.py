"""Unit tests for the TRM package (head, samplers, cost, oracle, diagnostics, io)."""

import numpy as np
import torch

from stable_worldmodel.trm import (
    LatentCache,
    MetricCost,
    PairwiseMetricHead,
    build_metric,
    diagnostics,
    learners,
    load_metric,
    oracle,
    save_metric,
)
from stable_worldmodel.trm.learners.regression import RegressionConfig
from stable_worldmodel.trm.samplers import (
    BalancedHorizonPairSampler,
    GeometricFutureSampler,
    TransitionSampler,
)


def _toy_cache(D=6, E=20, L=25, seed=0):
    g = np.random.default_rng(seed)
    zs, ep, st = [], [], []
    for e in range(E):
        base = g.standard_normal(D)
        for t in range(L):
            zs.append(base + 0.1 * t + 0.01 * g.standard_normal(D))
            ep.append(e)
            st.append(t)
    return LatentCache(torch.tensor(np.array(zs)).float(), torch.tensor(ep), torch.tensor(st))


def test_head_shape_and_nonneg():
    head = PairwiseMetricHead(8)
    z = torch.randn(4, 8)
    out = head(z, z + 0.1)
    assert out.shape == (4,)
    assert (out >= 0).all()  # softplus output


def test_head_symmetric_flag():
    head = PairwiseMetricHead(8, symmetric=True)
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.allclose(head(a, b), head(b, a), atol=1e-6)


def test_balanced_sampler_full_horizon():
    cache = _toy_cache()
    s = BalancedHorizonPairSampler(cache, n_buckets=5)
    batch = s.sample(2000)
    deltas = batch["label"].numpy()
    assert deltas.min() >= 1
    assert deltas.max() >= 15  # reaches long separations (full horizon)


def test_max_delta_cap():
    cache = _toy_cache()
    s = BalancedHorizonPairSampler(cache, max_delta=5)
    assert s.sample(500)["label"].numpy().max() <= 5


def test_transition_sampler_done_flag():
    cache = _toy_cache()
    s = TransitionSampler(cache)
    b = s.sample(256)
    assert set(np.unique(b["done"].numpy())).issubset({0.0, 1.0})
    for k in ("z_t", "z_tp1", "z_g"):
        assert b[k].shape == (256, cache.latent_dim)


def test_geometric_future_sampler():
    cache = _toy_cache()
    b = GeometricFutureSampler(cache, gamma=0.95).sample(128)
    assert b["z_s"].shape == b["z_g"].shape == (128, cache.latent_dim)


def test_geodesic_same_vs_cross():
    agent = torch.tensor([[50.0, 100.0]])
    same = torch.tensor([[80.0, 100.0]])
    cross = torch.tensor([[180.0, 100.0]])
    doors = torch.tensor([[112.0, 112.0]])
    d_same = float(oracle.geodesic_distance(agent, same, doors))
    d_cross = float(oracle.geodesic_distance(agent, cross, doors))
    assert abs(d_same - 30.0) < 1e-3            # straight line, same room
    assert d_cross > 130.0                       # must detour through door


def test_rollout_dynamics_wall_blocks():
    # push straight right into the wall, far from the door -> stays on left side
    pos0 = torch.tensor([[50.0, 30.0]])
    acts = torch.ones(1, 10, 2) * torch.tensor([1.0, 0.0])  # move +x
    doors = torch.tensor([[112.0, 180.0]])  # door far away (y=180)
    term = oracle.rollout_dynamics(pos0, acts, speed=10.0, door_points=doors, door_half=10.0)
    assert float(term[0, 0]) < 112.0  # blocked by wall


def test_metric_cost_modes_shapes():
    cache = _toy_cache()
    head = learners.regression.fit(cache, RegressionConfig(steps=50, batch_size=128, scale=25.0), "cpu")

    class _IdentityWM(torch.nn.Module):
        """Minimal base WM: latent == state, populates predicted_emb/goal_emb."""

        def get_cost(self, info, acts):
            B, S = acts.shape[0], acts.shape[1]
            z = info["state"][:, :, -1, :]  # (B,S,D) use last history frame as terminal
            info["predicted_emb"] = z.unsqueeze(2)        # (B,S,1,D)
            info["goal_emb"] = info["goal_state"][:, 0]   # (B,T,D)
            g = info["goal_emb"][:, -1].unsqueeze(1)
            return ((z - g) ** 2).sum(-1)

        def criterion(self, info):
            p = info["predicted_emb"][..., -1, :]
            g = info["goal_emb"][..., -1, :].unsqueeze(1)
            return ((p - g) ** 2).sum(-1)

    wm = _IdentityWM()
    B, S, D = 3, 7, cache.latent_dim
    info = {"state": torch.randn(B, S, 1, D), "goal_state": torch.randn(B, S, 1, D)}
    acts = torch.randn(B, S, 4, 2)
    for mode in ("latent", "replacement", "hybrid"):
        metric = None if mode == "latent" else head
        cost = MetricCost(wm, metric, mode=mode).get_cost(dict(info), acts)
        assert cost.shape == (B, S)
        assert torch.isfinite(cost).all()


def test_scsa_perfect_and_anti():
    perfect = diagnostics.scsa_pointwise([0.1, 0.2, 0.3, 0.9], [0.1, 0.2, 0.3, 0.9])
    assert perfect["spearman"] > 0.99
    assert perfect["regret"] == 0.0
    anti = diagnostics.scsa_pointwise([0.9, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.9])
    assert anti["spearman"] < -0.5


def test_metric_io_roundtrip(tmp_path):
    head = PairwiseMetricHead(8, hidden_dim=64)
    path = tmp_path / "m.pt"
    save_metric(head, "regression", 8, {"hidden_dim": 64, "depth": 2}, path)
    loaded = load_metric(path)
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    assert torch.allclose(head.cost(a, b), loaded.cost(a, b), atol=1e-5)
