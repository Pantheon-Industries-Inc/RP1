"""The eikonal penalty constrains the critic's INPUT GRADIENT magnitude.

A temporal-distance critic should change by one unit per one step's worth of
latent displacement, i.e. ``||grad_z V|| == 1 / step_norm``. The refiner
descends exactly this gradient through the frozen world model, so its
magnitude is a property worth pinning.
"""

from __future__ import annotations

import torch

from rlp.core.value.learners.td import TDConfig, fit
from rlp.data import LatentCache


def _cache() -> LatentCache:
    torch.manual_seed(0)
    return LatentCache(
        z=torch.randn(600, 16),
        episode_idx=torch.arange(600, dtype=torch.int64) // 40,
        step_idx=torch.arange(600, dtype=torch.int64) % 40,
    )


def _config(weight: float) -> TDConfig:
    return TDConfig(
        head="quasimetric",
        hidden_dim=32,
        depth=2,
        embed_dim=16,
        n_step=1,
        gamma=0.98,
        expectile=0.03,
        p_cross=0.3,
        batch_size=64,
        steps=150,
        seed=0,
        eikonal_weight=weight,
        max_delta=6,
    )


def _mean_gradient_norm(module: torch.nn.Module, cache: LatentCache) -> float:
    z = cache.z[:64].clone().requires_grad_(True)
    goal = cache.z[64:128]
    (gradient,) = torch.autograd.grad(module(z, goal).sum(), z)
    return float(gradient.norm(dim=-1).mean())


def _step_norm(cache: LatentCache) -> float:
    total, count = 0.0, 0
    for rows in cache.episodes().values():
        episode = cache.z[torch.as_tensor(rows)].float()
        total += float((episode[1:] - episode[:-1]).norm(dim=-1).sum())
        count += len(rows) - 1
    return total / count


def test_penalty_pulls_gradient_norm_toward_the_temporal_distance_scale() -> None:
    cache = _cache()
    target = 1.0 / _step_norm(cache)
    off = _mean_gradient_norm(fit(cache, _config(0.0), "cpu"), cache)
    on = _mean_gradient_norm(fit(cache, _config(10.0), "cpu"), cache)
    assert abs(on - target) < abs(off - target), (
        f"eikonal penalty did not move the gradient norm toward {target:.4f}: off={off:.4f} on={on:.4f}"
    )


def test_zero_weight_leaves_training_untouched() -> None:
    cache = _cache()
    a = fit(cache, _config(0.0), "cpu")
    b = fit(cache, _config(0.0), "cpu")
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb), "TD fit is not deterministic at a fixed seed"
