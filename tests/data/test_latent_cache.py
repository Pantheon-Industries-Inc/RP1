import torch

from rp1.data import LatentCache


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
