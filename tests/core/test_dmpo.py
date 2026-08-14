"""DMPO learned optimizer: structure, MPPI reduction, and gradient path.

The published method is a *residual* on MPPI with a near-zero-initialized last
layer, so an untrained network must reproduce the hand-written update; that
property is what makes the learned rule safe to drop into an MPC loop, and it
is the invariant these tests guard.
"""

import pytest
import torch

from rlp.core.planner.dmpo import DMPONet, gaussian_halton

H, A_DIM, N, B = 3, 4, 16, 5


def _net(**kwargs: object) -> DMPONet:
    defaults: dict[str, object] = {"horizon": H, "a_dim": A_DIM, "num_samples": N, "hidden": 32, "amax": 1.0}
    return DMPONet(**{**defaults, **kwargs})  # type: ignore[arg-type]


def test_halton_samples_are_deterministic_and_standardized() -> None:
    first = gaussian_halton(512, 6, seed=0)
    assert first.shape == (512, 6)
    assert torch.equal(first, gaussian_halton(512, 6, seed=0))
    assert not torch.equal(first, gaussian_halton(512, 6, seed=7))
    assert first.mean().abs() < 0.1
    assert 0.8 < float(first.std()) < 1.2


def test_first_sample_is_the_current_mean() -> None:
    net = _net()
    mean, std = net.initial(B, "cpu")
    mean = mean + 0.3
    plans = net.plans(mean, std)
    assert plans.shape == (B, N, H, A_DIM)
    assert torch.allclose(plans[:, 0], mean)


def test_untrained_update_reproduces_the_mppi_update() -> None:
    """Near-zero last layer => gate ~ 0 => the learned mean is the MPPI mean."""
    torch.manual_seed(0)
    net = _net(init_scale=1e-6)
    mean, std = net.initial(B, "cpu")
    plans = net.plans(mean, std)
    costs = plans.pow(2).sum(dim=(2, 3))
    updated, updated_std, mppi = net(mean, std, plans, costs)
    assert torch.allclose(updated, mppi, atol=1e-4)
    assert torch.allclose(updated_std, std, atol=1e-4)  # exp(~0) = 1, multiplicative


def test_gate_free_variant_is_additive_on_mppi() -> None:
    net = _net(gated=False, init_scale=1e-6)
    mean, std = net.initial(B, "cpu")
    plans = net.plans(mean, std)
    updated, _, mppi = net(mean, std, plans, plans.pow(2).sum(dim=(2, 3)))
    assert torch.allclose(updated, mppi, atol=1e-4)


def test_mppi_update_reduces_a_quadratic_cost() -> None:
    """The hand-written update alone must make progress on the sampled cost.

    It does not converge: the sample set is fixed by design, so with a fixed
    covariance the same directions are reproposed every iteration — shrinking
    it is exactly what DMPO's learned covariance update is for.
    """
    target = torch.full((B, H, A_DIM), 0.4)
    net = _net(num_samples=64)
    mean, std = net.initial(B, "cpu")

    def cost(plan: torch.Tensor) -> torch.Tensor:
        return (plan - target).pow(2).sum(dim=(1, 2))

    before = cost(mean)
    for _ in range(6):
        plans = net.plans(mean, std)
        mean = net.mppi_mean(mean, plans, (plans - target.unsqueeze(1)).pow(2).sum(dim=(2, 3)))
    assert bool((cost(mean) < before).all())


def test_plan_is_differentiable_in_the_networks() -> None:
    target = torch.full((B, H, A_DIM), 0.4)
    net = _net()

    def cost(plans: torch.Tensor) -> torch.Tensor:
        return (plans - target.unsqueeze(1)).pow(2).sum(dim=(2, 3))

    mean, std = net.initial(B, "cpu")
    final, _, history = net.plan(cost, mean, std, iters=2)
    assert len(history) == 2
    (final - target).pow(2).mean().backward()  # type: ignore[no-untyped-call]
    gradients = [p.grad for p in net.actor.parameters() if p.grad is not None]
    assert gradients and any(bool(g.abs().sum() > 0) for g in gradients)


def test_warm_start_shifts_forward_and_learns_a_residual() -> None:
    mean = torch.arange(B * H * A_DIM, dtype=torch.float32).view(B, H, A_DIM) / (B * H * A_DIM)
    std = torch.full_like(mean, 0.5)

    plain = _net(use_shift=False)
    shifted, shifted_std = plain.warm_start(mean, std, executed=1)
    assert torch.allclose(shifted[:, :-1], mean[:, 1:])
    assert torch.allclose(shifted[:, -1], torch.zeros_like(shifted[:, -1]))
    assert torch.allclose(shifted_std, std)

    learned = _net(use_shift=True, init_scale=1e-6)
    warm, warm_std = learned.warm_start(mean, std, executed=1)
    assert torch.allclose(warm, shifted, atol=1e-3)  # untrained residual is ~0
    assert torch.allclose(warm_std, shifted_std, atol=1e-3)

    # a full receding horizon leaves nothing to shift: the residual is the warm start
    empty, _ = plain.warm_start(mean, std, executed=H)
    assert torch.count_nonzero(empty) == 0


def test_solver_rejects_a_foreign_checkpoint(tmp_path: object) -> None:
    from rlp.core.solver.dmpo import DMPOSolver

    path = tmp_path / "planner.pt"  # type: ignore[operator]
    torch.save({"kind": "lip4", "sd": {}}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint kind"):
        DMPOSolver(model=torch.nn.Linear(2, 2), actor_path=str(path))
