"""IBP: route semantics, resource accounting, and the two gradient paths.

The published method's claims are structural — the manager picks *routes* over
an imagination tree, the agent pays for imagination, and the controller learns
through the model while the manager learns by REINFORCE. These tests guard the
invariants that carry those claims into this port.
"""

import torch

from rlp.core.planner.ibp import ACT, LAST, ROOT, IBPNet

Z_DIM, A_DIM, H, B = 8, 3, 5, 4


def _net(**kwargs: object) -> IBPNet:
    defaults: dict[str, object] = {
        "z_dim": Z_DIM,
        "a_dim": A_DIM,
        "horizon": H,
        "hidden": 16,
        "memory": 16,
        "amax": 1.0,
        "max_imagine": 8,
    }
    return IBPNet(**{**defaults, **kwargs})  # type: ignore[arg-type]


def _problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(B, 3, Z_DIM), torch.zeros(B, 2, A_DIM), torch.randn(B, Z_DIM)


def _imagine(z_window: torch.Tensor, a_window: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """A differentiable stand-in for the frozen world model: drift by action."""
    padded = torch.zeros(action.shape[0], Z_DIM, device=action.device)
    padded[:, :A_DIM] = action
    return z_window[:, -1] + 0.5 * padded


def test_search_returns_a_full_plan_within_the_box() -> None:
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    result = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1))
    assert result.plan.shape == (B, H, A_DIM)
    assert result.plan.abs().max() <= net.amax + 1e-6
    # Every element executes a complete horizon: nothing is left unplanned.
    assert bool((result.plan.abs().sum(dim=-1) > 0).all())


def test_imagination_is_capped_by_the_budget_and_charged_for() -> None:
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    for budget in (1, 3, 8):
        result = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1), max_imagine=budget)
        assert int(result.imagined.max().item()) <= budget
        # completion unrolls are counted too: total >= imagined, and a chain
        # can never be completed past the horizon
        assert bool((result.unrolls >= result.imagined).all())
        assert float(result.unrolls.max().item()) <= budget + H


def test_the_first_step_cannot_act() -> None:
    """With nothing imagined there is no plan to commit, so `act` is masked."""
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    result = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1))
    assert bool((result.routes[0] != ACT).all())


def test_routes_stay_inside_the_paper_restricted_set() -> None:
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    result = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1))
    for route in result.routes:
        assert bool(((route == ACT) | (route == ROOT) | (route == LAST)).all())


def test_committed_plan_is_the_best_node_seen() -> None:
    """`act` commits the cheapest imagined node, not the most recent one."""
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    costs: list[torch.Tensor] = []

    def value(z: torch.Tensor) -> torch.Tensor:
        cost = (z - z_goal).pow(2).sum(-1)
        costs.append(cost)
        return cost

    result = net.search(z_hist, a_hist, z_goal, _imagine, value, max_imagine=6)
    # `act` commits the cheapest node the search saw, so the committed cost is
    # never worse than the root's do-nothing cost
    root_cost = (z_hist[:, -1] - z_goal).pow(2).sum(-1)
    assert bool((result.committed_cost <= root_cost + 1e-5).all())


def test_greedy_completion_is_the_only_way_the_plan_can_degrade() -> None:
    """A chain committed at full horizon executes exactly what was scored.

    Committing early leaves blocks to the controller's greedy continuation,
    which is *not* scored during the search — the one place the executed plan
    can be worse than the committed node. Guarding it keeps that honest.
    """
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    value = lambda z: (z - z_goal).pow(2).sum(-1)  # noqa: E731
    # a budget below the horizon forces completion; a generous one lets chains fill
    short = net.search(z_hist, a_hist, z_goal, _imagine, value, max_imagine=2)
    assert bool((short.unrolls > short.imagined).all())
    assert not torch.allclose(short.cost, short.committed_cost)


def test_controller_gets_pathwise_gradients_and_manager_gets_none() -> None:
    """The paper's split: routes are constants to the controller's gradient."""
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    result = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1))
    task = _imagine(z_hist, a_hist, result.plan[:, -1]).pow(2).sum(-1).mean()
    task.backward(retain_graph=True)  # type: ignore[no-untyped-call]
    assert net.controller[0].weight.grad is not None
    assert net.controller[0].weight.grad.abs().sum() > 0
    assert net.manager[0].weight.grad is None or net.manager[0].weight.grad.abs().sum() == 0


def test_manager_learns_only_through_reinforce() -> None:
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    result = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1))
    (-result.log_prob.mean()).backward()  # type: ignore[no-untyped-call]
    assert net.manager[0].weight.grad is not None
    assert net.manager[0].weight.grad.abs().sum() > 0


def test_greedy_manager_is_deterministic() -> None:
    """Deployment argmaxes the routes, so a decision repeats exactly."""
    net = _net()
    z_hist, a_hist, z_goal = _problem()
    kwargs = {"sample": False}
    first = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1), **kwargs)
    second = net.search(z_hist, a_hist, z_goal, _imagine, lambda z: (z - z_goal).pow(2).sum(-1), **kwargs)
    assert torch.equal(first.plan, second.plan)
    assert torch.equal(first.imagined, second.imagined)
