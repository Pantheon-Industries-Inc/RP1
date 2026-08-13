"""GraphedRefinement: equivalence against the eager refinement computation.

The heavy equivalence/timing evidence lives in the 2026-08-12 benchmark jobs
(4840: 272.6 -> 29.3 ms/decision at B=1, final-plan deviation 5.96e-8). This
test guards the shipped module: the graphed step must reproduce the eager
score, trajectory features, and value gradient on a small stand-in stack.
"""

from typing import Any, cast

import pytest
import torch

from rlp.core.rollout import rollout_traj
from rlp.core.temporal import trajectory_value

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")

D, H, A_DIM, B = 16, 5, 6, 4


class TinyWM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = torch.nn.Linear(A_DIM, D)
        self.mix = torch.nn.Linear(2 * D, D)

    def predict(self, win_e: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([win_e.mean(dim=1), act_emb.mean(dim=1)], dim=-1)
        mixed: torch.Tensor = self.mix(joint)
        return mixed.unsqueeze(1).expand(-1, win_e.shape[1], -1)


def test_rejects_cpu() -> None:
    from rlp.core.solver.graphed import GraphedRefinement

    if torch.cuda.is_available():
        pytest.skip("CPU-rejection check only meaningful without CUDA")
    with pytest.raises(ValueError, match="CUDA"):
        GraphedRefinement(None, cast(Any, None), "terminal", H, A_DIM, D, "cpu")


@cuda
def test_graphed_step_matches_eager() -> None:
    from rlp.core.solver.graphed import GraphedRefinement

    torch.manual_seed(0)
    dev = "cuda"
    wm = TinyWM().to(dev).eval()
    head = torch.nn.Linear(2 * D, 1).to(dev).eval()
    for p in list(wm.parameters()) + list(head.parameters()):
        p.requires_grad_(False)

    def value(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        scored: torch.Tensor = head(torch.cat([state, goal], dim=-1))
        return scored.squeeze(-1)

    zh = torch.randn(B, 3, D, device=dev)
    ah = torch.zeros(B, 2, A_DIM, device=dev)
    zg = torch.randn(B, D, device=dev)
    A = 0.3 * torch.randn(B, H, A_DIM, device=dev)

    ref = GraphedRefinement(wm, value, "terminal", H, A_DIM, D, dev, warmup_iters=3)
    ref.bind(zh, ah, zg)
    score_g, traj_g, grad_g = ref.step(A)

    A_in = A.detach().requires_grad_(True)
    traj_e = rollout_traj(wm, zh, ah, A_in)
    score_e = trajectory_value(value, traj_e, zg, zh[:, -1], "terminal")
    (grad_e,) = torch.autograd.grad(score_e.sum(), A_in)

    assert torch.allclose(score_g, score_e.detach(), atol=1e-6)
    assert torch.allclose(traj_g, traj_e.detach(), atol=1e-6)
    assert torch.allclose(grad_g, grad_e, atol=1e-6)
    # selection scoring goes through the same graph
    assert torch.allclose(ref.score(A), score_e.detach(), atol=1e-6)
    # rebinding a different context changes results (buffers actually staged)
    ref.bind(zh + 1.0, ah, zg)
    assert not torch.allclose(ref.step(A)[0], score_e.detach(), atol=1e-3)
