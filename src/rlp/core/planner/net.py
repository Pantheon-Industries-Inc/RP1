import torch
from torch import nn

__all__ = ["PlannerNet"]


class PlannerNet(nn.Module):
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        hidden_dim: int,
        action_limit: float,
        head_scale: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_limit = action_limit
        input_dim = 2 * horizon * action_dim + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )
        if head_scale != 1.0:
            with torch.no_grad():
                self.net[-1].weight.mul_(head_scale)
                self.net[-1].bias.mul_(head_scale)

    def forward(self, plan: torch.Tensor, gradient: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = plan.shape[0]
        inputs = torch.cat(
            [
                plan.reshape(batch_size, -1),
                gradient.reshape(batch_size, -1),
                value.reshape(batch_size, 1),
            ],
            dim=-1,
        )
        update = self.net(inputs).view(batch_size, self.horizon, self.action_dim)
        return (plan + update).clamp(-self.action_limit, self.action_limit)
