"""Ground-truth task-state oracle for TwoRoom.

The paper uses an *oracle task-state cost* as an upper bound and to audit
candidate rankings (SCSA). TwoRoom exposes the true agent/target/door geometry
(``envs/two_room/env.py``), so we can compute the exact shortest-path-through-
doorway geodesic the env itself uses (``_constrain_target_by_min_steps``):

    same room  ->  ||agent - target||
    crossing   ->  min_door [ ||agent - door|| + ||door - target|| ]

We also provide a vectorised analytic point-mass rollout (mirroring the env's
``_apply_collisions``) so the oracle can act as an MPC planner and score CEM
candidates for diagnostics without stepping the real (slow) simulator.
"""

from __future__ import annotations

import numpy as np
import torch

WALL_CENTER = 112.0
BORDER_SIZE = 14.0
IMG_SIZE = 224.0


def geodesic_distance(
    agent: torch.Tensor,
    target: torch.Tensor,
    door_points: torch.Tensor,
    wall_center: float = WALL_CENTER,
    wall_axis: int = 1,
) -> torch.Tensor:
    """Shortest path agent->target respecting the wall.

    Args:
        agent: ``(..., 2)`` agent positions.
        target: ``(..., 2)`` target positions (broadcastable to ``agent``).
        door_points: ``(K, 2)`` doorway center points on the wall.
        wall_center: wall coordinate on the split axis.
        wall_axis: 1 = vertical wall (split on x), 0 = horizontal (split on y).
    Returns:
        ``(...)`` geodesic distances.
    """
    split = 0 if wall_axis == 1 else 1
    same_side = (agent[..., split] < wall_center) == (target[..., split] < wall_center)
    euclid = torch.linalg.vector_norm(agent - target, dim=-1)

    # crossing: go through the best door
    a = agent.unsqueeze(-2)                      # (..., 1, 2)
    t = target.unsqueeze(-2)                     # (..., 1, 2)
    d = door_points.view(*([1] * (agent.ndim - 1)), -1, 2)  # (..., K, 2)
    through = (
        torch.linalg.vector_norm(a - d, dim=-1)
        + torch.linalg.vector_norm(t - d, dim=-1)
    ).min(dim=-1).values                          # (...)
    return torch.where(same_side, euclid, through)


def rollout_dynamics(
    pos0: torch.Tensor,
    actions: torch.Tensor,
    speed: float,
    door_points: torch.Tensor,
    door_half: float,
    wall_thickness: int = 10,
    agent_radius: float = 7.0,
    wall_center: float = WALL_CENTER,
    wall_axis: int = 1,
) -> torch.Tensor:
    """Vectorised analytic point-mass rollout with wall+border collisions.

    Args:
        pos0: ``(N, 2)`` initial positions.
        actions: ``(N, T, 2)`` action sequence in ``[-1, 1]``.
        speed: action-to-pixel scale.
        door_points: ``(K, 2)`` doorway centers.
        door_half: door half-extent (passable band along the wall).
    Returns:
        ``(N, 2)`` terminal positions.
    """
    pos = pos0.clone()
    split = 0 if wall_axis == 1 else 1
    perp = 1 - split  # axis along the wall
    half = wall_thickness // 2
    eff_left = wall_center - half - agent_radius
    eff_right = wall_center + half + agent_radius
    door_perp = door_points[:, perp]  # (K,)

    for t in range(actions.shape[1]):
        nxt = pos + torch.clamp(actions[:, t], -1.0, 1.0) * speed
        # border clamp
        lo = BORDER_SIZE + agent_radius
        hi = IMG_SIZE - BORDER_SIZE - agent_radius
        nxt = nxt.clamp(lo, hi)
        # wall collision on the split axis
        started_left = pos[:, split] < wall_center
        # passable if perpendicular coord within any door band
        in_door = (
            (nxt[:, perp].unsqueeze(-1) - door_perp.unsqueeze(0)).abs() <= door_half
        ).any(dim=-1)
        cross_l2r = started_left & (nxt[:, split] > eff_left) & ~in_door
        cross_r2l = (~started_left) & (nxt[:, split] < eff_right) & ~in_door
        nxt[cross_l2r, split] = eff_left - 0.5
        nxt[cross_r2l, split] = eff_right + 0.5
        pos = nxt
    return pos


class TwoRoomOracleCost:
    """Oracle MPC planner: simulate candidates with true dynamics, score by
    geodesic distance to goal. Implements the Costable interface.

    Args:
        door_points: ``(K, 2)`` doorway centers.
        speed: action scale (env ``agent.speed``).
        door_half / wall_thickness / agent_radius: geometry for collisions.
        state_key / goal_state_key: info_dict keys for current agent xy and goal
            agent xy.
    """

    def __init__(
        self,
        door_points,
        speed: float,
        door_half: float = 21.0,
        wall_thickness: int = 10,
        agent_radius: float = 7.0,
        state_key: str = "state",
        goal_state_key: str = "goal_state",
        raw_action_dim: int = 2,
    ):
        self.raw_action_dim = raw_action_dim
        self.door_points = torch.as_tensor(door_points, dtype=torch.float32)
        self.speed = float(speed)
        self.door_half = float(door_half)
        self.wall_thickness = wall_thickness
        self.agent_radius = agent_radius
        self.state_key = state_key
        self.goal_state_key = goal_state_key

    def parameters(self):  # no-op; lets solvers query a (cpu) dtype
        return iter([torch.zeros(1)])

    def _terminal_positions(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # action_candidates: (B, S, H, 2); state: (B, S, 2) after CEM expand.
        # The analytic dynamics run on CPU (cheap, vectorised), independent of
        # where the solver placed its tensors.
        B, S, H, A = action_candidates.shape
        pos0 = torch.as_tensor(info_dict[self.state_key], dtype=torch.float32).cpu()
        pos0 = pos0[..., :2].reshape(B * S, 2)
        # unpack action-blocks: each planned step packs (A/raw) env actions of
        # dim `raw` -> simulate H*(A/raw) true env steps.
        raw = self.raw_action_dim
        acts = action_candidates.float().cpu().reshape(B * S, H * (A // raw), raw)
        term = rollout_dynamics(
            pos0, acts, self.speed, self.door_points, self.door_half,
            self.wall_thickness, self.agent_radius,
        )
        return term.reshape(B, S, 2)

    @torch.no_grad()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        term = self._terminal_positions(info_dict, action_candidates)  # (B,S,2) cpu
        goal = torch.as_tensor(info_dict[self.goal_state_key], dtype=torch.float32).cpu()
        goal = goal[..., :2].reshape(term.shape[0], -1, 2)[:, :1, :]  # (B,1,2)
        goal = goal.expand_as(term)
        cost = geodesic_distance(term, goal, self.door_points)  # (B,S) cpu
        return cost.to(action_candidates.device)


__all__ = ["geodesic_distance", "rollout_dynamics", "TwoRoomOracleCost"]
