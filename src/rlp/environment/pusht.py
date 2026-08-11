"""RLP's PushT variant, derived from Stable-WM's public environment."""

from __future__ import annotations

from typing import Any

import numpy as np


def register_rlp_envs() -> None:
    """Register RLP variants lazily so ordinary imports stay lightweight."""
    import gymnasium as gym

    if "rlp/PushT-v1" not in gym.registry:
        gym.register(
            id="rlp/PushT-v1",
            entry_point="rlp.environment.pusht:PushT",
        )


class PushT:  # constructed lazily through __new__ to avoid eager pygame imports
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from stable_worldmodel.envs.pusht.env import PushT as StablePushT

        class RLPushT(StablePushT):
            def __init__(self, *inner_args: Any, **inner_kwargs: Any) -> None:
                super().__init__(*inner_args, **inner_kwargs)
                self.shape_symmetry_angles = {
                    "square": np.pi / 2,
                    "+": np.pi / 2,
                    "I": np.pi,
                    "Z": np.pi,
                }

            def eval_state(self, goal_state: Any, cur_state: Any) -> tuple[bool, float]:
                success, distance = super().eval_state(goal_state, cur_state)
                shape = self.shapes[int(self.variation_space["block"]["shape"].value)]
                symmetry = self.shape_symmetry_angles.get(shape)
                if symmetry is None:
                    return success, distance

                goal_state = np.asarray(goal_state, dtype=np.float64)
                cur_state = np.asarray(cur_state, dtype=np.float64)
                pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
                angle_diff = abs(goal_state[4] - cur_state[4]) % symmetry
                angle_diff = min(angle_diff, symmetry - angle_diff)
                success = bool(pos_diff < 20 and angle_diff < np.pi / 9)
                return success, distance

            def add_Z(
                self,
                position: Any,
                angle: float,
                scale: float = 30,
                color: str = "LightSlateGray",
                mask: Any | None = None,
            ) -> Any:
                import pygame
                import pymunk

                if mask is None:
                    mask = pymunk.ShapeFilter.ALL_MASKS()
                mass = 1
                vertices1 = [
                    (-scale / 2, 0),
                    (-scale / 2, scale),
                    (3 * scale / 2, scale),
                    (3 * scale / 2, 0),
                ]
                vertices2 = [
                    (-3 * scale / 2, 0),
                    (scale / 2, 0),
                    (scale / 2, -scale),
                    (-3 * scale / 2, -scale),
                ]
                inertia = pymunk.moment_for_poly(mass, vertices=vertices1)
                inertia += pymunk.moment_for_poly(mass, vertices=vertices2)
                body = pymunk.Body(mass, inertia)
                shape1, shape2 = pymunk.Poly(body, vertices1), pymunk.Poly(body, vertices2)
                for item in (shape1, shape2):
                    item.color = pygame.Color(color)
                    item.filter = pymunk.ShapeFilter(mask=mask)
                body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
                body.position, body.angle, body.friction = position, angle, 1
                self.space.add(body, shape1, shape2)
                return body

        return RLPushT(*args, **kwargs)


__all__ = ["PushT", "register_rlp_envs"]
