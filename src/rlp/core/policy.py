"""RLP-specific policy baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_worldmodel.policy import BasePolicy


class NoMovePolicy(BasePolicy):
    """Emit a zero action as the honest no-planning success floor."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = "nomove"

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros_like(self.env.action_space.sample())


__all__ = ["NoMovePolicy"]
