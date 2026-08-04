import numpy as np
from dm_control.suite import reacher

_DEFAULT_QPOS_THRESHOLD = 0.05


class ReacherQPosMatchTask(reacher.Reacher):
    """Reacher task that terminates (success) when qpos matches a target qpos.

    The target qpos is set externally via the env's `set_target_qpos()`. Until
    it is set, the task never terminates early.

    Args:
        target_size: Tolerance radius for the standard finger-to-target reward.
        random: Random seed or RandomState.
        qpos_threshold: Per-joint absolute tolerance (in radians) used to
            determine whether the current qpos matches the target qpos.
    """

    def __init__(
        self, target_size, random=None, qpos_threshold=_DEFAULT_QPOS_THRESHOLD
    ):
        super().__init__(target_size=target_size, random=random)
        self.target_qpos = None
        self.qpos_threshold = qpos_threshold

    def is_matched(self, physics) -> bool:
        """Whether every joint is currently inside the tolerance ball."""
        if self.target_qpos is None:
            return False
        diff = np.abs(physics.data.qpos - self.target_qpos)
        return bool(np.all(diff < self.qpos_threshold))

    def get_termination(self, physics):
        # Deliberately never terminates early. Ending the episode at first
        # contact makes "reached and held" indistinguishable from "swung
        # through", and scores the latter as a success. The episode now runs the
        # full evaluation budget and residency is reported per step via the
        # env's info['qpos_in_ball'], so the harness can require the arm to
        # still be on target at the end.
        return None
