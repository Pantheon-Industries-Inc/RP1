"""RLP adapter for Stable-WM's CEM solver."""

from collections.abc import Sequence

from stable_worldmodel.solver.cem import CEMSolver as StableCEMSolver


class CEMSolver(StableCEMSolver):
    """CEM with deadline metadata forwarding to an RLP planning cost."""

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "set_align_remaining"):
            model.set_align_remaining(remaining_chunks)


__all__ = ["CEMSolver"]
