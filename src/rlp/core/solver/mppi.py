"""RLP adapter for Stable-WM's MPPI solver."""

from collections.abc import Sequence

from stable_worldmodel.solver.mppi import MPPISolver as StableMPPISolver


class MPPISolver(StableMPPISolver):
    """MPPI with deadline metadata forwarding to an RLP planning cost."""

    def set_align_remaining(self, remaining_chunks: Sequence[int] | None) -> None:
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "set_align_remaining"):
            model.set_align_remaining(remaining_chunks)


__all__ = ["MPPISolver"]
