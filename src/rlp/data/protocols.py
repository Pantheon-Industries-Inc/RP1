"""Typed dataset boundary used by cache, training, and evaluation code."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from numpy.typing import NDArray

Array = NDArray[Any]
RowBatch = dict[str, Array]


class Dataset(Protocol):
    """Small Stable-WM dataset surface consumed by RLP."""

    column_names: Sequence[str]

    def get_col_data(self, name: str) -> Array: ...

    def get_row_data(self, indices: list[int]) -> RowBatch: ...


__all__ = ["Array", "Dataset", "RowBatch"]
