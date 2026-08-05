"""Known external datasets and their canonical local locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """A reproducible external dataset source."""

    name: str
    repo_id: str
    revision: str
    remote_directory: str
    local_directory: str
    required_columns: tuple[str, ...] = ()


DATASETS: dict[str, DatasetSpec] = {
    "ogb_cube": DatasetSpec(
        name="ogb_cube",
        repo_id="galilai-group/ogb_cube_single",
        revision="2f0d4deb19cedaedfc71a55029f93eb9dbd36665",
        remote_directory="ogb_cube_single.lance",
        local_directory="ogb_cube_single.lance",
        required_columns=(
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "observation",
            "qpos",
            "qvel",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        ),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a named dataset or raise an error listing valid choices."""

    try:
        return DATASETS[name]
    except KeyError as error:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; available datasets: {choices}") from error


def data_home(override: str | Path | None = None) -> Path:
    """Return the data cache root, outside the source tree by default."""

    if override is not None:
        return Path(override).expanduser().resolve()
    configured = os.environ.get("RLP_DATA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "rlp"


def dataset_path(spec: DatasetSpec, cache_root: str | Path | None = None) -> Path:
    """Return the canonical local path for ``spec``."""

    return data_home(cache_root) / "datasets" / spec.local_directory


__all__ = ["DATASETS", "DatasetSpec", "data_home", "dataset_path", "get_dataset_spec"]
