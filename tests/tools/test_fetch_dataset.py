"""Tests for the external dataset registry and fetch tool."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from huggingface_hub.hf_api import DatasetInfo

import rlp.tools.data.fetch_dataset as fetch_module
from rlp.data import dataset_path, get_dataset_spec


def _dataset_info(size: int = 1024) -> DatasetInfo:
    spec = get_dataset_spec("ogb_cube")
    return DatasetInfo(  # type: ignore[no-untyped-call]
        id=spec.repo_id,
        sha=spec.revision,
        siblings=[{"rfilename": f"{spec.remote_directory}/data/example.lance", "size": size}],
    )


def test_dataset_path_uses_external_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RLP_DATA_HOME", str(tmp_path))
    assert dataset_path(get_dataset_spec("ogb_cube")) == tmp_path / "datasets" / "ogb_cube_single.lance"


def test_unknown_dataset_lists_available_names() -> None:
    with pytest.raises(ValueError, match="available datasets: ogb_cube"):
        get_dataset_spec("missing")


def test_cube_registry_includes_all_evaluation_pose_columns() -> None:
    required = set(get_dataset_spec("ogb_cube").required_columns)
    assert {"privileged_block_0_pos", "privileged_block_0_quat"}.issubset(required)


def test_dry_run_checks_metadata_without_downloading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _dataset_info()

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> DatasetInfo:
            assert repo_id == info.id
            assert revision == info.sha
            assert files_metadata
            return info

    monkeypatch.setattr(fetch_module, "HfApi", FakeApi)
    monkeypatch.setattr(fetch_module, "_disk_free", lambda path: 10 * 1024**3)
    monkeypatch.setattr(fetch_module, "snapshot_download", lambda **kwargs: pytest.fail("downloaded during dry run"))

    result = fetch_module.fetch_dataset("ogb_cube", cache_root=tmp_path, dry_run=True)

    assert result is None
    assert not dataset_path(get_dataset_spec("ogb_cube"), tmp_path).exists()


def test_fetch_validates_and_reuses_completed_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _dataset_info()
    api_calls = 0

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> DatasetInfo:
            nonlocal api_calls
            api_calls += 1
            return info

    def fake_download(**kwargs: object) -> str:
        destination = Path(str(kwargs["local_dir"])) / "ogb_cube_single.lance"
        columns: dict[str, list[object]] = {
            column: [b"pixel" if column == "pixels" else 1.0]
            for column in get_dataset_spec("ogb_cube").required_columns
        }
        lance.write_dataset(pa.table(columns), destination)
        remote_file = Path(str(kwargs["local_dir"])) / "ogb_cube_single.lance" / "data" / "example.lance"
        remote_file.write_bytes(b"0" * 1024)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(fetch_module, "HfApi", FakeApi)
    monkeypatch.setattr(fetch_module, "_disk_free", lambda path: 10 * 1024**3)
    monkeypatch.setattr(fetch_module, "snapshot_download", fake_download)

    first = fetch_module.fetch_dataset("ogb_cube", cache_root=tmp_path)
    second = fetch_module.fetch_dataset("ogb_cube", cache_root=tmp_path)

    assert first is not None
    assert first.rows == 1
    assert second == first
    assert api_calls == 1
    assert (tmp_path / "datasets" / ".rlp" / "ogb_cube.json").is_file()


def test_fetch_refuses_insufficient_disk_space(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info = _dataset_info(size=5 * 1024**3)

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool) -> DatasetInfo:
            return info

    monkeypatch.setattr(fetch_module, "HfApi", FakeApi)
    monkeypatch.setattr(fetch_module, "_disk_free", lambda path: 4 * 1024**3)

    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        fetch_module.fetch_dataset("ogb_cube", cache_root=tmp_path, min_free_gib=1.0)
