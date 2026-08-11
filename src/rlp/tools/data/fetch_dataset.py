"""Fetch a registered dataset from Hugging Face into the shared data cache.

Example::

    pixi run tool tool=fetch_dataset dataset=ogb_cube
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import lance
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import DatasetInfo, RepoSibling
from omegaconf import DictConfig

from rlp.config import dispatch, run_hydra
from rlp.data import DatasetSpec, data_home, dataset_path, get_dataset_spec
from rlp.logging import logger


@dataclass(frozen=True)
class FetchResult:
    """The verified location and metadata of a fetched dataset."""

    name: str
    repo_id: str
    revision: str
    path: str
    rows: int
    columns: tuple[str, ...]
    remote_bytes: int
    fetched_at: str


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _dataset_files(info: DatasetInfo, spec: DatasetSpec) -> list[RepoSibling]:
    prefix = f"{spec.remote_directory}/"
    files = [item for item in info.siblings or [] if item.rfilename.startswith(prefix)]
    if not files:
        raise RuntimeError(f"{spec.repo_id}@{spec.revision} does not contain {spec.remote_directory!r}")
    return files


def _remaining_bytes(root: Path, files: list[RepoSibling]) -> tuple[int, int]:
    total = 0
    present = 0
    for item in files:
        if item.size is None:
            raise RuntimeError(f"Hugging Face did not report a size for {item.rfilename!r}")
        size = item.size
        total += size
        relative = PurePosixPath(item.rfilename)
        if ".." in relative.parts or relative.is_absolute():
            raise RuntimeError(f"Unsafe path in dataset repository: {item.rfilename!r}")
        local = root.joinpath(*relative.parts)
        if local.is_file() and local.stat().st_size == size:
            present += size
    return total, max(total - present, 0)


def _disk_free(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _validate_dataset(path: Path, spec: DatasetSpec) -> tuple[int, tuple[str, ...]]:
    if not path.is_dir():
        raise RuntimeError(f"Dataset download did not create {path}")
    dataset = lance.dataset(path)
    columns = tuple(dataset.schema.names)
    missing = sorted(set(spec.required_columns).difference(columns))
    if missing:
        raise RuntimeError(f"Dataset at {path} is missing required columns: {', '.join(missing)}")
    rows = dataset.count_rows()
    if rows <= 0:
        raise RuntimeError(f"Dataset at {path} contains no rows")
    return rows, columns


def _manifest_path(spec: DatasetSpec, cache_root: str | Path | None) -> Path:
    return data_home(cache_root) / "datasets" / ".rlp" / f"{spec.name}.json"


def _read_completed_fetch(spec: DatasetSpec, cache_root: str | Path | None) -> FetchResult | None:
    manifest = _manifest_path(spec, cache_root)
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text())
        if not isinstance(payload, dict):
            return None
        columns = payload.get("columns")
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            return None
        payload["columns"] = tuple(columns)
        result = FetchResult(**payload)
    except (OSError, TypeError, ValueError):
        return None
    destination = dataset_path(spec, cache_root)
    if result.revision != spec.revision or Path(result.path) != destination:
        return None
    rows, columns = _validate_dataset(destination, spec)
    if rows != result.rows or columns != result.columns:
        return None
    return result


def _write_manifest(result: FetchResult, spec: DatasetSpec, cache_root: str | Path | None) -> None:
    manifest = _manifest_path(spec, cache_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)


def fetch_dataset(
    name: str,
    *,
    cache_root: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    max_workers: int = 4,
    min_free_gib: float = 2.0,
) -> FetchResult | None:
    """Fetch and validate a named dataset; return ``None`` for a dry run."""

    spec = get_dataset_spec(name)
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if min_free_gib < 0:
        raise ValueError("min_free_gib cannot be negative")
    destination = dataset_path(spec, cache_root)
    if not force:
        completed = _read_completed_fetch(spec, cache_root)
        if completed is not None:
            logger.info(f"Dataset {name!r} is already ready at {destination} ({completed.rows:,} rows)")
            return completed

    info = HfApi().dataset_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    if info.sha != spec.revision:
        raise RuntimeError(f"Resolved revision {info.sha!r} does not match pinned revision {spec.revision!r}")

    files = _dataset_files(info, spec)
    download_root = destination.parent
    remote_bytes, remaining_bytes = _remaining_bytes(download_root, files)
    if force:
        remaining_bytes = remote_bytes
    free_bytes = _disk_free(download_root)
    reserve_bytes = int(min_free_gib * 1024**3)
    logger.info(f"Dataset: {name} ({spec.repo_id}@{spec.revision[:12]})")
    logger.info(f"Destination: {destination}")
    logger.info(
        f"Remote size: {_format_bytes(remote_bytes)}; remaining: {_format_bytes(remaining_bytes)}; "
        f"free: {_format_bytes(free_bytes)}"
    )
    if remaining_bytes + reserve_bytes > free_bytes:
        required = _format_bytes(remaining_bytes + reserve_bytes)
        raise RuntimeError(f"Insufficient disk space at {download_root}: need {required} including reserve")
    if dry_run:
        logger.info("Dry run complete; no files were downloaded")
        return None

    download_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        repo_type="dataset",
        revision=spec.revision,
        local_dir=download_root,
        allow_patterns=[f"{spec.remote_directory}/**"],
        force_download=force,
        max_workers=max_workers,
    )

    _, incomplete_bytes = _remaining_bytes(download_root, files)
    if incomplete_bytes:
        raise RuntimeError(f"Dataset download is incomplete: {_format_bytes(incomplete_bytes)} are missing")

    rows, columns = _validate_dataset(destination, spec)
    result = FetchResult(
        name=spec.name,
        repo_id=spec.repo_id,
        revision=spec.revision,
        path=str(destination),
        rows=rows,
        columns=columns,
        remote_bytes=remote_bytes,
        fetched_at=datetime.now(UTC).isoformat(),
    )
    _write_manifest(result, spec, cache_root)
    logger.info(f"Dataset ready at {destination} ({rows:,} rows)")
    return result


def _run(cfg: DictConfig) -> FetchResult | None:
    return fetch_dataset(
        str(cfg.dataset),
        cache_root=cfg.cache_root,
        dry_run=bool(cfg.dry_run),
        force=bool(cfg.force),
        max_workers=int(cfg.max_workers),
        min_free_gib=float(cfg.min_free_gib),
    )


def main() -> object:
    return run_hydra(dispatch, config_name="tools/fetch_dataset")


if __name__ == "__main__":
    main()
