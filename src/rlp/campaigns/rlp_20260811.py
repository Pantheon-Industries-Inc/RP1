"""Validated manifest for the corrected 2026-08-11 RLP campaign."""

from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Sequence

from rlp.logging import logger

GPU_LIMIT = 64


@dataclasses.dataclass(frozen=True)
class Job:
    name: str
    group: str
    base: str
    horizon: int | None = None
    shard: str | None = None
    priority: str = "p1"
    gpus: int = 4
    cpus: int = 80
    memory_gib: int = 920


def jobs() -> list[Job]:
    matrix: list[Job] = []
    for horizon in (25, 100):
        for base in ("lejepa", "pldm"):
            for shard in ("a", "b", "c"):
                matrix.append(Job(f"rlp11-tw-{base}-h{horizon}-{shard}", "tworoom", base, horizon, shard))
    matrix.extend(
        [
            Job("rlp11-reacher-lejepa-deadline", "reacher", "lejepa"),
            Job("rlp11-reacher-pldm-deadline", "reacher", "pldm"),
            Job("rlp11-ogbench-lewm-terminal", "ogbench", "lewm"),
            Job("rlp11-ogbench-pldm-terminal", "ogbench", "pldm"),
        ]
    )
    return matrix


def validate(matrix: Sequence[Job] | None = None) -> list[Job]:
    selected = list(matrix or jobs())
    errors: list[str] = []
    names = [job.name for job in selected]
    if len(names) != len(set(names)):
        errors.append("job names must be unique")
    if len(selected) != 16:
        errors.append(f"expected 16 jobs, got {len(selected)}")
    if sum(job.gpus for job in selected) > GPU_LIMIT:
        errors.append(f"GPU request exceeds {GPU_LIMIT}")
    for job in selected:
        if job.priority != "p1":
            errors.append(f"{job.name}: priority must be p1")
        if job.memory_gib / job.gpus > 241:
            errors.append(f"{job.name}: memory exceeds 241 GiB/GPU")
    tworoom = [job for job in selected if job.group == "tworoom"]
    expected = {(horizon, base, shard) for horizon in (25, 100) for base in ("lejepa", "pldm") for shard in "abc"}
    actual = {(job.horizon, job.base, job.shard) for job in tworoom}
    if actual != expected:
        errors.append("TwoRoom matrix must be 2 horizons x 2 bases x 3 shards")
    if errors:
        raise ValueError("invalid campaign:\n  - " + "\n  - ".join(errors))
    return selected


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m rlp.campaigns.rlp_20260811 [validate|matrix]")
    action = sys.argv[1] if len(sys.argv) == 2 else "validate"
    if action not in {"validate", "matrix"}:
        raise SystemExit(f"unknown action {action!r}; expected validate or matrix")
    matrix = validate()
    if action == "matrix":
        logger.info(json.dumps([dataclasses.asdict(job) for job in matrix], indent=2))
    else:
        logger.info(f"validated {len(matrix)} jobs / {sum(job.gpus for job in matrix)} H200")


if __name__ == "__main__":
    main()
