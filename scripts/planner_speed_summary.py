"""Aggregate per-decision solver latencies from planner-speed benchmark logs.

Reads the logs written by ``scripts/sky/planner_speed.yaml`` — each solver
prints one ``<planner> solve completed in <seconds> seconds`` line per
decision — and reports median/mean ms per decision. The first two decisions of
every run are dropped: they carry CUDA context, allocator, and (for the graphed
LIP path) capture warmup.

    python scripts/planner_speed_summary.py <log_dir> <env> <base> "1 50" "l2o rlp cem mppi adam"
"""

from __future__ import annotations

import pathlib
import re
import statistics
import sys

LABELS = {"l2o": "L2O", "rlp": "LIP", "cem": "CEM", "mppi": "MPPI", "adam": "Adam", "dmpo": "DMPO"}
WARMUP_DECISIONS = 2


def main() -> int:
    if len(sys.argv) != 6:
        print(__doc__)
        return 2
    directory, env, base = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    batches, planners = sys.argv[4].split(), sys.argv[5].split()
    print(f"{'planner':8s} {'B':>4s} {'n':>4s} {'median_ms':>10s} {'mean_ms':>9s} {'per_ep_s':>9s}")
    for batch in batches:
        for planner in planners:
            log = directory / f"log_{env}_{base}_{planner}_b{batch}.txt"
            if not log.is_file():
                print(f"{planner:8s} {batch:>4s}  MISSING {log.name}")
                continue
            pattern = rf"{LABELS[planner]} solve completed in ([\d.]+) seconds"
            times = [float(value) for value in re.findall(pattern, log.read_text())]
            warm = times[WARMUP_DECISIONS:] if len(times) > WARMUP_DECISIONS + 1 else times
            if not warm:
                print(f"{planner:8s} {batch:>4s}  NO_TIMINGS (raw={len(times)})")
                continue
            median = 1000 * statistics.median(warm)
            mean = 1000 * statistics.fmean(warm)
            # one h100 episode = 8 decisions; per-episode planner seconds at this batch
            print(f"{planner:8s} {batch:>4s} {len(warm):>4d} {median:>10.1f} {mean:>9.1f} {8 * mean / 1000:>9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
