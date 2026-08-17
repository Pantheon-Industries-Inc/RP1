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
    print(
        f"{'planner':8s} {'B':>4s} {'n':>4s} {'median_ms':>10s} {'mean_ms':>9s} "
        f"{'plan_med':>9s} {'plan_max':>9s} {'plan_min':>9s}"
    )
    for batch in batches:
        for planner in planners:
            # one log per process (seed); each carries its own warmup, so drop
            # the first decisions per file rather than once across the pool
            logs = sorted(directory.glob(f"log_{env}_{base}_{planner}_b{batch}_s*.txt"))
            legacy = directory / f"log_{env}_{base}_{planner}_b{batch}.txt"
            if not logs and legacy.is_file():
                logs = [legacy]
            if not logs:
                print(f"{planner:8s} {batch:>4s}  MISSING log_{env}_{base}_{planner}_b{batch}_s*.txt")
                continue
            pattern = rf"{LABELS[planner]} solve completed in ([\d.]+) seconds"
            # LIP and L2O also report the encode/plan split; plan-only is the
            # number comparable to a graph-capture benchmark
            plan_pattern = rf"{LABELS[planner]} solve completed in [\d.]+ seconds \(encode [\d.]+, plan ([\d.]+)\)"
            warm: list[float] = []
            plan_warm: list[float] = []
            raw_total = 0
            for log in logs:
                text = log.read_text()
                times = [float(value) for value in re.findall(pattern, text)]
                plans = [float(value) for value in re.findall(plan_pattern, text)]
                raw_total += len(times)
                warm.extend(times[WARMUP_DECISIONS:] if len(times) > WARMUP_DECISIONS else [])
                plan_warm.extend(plans[WARMUP_DECISIONS:] if len(plans) > WARMUP_DECISIONS else [])
            if not warm:
                print(f"{planner:8s} {batch:>4s}  NO_TIMINGS (raw={raw_total}, files={len(logs)})")
                continue
            median = 1000 * statistics.median(warm)
            mean = 1000 * statistics.fmean(warm)
            # plan-only spread exposes bimodality (graph capture / shape changes)
            if plan_warm:
                plan_med = f"{1000 * statistics.median(plan_warm):.1f}"
                plan_max = f"{1000 * max(plan_warm):.1f}"
                plan_min = f"{1000 * min(plan_warm):.1f}"
            else:
                plan_med = plan_max = plan_min = "-"
            print(
                f"{planner:8s} {batch:>4s} {len(warm):>4d} {median:>10.1f} {mean:>9.1f} "
                f"{plan_med:>9s} {plan_max:>9s} {plan_min:>9s}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
