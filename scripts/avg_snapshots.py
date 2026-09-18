"""Average deployable planner snapshots into one checkpoint.

    python scripts/avg_snapshots.py OUT.pt SNAP1.pt SNAP2.pt ...

Ships one actor running a single K-iteration pass, so the deploy budget is
unchanged; see `rlp.train.soup`.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rlp.train.soup import average_planner_checkpoints

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    inputs = [Path(p) for p in sys.argv[2:]]
    average_planner_checkpoints(inputs, out)
    print(f"[soup] averaged {len(inputs)} snapshots -> {out}")
    for p in inputs:
        print(f"[soup]   {p.name}")
