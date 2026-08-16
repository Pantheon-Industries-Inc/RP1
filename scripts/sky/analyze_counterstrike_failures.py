"""Summarize counterstrike failure-analysis probes: planned value E per arm.

Usage: python analyze_counterstrike_failures.py <results_dir> <R1> <R2> ...

Each arm directory <results_dir>/r<R>/probes holds one probe_NNNN.pt per
replan call; each probe stores E (planned trajectory value, lower = better,
argmin objective) for the batch of parallel environments at that replan.
Prints per-arm distributions and the per-environment E trajectory so the
optimization-limited vs value-miscalibration verdict is one look.
"""

import sys
from pathlib import Path

import torch


def main() -> None:
    root = Path(sys.argv[1])
    arms = sys.argv[2:]
    per_arm: dict[str, torch.Tensor] = {}
    for arm in arms:
        probe_dir = root / f"r{arm}" / "probes"
        files = sorted(probe_dir.glob("probe_*.pt"))
        if not files:
            print(f"r{arm}: NO PROBES at {probe_dir}")
            continue
        # (replans, B) planned value at each replan for each parallel env
        e = torch.stack([torch.load(f, map_location="cpu")["E"].reshape(-1) for f in files])
        per_arm[arm] = e
        flat = e.reshape(-1)
        q = torch.quantile(flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
        print(
            f"r{arm}: replans={e.shape[0]} envs={e.shape[1]} "
            f"E mean={flat.mean():.3f} "
            f"p10/p25/p50/p75/p90={'/'.join(f'{v:.2f}' for v in q)}"
        )
        first, last = e[0], e[-1]
        print(f"  first replan E median={first.median():.3f}  last replan E median={last.median():.3f}")
    if len(per_arm) >= 2:
        keys = sorted(per_arm, key=int)
        base = per_arm[keys[0]]
        for other in keys[1:]:
            o = per_arm[other]
            n = min(base.shape[0], o.shape[0])
            delta = (base[:n] - o[:n]).reshape(-1)
            frac = (delta > 0).float().mean()
            print(
                f"r{keys[0]} vs r{other}: median planned-E improvement "
                f"{delta.median():.3f} ({frac * 100:.0f}% of replans improved)"
            )


if __name__ == "__main__":
    main()
