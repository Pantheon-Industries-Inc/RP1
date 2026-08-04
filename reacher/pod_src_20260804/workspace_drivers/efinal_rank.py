"""Where does E_final rank, and is there an E_final level that predicts the peak?

Two questions:
  1. Does the anti-correlation (corr(E_final, HELD) = +0.583, measured across
     FINAL actors) also hold WITHIN a training run, snapshot by snapshot?
  2. More useful: does held peak at a CONSISTENT E_final value across seeds and
     configs? If it does, "train until E_final reaches X" is a single recipe that
     needs no environment evaluation -- which would remove the main objection to
     early stopping (that it is really N recipes plus a selection procedure).

Pairs the E_final printed in each training log at step N with the held@0.05 that
snapshot scored on the selection seeds {50,51}.
"""

import re
from pathlib import Path

SUM = Path("/workspace/results/summary_earlystop_lejepa.csv")
LOGS = Path("/workspace/logs/earlystop")
STEPS = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
SEL = [50, 51]

rows = {}
for line in SUM.read_text().splitlines():
    rows[line.split(",", 1)[0]] = line


def held(pre):
    v = []
    for s in SEL:
        ln = rows.get(f"{pre}{s}")
        if ln:
            m = re.search(r"held=([0-9.]+)", ln)
            if m:
                v.append(float(m.group(1)))
    return sum(v) / len(v) if v else None


def efinals(tag, seed):
    """E_final at each printed step in the training log."""
    f = LOGS / f"train_{tag}_s{seed}.log"
    out = {}
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        m = re.match(r"step (\d+): E_final ([\d.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


print("%-9s %-4s %6s %8s %7s %s" % ("config", "seed", "step", "E_final", "HELD", "peak?"))
print("-" * 56)
peaks = []
allpairs = []
for tag in ("base", "lr1e4", "gearly"):
    for seed in (0, 1, 2):
        ef = efinals(tag, seed)
        if not ef:
            continue
        series = []
        for st in STEPS:
            h = held(f"es_{tag}_s{seed}_{st}_sel")
            # E_final is printed every 500 steps; take the nearest printed step <= st
            cand = [k for k in ef if k <= st]
            e = ef[max(cand)] if cand else None
            if h is not None and e is not None:
                series.append((st, e, h))
        hfin = held(f"es_{tag}_s{seed}_final_sel")
        cand = [k for k in ef]
        if hfin is not None and cand:
            series.append((8000, ef[max(cand)], hfin))
        if not series:
            continue
        best = max(series, key=lambda x: x[2])
        for st, e, h in series:
            mark = "  <-- PEAK" if (st, e, h) == best else ""
            print("%-9s s%-3d %6d %8.3f %7.1f %s" % (tag, seed, st, e, h, mark))
            allpairs.append((e, h))
        peaks.append((tag, seed, best[0], best[1], best[2]))
        print()

print("=" * 56)
print("PEAK per actor:")
print("%-9s %-4s %6s %8s %7s" % ("config", "seed", "step", "E_final", "HELD"))
for tag, seed, st, e, h in peaks:
    print("%-9s s%-3d %6d %8.3f %7.1f" % (tag, seed, st, e, h))

if peaks:
    es = [p[3] for p in peaks]
    print(f"\nE_final at the peak: min {min(es):.3f}  max {max(es):.3f}  "
          f"mean {sum(es)/len(es):.3f}  spread {max(es)-min(es):.3f}")
    print("=> a single 'stop when E_final <= X' rule is viable only if that spread is small")

if len(allpairs) >= 4:
    def corr(xs, ys):
        n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
        sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        sxx = sum((x-mx)**2 for x in xs)**0.5
        syy = sum((y-my)**2 for y in ys)**0.5
        return sxy/(sxx*syy) if sxx and syy else float("nan")
    xs = [p[0] for p in allpairs]; ys = [p[1] for p in allpairs]
    print(f"\nwithin-run corr(E_final, HELD) over {len(allpairs)} snapshots = {corr(xs, ys):+.3f}")
    print("  (positive => lower imagined cost goes with WORSE real performance,")
    print("   i.e. the same exploitation seen across final actors, now within training)")
