#!/usr/bin/env python3
"""Reacher planner-control rows from the latched_measurements dump (planners_read.yaml).
CSV: <base>_<cost>[_cs<seed>]_<planner>_latched_tau<0p1|0p05>_rh5_h25_e<draw>,...,success=<v>
  latent = one arm; value = one arm per training seed (the cell's locked teacher) -> n=6."""
import sys, re, statistics, collections
rows = collections.defaultdict(dict)
pat = re.compile(r"^([a-z]+)_(latent|value)(?:_cs(\d+))?_([a-z0-9]+)_latched_tau(0p1|0p05)_rh\d+_h25_e(\d+),.*success=([0-9.]+)")
cur=None
for line in open(sys.argv[1]):
    m0 = re.match(r"^=== CSV (\S+) (\S+)", line)
    if m0: cur=m0.group(1); continue
    if line.startswith("=== END"): cur=None; continue
    m = pat.match(line.strip())
    if not m or not cur: continue
    base, cost, seed, planner, tau, draw, val = m.groups()
    rows[(base, planner, cost, tau, seed)][int(draw)] = float(val)
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for (base, planner, cost, tau, seed), d in rows.items():
    if len(d) < 3: continue
    cells[(base, planner, tau)][cost].append((seed, statistics.mean(d.values())))
for key in sorted(cells):
    base, planner, tau = key
    print(f"\n### reacher / {base} / h25 -- {planner.upper()}  (tau {tau.replace('0p','0.')})")
    for cost in ("latent", "value"):
        arms = sorted(cells[key].get(cost, []), key=lambda a: (a[0] or ""))
        if not arms: continue
        means=[a[1] for a in arms]
        agg = f"median {statistics.median(means):.1f} mean {statistics.mean(means):.2f} (n={len(means)})" if len(means)>1 else f"{means[0]:.1f} (single arm)"
        print(f"  {cost:7s}: {agg}   [" + "  ".join(f"s{a[0]}:{a[1]:.1f}" if a[0] else f"{a[1]:.1f}" for a in arms) + "]")
