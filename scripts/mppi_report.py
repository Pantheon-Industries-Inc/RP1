#!/usr/bin/env python3
"""MPPI baseline rows from the planners CSV dump (scripts/sky/tools/planners_read.yaml).

  python3 scripts/mppi_report.py dump.txt

CSV rows: <env>_<base>_<cost>[_cs<seed>]_<planner>_h<off>_e<draw>,<success>
  latent cost  -> one arm, n=3 draws (no critic involved)
  value  cost  -> one arm per TRAINING SEED (the cell's locked teacher snapshot), n=6 seeds x 3 draws
"""
import sys, re, statistics, collections
rows = collections.defaultdict(dict)   # (tag, env, base, cost, seed, off) -> draw -> value
pat = re.compile(r"^([a-z0-9]+)_([a-z]+)_(latent|value)(?:_cs(\d+))?_([a-z0-9]+)_h(\d+)_e(\d+),([0-9.]+|FAIL)\s*$")
cur = None
for line in open(sys.argv[1]):
    m = re.match(r"^=== CSV (\S+) (\S+)", line)
    if m: cur = m.group(1); continue
    if line.startswith("=== END"): cur = None; continue
    if not cur: continue
    m = pat.match(line.strip())
    if not m or m.group(8) == "FAIL": continue
    env, base, cost, seed, planner, off, draw, val = m.groups()
    rows[(cur, env, base, cost, planner, off, seed)][int(draw)] = float(val)
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for (tag, env, base, cost, planner, off, seed), d in rows.items():
    if len(d) < 3: continue
    cells[(env, base, off, planner)][cost].append((seed, statistics.mean(d.values()), d))
for key in sorted(cells):
    env, base, off, planner = key
    print(f"\n### {env} / {base} / h{off} -- {planner.upper()}")
    for cost in ("latent", "value"):
        arms = sorted(cells[key].get(cost, []), key=lambda a: (a[0] or ""))
        if not arms: continue
        means = [a[1] for a in arms]
        detail = "  ".join(f"s{a[0]}:{a[1]:.1f}" if a[0] else f"{a[1]:.1f}" for a in arms)
        agg = (f"median {statistics.median(means):.1f} mean {statistics.mean(means):.2f} (n={len(means)})"
               if len(means) > 1 else f"{means[0]:.1f} (single arm, draws " +
               "/".join(f"{v:.0f}" for _, v in sorted(arms[0][2].items())) + ")")
        print(f"  {cost:7s}: {agg}   [{detail}]")
