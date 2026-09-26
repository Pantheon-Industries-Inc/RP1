#!/usr/bin/env python3
"""FIXED stop per cell -- the main protocol (user decision 2026-09-22).

For each cell (tag = env x base x stack) choose ONE (teacher budget, actor step) pair by the MEAN
selection score over seeds, on the stack's own horizon (h25 for '-h25' tags, h100 otherwise),
ties -> smallest teacher budget, then smallest actor step. Only pairs scored on every seed of the
cell are eligible. Then, per seed: if the deployed snapshot (the yaml's per-seed pick) IS that pair,
its report number exists; otherwise the snapshot must be re-evaluated on draws 42-44.

Input: SNAPH lines from scripts/collect_ckptval.sh, plus the deployed lines from
scripts/collect_ckpt_select.sh (\"<tag> <row> <seed> <val>\" where val = mean_all of the deployed
snapshot) to detect which snapshot was deployed.

    python3 scripts/fixed_stop_select.py snaph.txt ckpt_select.txt [--seeds 6]
"""
import collections, statistics, sys
want_seeds = 6
args = [a for a in sys.argv[1:]]
if "--seeds" in args:
    want_seeds = int(args[args.index("--seeds") + 1]); del args[args.index("--seeds"):args.index("--seeds") + 2]
snap = collections.defaultdict(lambda: collections.defaultdict(dict))   # tag -> (row, step) -> seed -> (h25, h100)
dep = {}
import re
def cell(tag):   # one cell's seeds may run under several tags (…-s345-…, …-s25-…, pod shards): merge them
    return re.sub(r"-s\d+(?=-)", "", tag)
for f in args:
    for line in open(f):
        p = line.split()
        if p and p[0] == "SNAPH" and len(p) == 7:
            snap[cell(p[1])][(p[2], p[4])][int(p[3])] = (None if p[5] == "na" else float(p[5]), None if p[6] == "na" else float(p[6]))
        elif p and p[0] == "SNAP" and len(p) == 6 and p[1].startswith("rlp-re-"):
            snap[cell(p[1])][(p[2], p[4] if p[4] else "final")][int(p[3])] = (float(p[5]), None)   # reacher: val is the h25 (tau .05/.1 avg) score
        elif len(p) in (4, 5) and not p[0].startswith("SNAP"):
            dep[(cell(p[0]), p[1], int(p[2]))] = (float(p[3]), p[4] if len(p) == 5 else None)   # (val, deployed step)
def budget(row): return int(row.split("_t")[1]) if "_t" in row else 18000
def stepnum(st): return 10**9 if st == "final" else int(st)
# guard (2026-09-23): a row whose snapshots lack 8000/10000 was trained under the old 8k actor cap and reused by
# file existence -- it is NOT a valid 12k row; report it so the cell is never locked on it
for _c, _pairs in snap.items():
    _rows = {}
    for (_row, _step), _d in _pairs.items():
        for _s in _d: _rows.setdefault((_row, _s), set()).add(_step)
    _bad = sorted(k for k, st in _rows.items() if not ({"8000", "10000"} & st))
    if _bad: print(f"{_c}: WARNING {len(_bad)} row(s) without 8k/10k snapshots (old 8k cap?): " + ", ".join(f"{r} s{s}" for r, s in _bad[:8]))
for tag, pairs in sorted(snap.items()):
    hidx = 0 if ("-h25" in tag or tag.startswith("rlp-re-")) else 1
    seeds = sorted({s for v in pairs.values() for s in v})
    # completeness: every seed must have scored the FULL grid (6 teacher rows x 6 actor steps = 36 pairs)
    per_seed = {sd: sum(1 for v in pairs.values() if sd in v) for sd in seeds}
    incomplete = {sd: n for sd, n in per_seed.items() if n < 36}
    elig = [k for k, v in pairs.items() if set(v) >= set(seeds) and all(x[hidx] is not None for x in v.values())]
    if not elig:
        print(f"{tag}: no pair scored on all seeds yet ({len(seeds)} seeds seen)"); continue
    score = lambda k: statistics.mean(v[hidx] for v in pairs[k].values())
    best = max(elig, key=lambda k: (score(k), -budget(k[0]), -stepnum(k[1])))
    hz = "h25" if hidx == 0 else "h100"
    status = "" if (len(seeds) >= want_seeds and not incomplete) else "  [PRELIMINARY: " + (f"{len(seeds)}/{want_seeds} seeds" if len(seeds) < want_seeds else "") + (("; rows/36 per seed " + str(incomplete)) if incomplete else "") + "]"
    print(f"{tag}: {len(seeds)}/{want_seeds} seeds -> FIXED teacher {budget(best[0])} / actor {best[1]}  (mean {hz} val {score(best):.2f})" + status)
    for s in seeds:
        v = pairs[best][s]; d = dep.get((tag, best[0], s))
        same = d is not None and d[1] is not None and str(d[1]) == str(best[1])   # same SNAPSHOT deployed, not just same score
        print(f"    s{s}: {hz} val {v[hidx]}  -> " + ("report EXISTS (deployed snapshot == fixed pair)" if same else f"RE-EVAL {best[0]}_s{s}{'' if best[1]=='final' else '_step'+best[1]} on 42-44"))
