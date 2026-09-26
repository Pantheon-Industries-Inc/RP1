#!/usr/bin/env python3
"""Reacher shared-stop report.

  python3 scripts/reacher_fixed_report.py --snaph snaph.txt --deployed deployed.txt --summary reacher_summary.txt

  snaph.txt / deployed.txt : from scripts/collect_ckpt_select.sh (driver '[ckpt-select]' lines; val = held10 mean
                             over the validation draws; the step comes from the snapshot FILENAME)
  reacher_summary.txt      : '<tag> ' + results/summary.csv lines (ckptval_read.yaml '=== SUMMARY <tag>' sections);
                             the DEPLOYED winner's report evals are '<row>_s<seed>_final_e<42|43|44>,latched10=,latched05='
Per cell (tags merged over -s<digits>- suffixes): the (teacher row, actor step) pair with the highest mean
validation held10 over all six seeds (ties -> smallest teacher budget, then smallest step); a pair must be
evaluated on every seed. Test (latched, draws 42-44) is reported per seed only where the deployed snapshot IS
that pair; otherwise the seed needs a re-eval (REPORT_ONLY import).
"""
import sys, re, collections, statistics
a = sys.argv[1:]
def opt(k): i = a.index(k); return a[i + 1]
snaph_f, dep_f, sum_f = opt("--snaph"), opt("--deployed"), opt("--summary")
def cell(tag): return re.sub(r"-s\d+(?=-)", "", tag)
def tkey(row): m = re.search(r"_t(\d+)$", row); return int(m.group(1)) if m else 10**9
def skey(step): return 10**9 if step == "final" else int(step)
sel = collections.defaultdict(lambda: collections.defaultdict(dict))   # cell -> (row, step) -> seed -> val
for line in open(snaph_f):
    p = line.split()
    if len(p) >= 6 and p[0] == "SNAPH" and p[1].startswith("rlp-re-") and p[5] != "na":
        sel[cell(p[1])][(p[2], p[4])][int(p[3])] = float(p[5])
dep = {}
for line in open(dep_f):
    p = line.split()
    if len(p) >= 5 and p[0].startswith("rlp-re-"): dep[(cell(p[0]), p[1], int(p[2]))] = p[4]
test = collections.defaultdict(dict)   # (cell,row,seed) -> draw -> (l10, l05)
pat = re.compile(r"^(\S+) (rs_x0_r0(?:_t\d+)?)_s(\d+)_final_e(4[234]),latched10=([0-9.]+),latched05=([0-9.]+)")
for line in open(sum_f):
    m = pat.match(line.strip())
    if m:
        tag, row, seed, draw, l10, l05 = m.groups()
        test[(cell(tag), row, int(seed))][int(draw)] = (float(l10), float(l05))
for c in sorted(sel):
    pairs = sel[c]; seeds_all = sorted({s for d in pairs.values() for s in d})
    per_seed = {s: sum(1 for p in pairs if s in pairs[p]) for s in seeds_all}
    best = None
    for (row, step), d in pairs.items():
        if len(d) < len(seeds_all): continue
        mv = statistics.mean(d.values()); k = (-mv, tkey(row), skey(step))
        if best is None or k < best[0]: best = (k, row, step, mv)
    if best is None: print(f"{c}: no pair evaluated on all seeds"); continue
    _, row, step, mv = best
    final = len(seeds_all) == 6 and all(v >= 36 for v in per_seed.values())
    print(f"{c}: {len(seeds_all)}/6 seeds -> FIXED teacher {'18000' if tkey(row) == 10**9 else tkey(row)} / actor {step}  (mean val held10 {mv:.2f})"
          + ("" if final else f"  [PRELIMINARY: pairs/36 per seed {per_seed}]"))
    t10, t05 = [], []
    for s in seeds_all:
        d = dep.get((c, row, s)); tr = test.get((c, row, s), {})
        if d == step and len(tr) == 3:
            m10 = statistics.mean(v[0] for v in tr.values()); m05 = statistics.mean(v[1] for v in tr.values())
            t10.append(m10); t05.append(m05)
            print(f"    s{s}: val {pairs[(row, step)][s]:.1f}; deployed == pair -> test latched tau.1 {m10:.1f} | tau.05 {m05:.1f}  (draws " + ", ".join(f"{k}:{v[0]:.0f}/{v[1]:.0f}" for k, v in sorted(tr.items())) + ")")
        else:
            print(f"    s{s}: val {pairs[(row, step)][s]:.1f}; deployed {d} != pair {step} (or report missing) -> RE-EVAL {row} s{s} step {step} on 42-44")
    if len(t05) == 6:
        print(f"    ==> n=6 test: tau.05 median {statistics.median(t05):.1f} mean {statistics.mean(t05):.1f} | tau.1 median {statistics.median(t10):.1f} mean {statistics.mean(t10):.1f}" + ("" if final else "   [PRELIMINARY]"))
