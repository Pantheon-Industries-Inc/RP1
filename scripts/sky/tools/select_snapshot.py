#!/usr/bin/env python3
"""Pick the deployed optimizer snapshot for an L2O / DMPO cell on the VALIDATION draws.

Mirrors the rule the RLP rows use for their (teacher, actor) pair: ONE shared step for the whole
cell, chosen by the mean validation score over the training seeds, ties broken toward the EARLIER
step (the final iterate is treated as the latest). Per-seed selection would be a different, more
permissive protocol than RLP's and is deliberately not offered.

  select_snapshot.py <result-dir> "<train seeds>" "<val draws>" <horizon>
prints the winning step token ("final" or an integer) on stdout; the full table goes to stderr.
"""
import collections
import glob
import os
import re
import statistics
import sys

d, seeds, draws, vh = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4]
per = collections.defaultdict(list)
pat = re.compile(r"results_val_a(\d+)_step(\w+)_h%s_s(\d+)\.txt$" % re.escape(vh))
for f in sorted(glob.glob(os.path.join(d, "results_val_a*_step*_h%s_s*.txt" % vh))):
    m = pat.search(os.path.basename(f))
    if not m:
        continue
    try:
        v = float(re.search(r"'success_rate': ([0-9.]+)", open(f).read()).group(1))
    except Exception:
        continue
    per[m.group(2)].append(v)

need = len(seeds) * len(draws)
full = {k: statistics.mean(v) for k, v in per.items() if len(v) == need}
partial = {k: len(v) for k, v in per.items() if len(v) != need}
for k, n in sorted(partial.items()):
    print(f"[select] step {k}: only {n}/{need} validation cells -- ignored", file=sys.stderr)
if not full:
    print("[select] no snapshot has a complete validation grid; deploying the final iterate", file=sys.stderr)
    print("final")
    sys.exit(0)


def order_key(k: str) -> int:
    return 10**9 if k == "final" else int(k)


for k in sorted(full, key=order_key):
    print(f"[select] step {k:<7} val mean {full[k]:6.2f} (n={len(per[k])})", file=sys.stderr)
best = min(full, key=lambda k: (-full[k], order_key(k)))
print(f"[select] winner: {best} (val mean {full[best]:.2f})", file=sys.stderr)
print(best)
