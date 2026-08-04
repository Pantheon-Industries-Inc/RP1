"""Paired comparison: early-stopped (selected on seeds 50/51) vs trained-to-8000,
both scored on the untouched reporting seeds 42-47.
"""

import re
from pathlib import Path

rows = {}
for line in Path("/workspace/results/summary_earlystop_lejepa.csv").read_text().splitlines():
    rows[line.split(",", 1)[0]] = line

REP = list(range(42, 48))


def card(pre, key="held"):
    v = []
    for s in REP:
        ln = rows.get(f"{pre}{s}")
        if ln:
            m = re.search(rf"{key}=([0-9.]+)", ln)
            if m:
                v.append(float(m.group(1)))
    return (sum(v) / len(v) if v else None), len(v)


hdr = "%-9s %-5s %14s %11s %8s" % ("config", "seed", "early-stopped", "full 8000", "delta")
print(hdr)
print("-" * len(hdr))
ts = tf = 0.0
k = 0
for tag in ("base", "lr1e4", "gearly"):
    for s in range(3):
        a, na = card(f"es_{tag}_s{s}_SEL_e")
        b, nb = card(f"es_{tag}_s{s}_FIN_e")
        if a is None or b is None:
            print("%-9s s%-4d %14s %11s %8s" % (tag, s, "-" if a is None else round(a, 1),
                                                "-" if b is None else round(b, 1), "-"))
            continue
        print("%-9s s%-4d %14.1f %11.1f %+8.1f" % (tag, s, a, b, a - b))
        ts += a
        tf += b
        k += 1
if k:
    print("-" * len(hdr))
    print("%-9s %-5s %14.1f %11.1f %+8.1f" % ("POOLED", "", ts / k, tf / k, ts / k - tf / k))
    print("\nbar: Latent+CEM-window 44.7 @0.05")
