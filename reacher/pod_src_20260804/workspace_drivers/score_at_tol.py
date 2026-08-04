"""Re-score every arm at a different held-at-end tolerance, from existing logs.

`[threshold-sweep] full-arm held-at-end: 0.015rad A | 0.025rad B | 0.05rad C |
0.1rad D | 0.2rad E | median worst-joint M rad` is printed by every eval run, so
switching the tolerance costs nothing -- no GPU, no re-runs.

Context for why 0.1 is worth looking at: the campaign's 0.05 rad is the authors'
own number (their custom_tasks/reacher.py) but its provenance is a suspected
units copy from dm_control's _BIG_TARGET = .05, which is a geom radius in METRES
for a finger-to-target test. 0.05 also sits on the steepest part of the
sensitivity curve, so small differences in terminal precision swing the score
wildly; 0.1 is flatter and separates arms by whether they arrive at all.
"""

import re
from pathlib import Path

L = Path("/workspace/logs/waveA")
TOLS = ["0.015", "0.025", "0.05", "0.1", "0.2"]
RE_SWEEP = re.compile(
    r"\[threshold-sweep\] full-arm held-at-end: "
    r"0.015rad ([\d.]+) \| 0.025rad ([\d.]+) \| 0.05rad ([\d.]+) \| "
    r"0.1rad ([\d.]+) \| 0.2rad ([\d.]+) \| median worst-joint ([\d.]+)")
RE_EVER = re.compile(r"ever-in-ball ([\d.]+)")


def cell(path):
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    m = RE_SWEEP.search(txt)
    e = RE_EVER.search(txt)
    if not m:
        return None
    return [float(m.group(i)) for i in range(1, 6)] + [float(m.group(6))], (
        float(e.group(1)) if e else None)


def card(paths):
    got = [cell(p) for p in paths]
    got = [g for g in got if g]
    if not got:
        return None
    n = len(got)
    tols = [sum(g[0][i] for g in got) / n for i in range(5)]
    med = sum(g[0][5] for g in got) / n
    ever = [g[1] for g in got if g[1] is not None]
    return tols, med, (sum(ever) / len(ever) if ever else None), n


ARMS = [("Latent+CEM  (window, unlearned)", [L / f"ref_l2_s{s}.log" for s in range(42, 48)]),
        ("TD+CEM      (window, learned)", [L / f"ref_td_s{s}.log" for s in range(42, 48)])]
for tag in ["centre", "amax18", "amax20", "amax26", "mw03", "mw05", "alr1e4", "alr1e3", "steps16k"]:
    for s in (0, 1, 2):
        ARMS.append((f"LIP {tag} s{s}",
                     [L / f"wa_{tag}_s{s}_e{e}.log" for e in range(42, 48)]))

print(f"{'arm':<34} {'0.015':>7} {'0.025':>7} {'0.05':>7} {'0.10':>7} {'0.20':>7} "
      f"{'medErr':>7} {'ever':>6} {'n':>3}")
print("-" * 96)
lip_by_tag = {}
for name, paths in ARMS:
    c = card(paths)
    if not c:
        continue
    tols, med, ever, n = c
    ev = f"{ever:6.1f}" if ever is not None else "     -"
    print(f"{name:<34} {tols[0]:7.1f} {tols[1]:7.1f} {tols[2]:7.1f} {tols[3]:7.1f} "
          f"{tols[4]:7.1f} {med:7.4f} {ev} {n:3d}")
    if name.startswith("LIP "):
        lip_by_tag.setdefault(name.split()[1], []).append(tols)

print("\nLIP pooled over training seeds (per config):")
print(f"{'config':<12} {'0.05':>7} {'0.10':>7} {'0.20':>7} {'seeds':>6}")
for tag, lst in lip_by_tag.items():
    k = len(lst)
    print(f"{tag:<12} {sum(x[2] for x in lst)/k:7.1f} {sum(x[3] for x in lst)/k:7.1f} "
          f"{sum(x[4] for x in lst)/k:7.1f} {k:6d}")
