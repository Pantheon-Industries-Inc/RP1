"""LeWM (lejepa) comparison across every arm measured on the rebuilt H200 pod.

Scores each arm at BOTH tolerances by re-parsing the `[threshold-sweep]` line
from the eval logs (the driver's own @0.1 column had a bad regex group and
printed a constant 10.0 -- ignore that; these numbers come from the logs).
"""

import re
from pathlib import Path

RE_SWEEP = re.compile(
    r"\[threshold-sweep\] full-arm held-at-end: "
    r"0.015rad ([\d.]+) \| 0.025rad ([\d.]+) \| 0.05rad ([\d.]+) \| "
    r"0.1rad ([\d.]+) \| 0.2rad ([\d.]+) \| median worst-joint ([\d.]+)")
RE_EVER = re.compile(r"ever-in-ball ([\d.]+)")
REP = list(range(42, 48))


def card(paths):
    h05, h10, med, ever = [], [], [], []
    for p in paths:
        if not p.exists():
            continue
        t = p.read_text(errors="ignore")
        m, e = RE_SWEEP.search(t), RE_EVER.search(t)
        if not m:
            continue
        h05.append(float(m.group(3)))
        h10.append(float(m.group(4)))
        med.append(float(m.group(6)))
        if e:
            ever.append(float(e.group(1)))
    if not h05:
        return None
    n = len(h05)
    return (sum(h05) / n, sum(h10) / n, sum(med) / n,
            (sum(ever) / len(ever) if ever else float("nan")), n)


WA = Path("/workspace/logs/waveA")
ES = Path("/workspace/logs/earlystop")
GC = Path("/workspace/logs/gridC")

ARMS = []
ARMS.append(("Latent+CEM  · window  (paper's method)",
             [WA / f"ref_l2_s{s}.log" for s in REP]))
ARMS.append(("TD+CEM      · window  (learned metric)",
             [WA / f"ref_td_s{s}.log" for s in REP]))

# LIP: wave A centre (8k steps, canonical lr)
for s in range(3):
    ARMS.append((f"LIP · window  centre 8k        s{s}",
                 [WA / f"wa_centre_s{s}_e{e}.log" for e in REP]))
# LIP: grid C winner
for s in range(3):
    ARMS.append((f"LIP · window  geom-early 4k lr1e-4 s{s}",
                 [GC / f"gc_gest4k1e4_s{s}_e{e}.log" for e in REP]))
# LIP: early-stopped
for tag, lbl in (("base", "base"), ("lr1e4", "lr1e-4"), ("gearly", "geom-early")):
    for s in range(3):
        ARMS.append((f"LIP · window  EARLY-STOP {lbl:<10} s{s}",
                     [ES / f"es_{tag}_s{s}_SEL_e{e}.log" for e in REP]))

print("%-42s %7s %7s %9s %7s %4s" % ("arm (lejepa, h25, held-at-end)",
                                     "@0.05", "@0.10", "medErr", "ever", "n"))
print("-" * 82)
groups = {}
for name, paths in ARMS:
    c = card(paths)
    if not c:
        continue
    h05, h10, med, ever, n = c
    print("%-42s %7.1f %7.1f %9.4f %7.1f %4d" % (name, h05, h10, med, ever, n))
    key = name.rsplit("s", 1)[0].strip()
    groups.setdefault(key, []).append((h05, h10))

print("\n%-42s %7s %7s %6s" % ("POOLED over training seeds", "@0.05", "@0.10", "seeds"))
print("-" * 66)
for k, v in groups.items():
    if len(v) < 2:
        continue
    n = len(v)
    print("%-42s %7.1f %7.1f %6d" % (k, sum(x[0] for x in v) / n,
                                     sum(x[1] for x in v) / n, n))
