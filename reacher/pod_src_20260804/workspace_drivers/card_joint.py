"""Recard the joint sweep winners with a correct held10 parse.

run_joint_sweep.sh's meanof() did  grep -oE "held10=[0-9.]+" | grep -oE "[0-9.]+"
and the second grep returns the "10" from the KEY before reaching the value, so
every @0.1 figure it printed was a constant 10.0. The per-seed rows in the CSV
are fine, so the cards are recomputed here from those.
"""
import re

rows = {}
for ln in open("/workspace/results/summary_joint.csv"):
    m = re.match(r"^(\S+?),held=([0-9.]+|FAIL),held10=([0-9.]+|FAIL)", ln.strip())
    if m:
        rows[m.group(1)] = (m.group(2), m.group(3))

REP = [42, 43, 44, 45, 46, 47]
BAR = {"lejepa": 44.7, "pldm": 39.3}
TAG = {"lejepa": "lejepa_e005a22l3e4r05", "pldm": "pldm_e005a18l3e4r05"}

print("base     seed    HELD@0.05     @0.1     n")
print("-" * 46)
for B in ["lejepa", "pldm"]:
    allh, allt = [], []
    for sd in [0, 1, 2]:
        h, t = [], []
        for s in REP:
            k = "js_%s_s%d_e%d" % (TAG[B], sd, s)
            if k in rows:
                a, b = rows[k]
                if a != "FAIL":
                    h.append(float(a))
                if b != "FAIL":
                    t.append(float(b))
        if h:
            mt = sum(t) / len(t) if t else 0.0
            print("%-8s s%-6d %8.1f %8.1f %5d" % (B, sd, sum(h) / len(h), mt, len(h)))
            allh += h
            allt += t
    if allh:
        mh = sum(allh) / len(allh)
        mt = sum(allt) / len(allt) if allt else 0.0
        print("%-8s %-7s %8.1f %8.1f    bar %.1f   margin %+.1f" % (
            B, "POOLED", mh, mt, BAR[B], mh - BAR[B]))
    print()

# min-margin check: would the "both above their own bar" criterion have picked
# the same shared (expectile, replay) pair the cross-base mean did?
print("shared-pair check, screen values vs each base's own bar")
print("pair                 lejepa   pldm    min-margin")
print("-" * 52)
scr = [ln.split() for ln in open("/workspace/results/joint_screen.txt")]
for ex in ["005", "01"]:
    for rp in ["0", "0.5"]:
        best = {}
        for r in scr:
            if len(r) >= 7 and r[2] == ex and r[5] == rp:
                b = r[1]
                v = float(r[0])
                if v > best.get(b, -1e9):
                    best[b] = v
        if len(best) < 2:
            continue
        mm = min(best["lejepa"] - BAR["lejepa"], best["pldm"] - BAR["pldm"])
        print("expectile%-4s replay%-5s %6.1f %6.1f %11.1f" % (
            ex, rp, best["lejepa"], best["pldm"], mm))
