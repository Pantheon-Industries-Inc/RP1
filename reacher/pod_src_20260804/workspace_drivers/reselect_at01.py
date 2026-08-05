"""Re-run the joint sweep's selection with @0.1 rad as the objective.

The headline metric changed from held-at-end @0.05 to @0.1. Every selection the
sweep made -- the shared (expectile, replay) pair and each base's (amax, lr) --
was made by @0.05 on the selection seeds. The @0.1 numbers were recorded for the
same runs at the same time, so the whole selection can be redone for free.

Selection uses SELECTION seeds {50,51} only, exactly as the sweep did. Reporting
seeds {42..47} are used solely to card whatever selection picks, so this is not
picking a winner on the test seeds.

Bars to beat at @0.1 (reporting seeds, measured):  lejepa 84.3   pldm 78.3
"""
import re
from collections import defaultdict

SUM = "/workspace/results/summary_joint.csv"
BAR = {"lejepa": 84.3, "pldm": 78.3}
SEL = [50, 51]
REP = [42, 43, 44, 45, 46, 47]

rows = {}
for ln in open(SUM):
    m = re.match(r"^(\S+?),held=([0-9.]+|FAIL),held10=([0-9.]+|FAIL)", ln.strip())
    if m:
        rows[m.group(1)] = (m.group(2), m.group(3))

# js_<base>_e<EX>a<amax>l<lr>r<rp>_s<seed>_sel<s>
pat = re.compile(r"^js_(lejepa|pldm)_e(005|01)a(18|22)l(3e4|1e3)r(0|05)_s(\d)_sel(\d+)$")
cfg = defaultdict(list)          # (base,ex,rp,am,lr) -> [held10 over seeds x sel]
cfg05 = defaultdict(list)
for k, (h5, h10) in rows.items():
    m = pat.match(k)
    if not m:
        continue
    b, ex, am, lr, rp, sd, s = m.groups()
    if int(s) not in SEL:
        continue
    key = (b, ex, rp, am, lr)
    if h10 != "FAIL":
        cfg[key].append(float(h10))
    if h5 != "FAIL":
        cfg05[key].append(float(h5))

mean = lambda v: sum(v) / len(v) if v else 0.0

print("=" * 74)
print("SHARED PAIR SELECTION, scored at @0.1 (selection seeds, per-base best)")
print("=" * 74)
print("pair                      lejepa   pldm   cross-base   min-margin")
best_pair, best_mm = None, -1e9
for ex in ["005", "01"]:
    for rp in ["0", "05"]:
        per = {}
        for b in ["lejepa", "pldm"]:
            cands = [(mean(v), k) for k, v in cfg.items()
                     if k[0] == b and k[1] == ex and k[2] == rp and v]
            if cands:
                per[b] = max(cands)
        if len(per) < 2:
            continue
        xb = (per["lejepa"][0] + per["pldm"][0]) / 2
        mm = min(per["lejepa"][0] - BAR["lejepa"], per["pldm"][0] - BAR["pldm"])
        star = ""
        if mm > best_mm:
            best_mm, best_pair, star = mm, (ex, rp), "  <-"
        print("expectile %-4s replay %-4s %7.1f %6.1f %10.1f %11.1f%s"
              % (ex, rp, per["lejepa"][0], per["pldm"][0], xb, mm, star))

print()
print("chosen at @0.1 : expectile %s, replay %s" % best_pair)
print("chosen at @0.05: expectile 005, replay 0.5   (what the sweep used)")
print()

ex, rp = best_pair
print("=" * 74)
print("PER-BASE WINNER inside that pair, at @0.1")
print("=" * 74)
winners = {}
for b in ["lejepa", "pldm"]:
    cands = sorted([(mean(v), k) for k, v in cfg.items()
                    if k[0] == b and k[1] == ex and k[2] == rp and v], reverse=True)
    for sc, k in cands:
        print("  %-7s amax %-3s lr %-4s  sel@0.1 %5.1f   sel@0.05 %5.1f"
              % (b, k[3], k[4], sc, mean(cfg05[k])))
    winners[b] = cands[0][1]
    print("  -> %s winner: amax %s lr %s\n" % (b, cands[0][1][3], cands[0][1][4]))

print("=" * 74)
print("CARD on reporting seeds {42..47}")
print("=" * 74)
for b in ["lejepa", "pldm"]:
    k = winners[b]
    am = {"18": "18", "22": "22"}[k[3]]
    tag = "%s_e%sa%sl%sr%s" % (b, k[1], am, k[4], k[2])
    allh, allt = [], []
    for sd in [0, 1, 2]:
        h, t = [], []
        for s in REP:
            kk = "js_%s_s%d_e%d" % (tag, sd, s)
            if kk in rows:
                a, c = rows[kk]
                if a != "FAIL":
                    h.append(float(a))
                if c != "FAIL":
                    t.append(float(c))
        if h:
            print("  %-7s s%d   @0.05 %5.1f   @0.1 %5.1f   (n=%d)"
                  % (b, sd, mean(h), mean(t), len(h)))
            allh += h
            allt += t
    if allt:
        print("  %-7s POOLED @0.05 %5.1f   @0.1 %5.1f   bar %.1f   margin %+.1f"
              % (b, mean(allh), mean(allt), BAR[b], mean(allt) - BAR[b]))
    else:
        print("  %-7s NOT CARDED on reporting seeds (only screened)" % b)
    print()
