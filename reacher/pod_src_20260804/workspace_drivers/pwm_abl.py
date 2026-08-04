"""PWM ablation: steps x objective, against the 8000/dense card already taken."""
import re, collections
rows = {}
for ln in open("/workspace/results/summary_pwmabl.csv"):
    m = re.match(r"^(abl_\S+?),held=([0-9.]+|FAIL),held10=([0-9.]+|FAIL)", ln.strip())
    if m:
        rows[m.group(1)] = (m.group(2), m.group(3))
agg = collections.defaultdict(lambda: [[], []])
for k, (h, t) in rows.items():
    m = re.match(r"^abl_(lejepa|pldm)_s(\d+)_(dense|terminal)_s(\d)_e(\d+)$", k)
    if not m:
        continue
    key = (m.group(1), m.group(3), m.group(2))
    if h != "FAIL":
        agg[key][0].append(float(h))
    if t != "FAIL":
        agg[key][1].append(float(t))
CARDED = {("lejepa", "dense", "8000"): (85.9, 49.4), ("pldm", "dense", "8000"): (99.0, 71.5)}
print("%-8s %-9s %-6s %7s %7s %6s" % ("base", "objective", "steps", "@0.1", "@0.05", "n"))
print("-" * 48)
for B in ["lejepa", "pldm"]:
    for obj in ["dense", "terminal"]:
        for st in ["1000", "8000"]:
            key = (B, obj, st)
            if key in CARDED:
                a, c = CARDED[key]
                print("%-8s %-9s %-6s %7.1f %7.1f %6s" % (B, obj, st, a, c, "18/18"))
                continue
            v = agg.get(key)
            if not v or not v[1]:
                print("%-8s %-9s %-6s %7s %7s %6s" % (B, obj, st, "-", "-", "0"))
                continue
            print("%-8s %-9s %-6s %7.1f %7.1f %6s" % (
                B, obj, st, sum(v[1]) / len(v[1]), sum(v[0]) / len(v[0]), "%d/18" % len(v[1])))
    print()
print("references: LIP@1000  lejepa 98.2  pldm 94.2   |   Latent+CEM bar  84.3 / 78.3")
