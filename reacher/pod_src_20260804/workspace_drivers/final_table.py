"""Rebuild the reacher method table from the per-cell CSVs, not from notes."""
import glob
import re

REP = [42, 43, 44, 45, 46, 47]


def rows():
    out = {}
    for f in glob.glob("/workspace/results/summary_*.csv"):
        for ln in open(f):
            m = re.match(r"^(\S+?),held=([0-9.]+|FAIL),held10=([0-9.]+|FAIL)", ln.strip())
            if m:
                out[m.group(1)] = (m.group(2), m.group(3))
    return out


R = rows()


def agg(pats):
    """Mean over every row whose name matches any regex, split by metric."""
    h, t = [], []
    for k, (a, b) in R.items():
        if any(re.match(p + r"$", k) for p in pats):
            if a != "FAIL":
                h.append(float(a))
            if b != "FAIL":
                t.append(float(b))
    return (sum(t) / len(t) if t else None, sum(h) / len(h) if h else None, len(t))


SEEDS = "|".join(str(s) for s in REP)
ARMS = [
    ("LIP (learned value)", "~16",
     {"lejepa": [rf"lk6_lejepa_s\d_e({SEEDS})"], "pldm": [rf"lk6_pldm_s\d_e({SEEDS})"]}),
    ("Latent + CEM", "3000",
     {"lejepa": [rf"bl_lejepa_l2cem_s({SEEDS})"], "pldm": [rf"bl_pldm_l2cem_s({SEEDS})"]}),
    ("Latent + Adam", "3000",
     {"lejepa": [rf"bl_lejepa_l2adam_s({SEEDS})"], "pldm": [rf"bl_pldm_l2adam_s({SEEDS})"]}),
    ("Value + CEM", "3000",
     {"lejepa": [rf"tdg98_lejepa_s({SEEDS})"], "pldm": [rf"tdg98_pldm_s({SEEDS})"]}),
    ("Value + Adam", "3000",
     {"lejepa": [rf"va_lejepa_s({SEEDS})"], "pldm": [rf"va_pldm_s({SEEDS})"]}),
    ("PWM (terminal, budget-matched)", "0",
     {"lejepa": [rf"abl_lejepa_s1000_terminal_s\d_e({SEEDS})"],
      "pldm": [rf"abl_pldm_s1000_terminal_s\d_e({SEEDS})"]}),
    ("PWM (dense) [ablation]", "0",
     {"lejepa": [rf"abl_lejepa_s1000_dense_s\d_e({SEEDS})"],
      "pldm": [rf"abl_pldm_s1000_dense_s\d_e({SEEDS})"]}),
]

print("%-34s %6s | %-18s | %-18s" % ("method", "roll", "LeWM  @0.1 / @0.05", "PLDM  @0.1 / @0.05"))
print("-" * 84)
for name, roll, pats in ARMS:
    cells = []
    for b in ("lejepa", "pldm"):
        t, h, n = agg(pats[b])
        cells.append("  --  " if t is None else "%5.1f / %5.1f  (n=%d)" % (t, h, n))
    print("%-34s %6s | %-18s | %-18s" % (name, roll, cells[0], cells[1]))
