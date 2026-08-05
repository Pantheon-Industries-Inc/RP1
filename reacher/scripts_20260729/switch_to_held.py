"""Switch the primary metric to HELD-AT-END (user directive 2026-07-29).

Held = every joint within 0.05 rad AT THE FINAL STEP (arrive AND stay).
Latched = within 0.05 rad at ANY step in the budget (the authors' convention,
kept as the paper-comparability column).

Both are printed by every run's `[success-convention]` line and both are already
stored in the summary CSVs, so no eval re-runs are needed. What DOES need fixing
is selection: the amax sweep and the LIP-boost sweep were written to record and
rank by latched, and the amax winner feeds the gated Dyna recipe. Patch them
before they start doing work.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------- 1. patch drivers
PATCHES = {
    "/workspace/run_amax_sweep.sh": [
        # record both metrics
        ('  local ever\n'
         '  ever=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")\n'
         '  echo "${nm},latched=${ever:-FAIL}" >> "$SUM"\n'
         '  log "  ${nm}: latched ${ever:-FAIL}"',
         '  local ever held\n'
         '  ever=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")\n'
         '  held=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")\n'
         '  echo "${nm},held=${held:-FAIL},latched=${ever:-FAIL}" >> "$SUM"\n'
         '  log "  ${nm}: HELD ${held:-FAIL} | latched ${ever:-FAIL}"'),
        # rank by held
        ('    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "latched=[0-9.]+" | grep -oE "[0-9.]+")',
         '    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")'),
        ('    e=$(grep "^${pre}${s}," "$SSUM" 2>/dev/null | tail -1 | grep -oE "latched=[0-9.]+" | grep -oE "[0-9.]+")',
         '    e=$(grep "^${pre}${s}," "$SSUM" 2>/dev/null | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")'),
        ('latched h25 $(mean6 "amx_${tag}_s")', 'HELD h25 $(mean6 "amx_${tag}_s")'),
    ],
    "/workspace/run_lipboost.sh": [
        ('  local ever\n'
         '  ever=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")\n'
         '  echo "${nm},latched=${ever:-FAIL}" >> "$SUM"\n'
         '  log "  ${nm}: latched ${ever:-FAIL}"',
         '  local ever held\n'
         '  ever=$(grep -oE "ever-in-ball [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")\n'
         '  held=$(grep -oE "HELD-at-end [0-9.]+" "$L/${nm}.log" | tail -1 | grep -oE "[0-9.]+")\n'
         '  echo "${nm},held=${held:-FAIL},latched=${ever:-FAIL}" >> "$SUM"\n'
         '  log "  ${nm}: HELD ${held:-FAIL} | latched ${ever:-FAIL}"'),
        ('    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "latched=[0-9.]+" | grep -oE "[0-9.]+")',
         '    e=$(grep "^${pre}${s}," "$SUM" | tail -1 | grep -oE "held=[0-9.]+" | grep -oE "[0-9.]+")'),
        ('latched h25 $(mean6 "bst_${tag}_s${seed}_e")', 'HELD h25 $(mean6 "bst_${tag}_s${seed}_e")'),
        ('[ref amax1.8 88.3 | TARGET > Latent+CEM 89.7]',
         '[HELD refs: LIP-w pooled ~37 | Latent+CEM-w ~44 -- TARGET: beat Latent+CEM on HELD]'),
    ],
}
for path, subs in PATCHES.items():
    p = Path(path)
    if not p.exists():
        print(f"  {p.name}: MISSING, skipped")
        continue
    s = p.read_text()
    if "HELD ${held" in s:
        print(f"  {p.name}: already switched")
        continue
    n = 0
    for old, new in subs:
        if old in s:
            s = s.replace(old, new)
            n += 1
    p.write_text(s)
    print(f"  {p.name}: {n}/{len(subs)} substitutions applied")

# ---------------------------------------------------------------- 2. recompute cards
SEEDS = [42, 43, 44, 45, 46, 47]
GROUPS = [
    ("h25  LIP window (plain)", "/workspace/results/summary_w3lip_lejepa.csv",
     [("s0", "w3lip0_lejepa_s"), ("s1", "w3lip1_lejepa_s"), ("s2", "w3lip2_lejepa_s")]),
    ("h25  CEM window arms", "/workspace/results/summary_window_lejepa.csv",
     [("Latent+CEM-w", "w3l2_tdcem_lejepa_s"), ("TD+CEM-w", "w3_tdcem_lejepa_s")]),
    ("h25  terminal-cost refs", "/workspace/results/summary_anchor10_lejepa.csv",
     [("Latent+CEM terminal", "a10_lejepa_h25_s")]),
    ("h25  hyper sweep (single train seed)", "/workspace/results/summary_w3sweep_lejepa.csv",
     [(t, f"swp_{t}_s") for t in ["amax18", "amax20", "it6", "it10", "alr1e4", "alr1e3",
                                  "clr3e4", "clr3e3", "tau005", "tau02", "cmb"]]),
    ("h50  window arms", "/workspace/results/summary_h50w3_lejepa.csv",
     [("LIP s0", "h50w3lip0_s"), ("LIP s1", "h50w3lip1_s"), ("LIP s2", "h50w3lip2_s"),
      ("Latent+CEM-w", "h50w3l2_s"), ("TD+CEM-w", "h50w3td_s")]),
    ("h100 window arms", "/workspace/results/summary_h100w3_lejepa.csv",
     [("LIP s0", "h100w3lip0_s"), ("LIP s1", "h100w3lip1_s"), ("LIP s2", "h100w3lip2_s"),
      ("Latent+CEM-w", "h100w3l2_s"), ("TD+CEM-w", "h100w3td_s")]),
]


def card(csv, prefix):
    p = Path(csv)
    if not p.exists():
        return None, None, 0
    rows = {}
    for line in p.read_text().splitlines():
        nm = line.split(",", 1)[0]
        rows[nm] = line
    h, l = [], []
    for s in SEEDS:
        line = rows.get(f"{prefix}{s}")
        if not line:
            continue
        mh = re.search(r"held=([0-9.]+)", line)
        ml = re.search(r"latched=([0-9.]+)", line)
        if mh:
            h.append(float(mh.group(1)))
        if ml:
            l.append(float(ml.group(1)))
    if not h and not l:
        return None, None, 0
    return (sum(h) / len(h) if h else None), (sum(l) / len(l) if l else None), max(len(h), len(l))


print("\n" + "=" * 74)
print("ALL CARDS RE-SCORED — HELD-AT-END primary, latched in brackets")
print("=" * 74)
for title, csv, arms in GROUPS:
    print(f"\n{title}")
    out = []
    for label, pre in arms:
        hh, ll, n = card(csv, pre)
        if n == 0:
            continue
        out.append((hh if hh is not None else -1, label, hh, ll, n))
    for _, label, hh, ll, n in sorted(out, reverse=True):
        hs = f"{hh:5.1f}" if hh is not None else "    -"
        ls = f"{ll:5.1f}" if ll is not None else "    -"
        print(f"   {label:<24} HELD {hs}   [latched {ls}]  n={n}")
print("\nSWITCH_TO_HELD_DONE")
