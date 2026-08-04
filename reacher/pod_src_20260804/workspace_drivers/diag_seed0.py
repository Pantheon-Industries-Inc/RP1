"""Why does training seed 0 deploy so badly?

Observation that prompted this: across wave-A configs, seed 0 reaches the goal
ball nearly as often as seeds 1/2 (ever-in-ball 72-82 vs 90-100) but HOLDS far
less (4-16 vs 46-50), and its median terminal error is ~1.7x worse
(0.0966 vs 0.0576 rad) -- while its TRAINING objective is the BEST of the three
(E_final 0.906 vs 1.020 / 1.524).

Hypothesis: the actor is exploiting world-model error. A lower imagined cost is
being bought by finding plans the WM is optimistic about, which do not transfer.
If so, E_final should ANTI-correlate with deployed held across all 27 actors --
the opposite of what a well-specified objective would do.

This settles it with the data already on disk: E_final from each training log,
held from the eval CSV. No GPU needed.
"""

import re
from pathlib import Path

SUM = Path("/workspace/results/summary_waveA_lejepa.csv")
LOGS = Path("/workspace/logs/waveA")
TAGS = ["centre", "amax18", "amax20", "amax26", "mw03", "mw05",
        "alr1e4", "alr1e3", "steps16k"]

rows = {}
for line in SUM.read_text().splitlines():
    rows[line.split(",", 1)[0]] = line


def held(tag, seed):
    v = []
    for e in range(42, 48):
        ln = rows.get(f"wa_{tag}_s{seed}_e{e}")
        if ln:
            m = re.search(r"held=([0-9.]+)", ln)
            if m:
                v.append(float(m.group(1)))
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def train_stats(tag, seed):
    f = LOGS / f"train_{tag}_s{seed}.log"
    if not f.exists():
        return None
    ef = efirst = None
    for line in f.read_text().splitlines():
        m = re.match(r"step (\d+): E_final ([\d.]+) E_first ([\d.]+)", line)
        if m:
            ef, efirst = float(m.group(2)), float(m.group(3))
    return (ef, efirst)


pairs = []
print(f"{'config':<10} {'seed':<5} {'E_final':>8} {'E_first':>8} {'gain':>7} {'HELD':>7} {'n':>3}")
for tag in TAGS:
    for s in (0, 1, 2):
        ts = train_stats(tag, s)
        h, n = held(tag, s)
        if ts is None or ts[0] is None or h is None or n < 6:
            continue
        ef, efirst = ts
        gain = efirst - ef                      # how much the actor reduced the imagined cost
        pairs.append((ef, gain, h, tag, s))
        print(f"{tag:<10} s{s:<4} {ef:8.3f} {efirst:8.3f} {gain:7.3f} {h:7.1f} {n:3d}")

if len(pairs) >= 4:
    def corr(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs) ** 0.5
        syy = sum((y - my) ** 2 for y in ys) ** 0.5
        return sxy / (sxx * syy) if sxx and syy else float("nan")

    ef = [p[0] for p in pairs]
    gn = [p[1] for p in pairs]
    hd = [p[2] for p in pairs]
    print(f"\nn = {len(pairs)} actors with complete 6-seed cards")
    print(f"  corr(E_final,  HELD) = {corr(ef, hd):+.3f}   "
          f"(NEGATIVE would mean: lower imagined cost -> better real; "
          f"POSITIVE means the objective ANTI-predicts success)")
    print(f"  corr(imagined gain, HELD) = {corr(gn, hd):+.3f}   "
          f"(POSITIVE would mean: optimising harder helps; NEGATIVE = exploitation)")
    best = min(pairs, key=lambda p: p[0])
    worst = max(pairs, key=lambda p: p[0])
    print(f"\n  best imagined  : {best[3]} s{best[4]}  E_final {best[0]:.3f} -> HELD {best[2]:.1f}")
    print(f"  worst imagined : {worst[3]} s{worst[4]} E_final {worst[0]:.3f} -> HELD {worst[2]:.1f}")
