#!/usr/bin/env python3
"""Joint teacher x actor early stopping, resolved from W&B.

For every (env, base, tag, seed) the yaml already picked the best ACTOR snapshot within each
teacher row on the selection draws (eval/ckv5/*). This script picks the TEACHER row per seed
with the same criterion -- the row whose deployed actor has the highest selection score
(eval/ckv5/mean_rh5_all: the mean over the horizons the job evaluates, i.e. h25+h100 for the
h100-stack jobs and h25 alone for the -h25 jobs) -- ties broken toward the SMALLEST teacher
budget (rule fixed 2026-09-22 before report data was inspected). It then prints the selected
row's REPORT numbers (eval/rh5/*, draws 42-44) and the per-cell median/mean over seeds.

    python3 scripts/wandb_select_joint.py 'rlp-tw-uniJ(-p)?-20260921'
"""
import collections, re, statistics, sys
import wandb

pat = re.compile(sys.argv[1])
# --ckpt-log FILE: "<tag> <row> <seed> <val>" lines from scripts/collect_ckpt_select.sh = the DEPLOYED
# snapshot's selection score (the W&B summary only keeps the LAST evaluated snapshot's score unless
# the job wrote eval/ckv5/best_all).
override = {}
if "--ckpt-log" in sys.argv:
    for line in open(sys.argv[sys.argv.index("--ckpt-log") + 1]):
        parts = line.split()
        if len(parts) in (4, 5) and parts[0] != "SNAP":
            override[(parts[0], parts[1], int(parts[2]))] = float(parts[3])   # 5th field (deployed step) ignored here
api = wandb.Api(timeout=180)
rows = collections.defaultdict(dict)   # (env, base, tag, seed) -> teacher_steps -> record
for r in api.runs("armin-sommer/RLP", order="-created_at", per_page=500):
    m = re.match(r"(?P<tag>.+)_g_(?P<env>tworoom|cube)_(?P<base>lejepa|lewm|pldm)_unig_ctrl_a2\.5(?:_t(?P<t>\d+))?_s(?P<seed>\d+)$", r.name)
    if not m or not pat.search(m["tag"]):
        continue
    s = r.summary
    if not any(k.startswith("eval/rh5/") for k in s.keys()):
        continue   # no report eval logged yet
    rep = {h: s.get(f"eval/rh5/mean_rh5_h{h}") for h in (25, 100)}
    t = int(m["t"]) if m["t"] else 18000
    row = "unig_ctrl_a2.5" + (f"_t{m['t']}" if m["t"] else "")
    val = s.get("eval/ckv5/best_all")                      # written by jobs launched after 2026-09-22
    if val is None:
        val = override.get((m["tag"], row, int(m["seed"])))   # from the job logs
    if val is None and "eval/ckv5/mean_rh5_all" in s:
        val = float(s["eval/ckv5/mean_rh5_all"]); src = "LAST-SNAPSHOT-ONLY"
    elif val is None:
        continue   # no selection score available for this row yet
    else:
        src = "deployed"
    rows[(m["env"], m["base"], m["tag"], int(m["seed"]))][t] = dict(val=float(val), rep=rep, run=r.name, src=src)

cells = collections.defaultdict(dict)
for (env, base, tag, seed), byt in sorted(rows.items()):
    best_t = max(byt, key=lambda t: (byt[t]["val"], -t))   # highest val, then smallest budget
    rec = byt[best_t]
    n_rows = len(byt)
    rep = rec["rep"]
    srcs = {v["src"] for v in byt.values()}
    print(f"{env}/{base} {tag} s{seed}: rows={n_rows}/6 -> teacher {best_t} (val {rec['val']:.2f}) "
          f"report h25={rep[25]} h100={rep[100]}" + ("" if n_rows == 6 else "  [INCOMPLETE]")
          + ("" if srcs == {"deployed"} else "  [val source: " + ",".join(sorted(srcs)) + "]"))
    cells[(env, base, tag)][seed] = (rep, n_rows == 6)
for (env, base, tag), by_seed in sorted(cells.items()):
    for h in (25, 100):
        vals = [rep[h] for rep, done in by_seed.values() if rep[h] is not None]
        if not vals:
            continue
        done = sum(1 for _, d in by_seed.values() if d)
        print(f"  {env}/{base} {tag} h{h}: n={len(vals)} (complete seeds {done}) median {statistics.median(vals):.1f} mean {sum(vals)/len(vals):.1f}")
