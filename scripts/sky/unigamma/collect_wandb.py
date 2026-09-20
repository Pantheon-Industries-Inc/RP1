#!/usr/bin/env python3
"""Collect TwoRoom/Cube RLP results straight from W&B (no cluster job).

The tworoom_g98_rescue.yaml eval stage logs one W&B run per (tag, actor, seed)
with summary keys ``eval/rh5/rh5_h{25,100}_s{42,43,44}`` (report draws) and
``eval/ckv5/rh5_h{25,100}_s{48..51}`` (selection draws, informational only).
Reacher jobs (reacher_gamma.yaml) log NOTHING to W&B beyond config -- their
numbers live only in ``/checkpoints/<user>/<tag>/results/summary.csv`` on the
volume and need a collect job.

Protocol: per training seed = mean over report draws 42/43/44; cell = median
over seeds. Selection draws are never reported.

    python3 scripts/sky/unigamma/collect_wandb.py 'rlp-(tw|cu)-v2l05' [--offsets 25 100]
"""
import argparse, collections, re, statistics as st
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("tag_regex", help="regex matched against the W&B run name (which starts with EXPERIMENT_TAG)")
ap.add_argument("--offsets", nargs="+", default=["25", "100"])
ap.add_argument("--draws", nargs="+", default=["42", "43", "44"])
ap.add_argument("--project", default="armin-sommer/RLP")
ap.add_argument("--merge", nargs="*", default=[], help="regex substitutions 'from=to' applied to tags so seed waves pool (e.g. '-s345=')")
args = ap.parse_args()

api = wandb.Api(timeout=120)
runs = api.runs(args.project, filters={"$and": [{"display_name": {"$regex": args.tag_regex}}, {"jobType": "eval"}]}, per_page=500)
cells = collections.defaultdict(dict)   # (tag, env, base, actor) -> seed -> {offset: mean over draws}
sel = collections.defaultdict(dict)
pat = re.compile(r"^(?P<tag>.+?)_g_(?P<env>tworoom|cube)_(?P<base>[a-z]+)_(?P<actor>.+)_s(?P<seed>\d+)$")
n = 0
for r in runs:
    m = pat.match(r.name)
    if not m or r.state != "finished":
        continue
    n += 1
    tag = m["tag"]
    for f, t in (x.split("=", 1) for x in args.merge):
        tag = re.sub(f, t, tag)
    key = (tag, m["env"], m["base"], m["actor"])
    s = r.summary
    per = {}
    for off in args.offsets:
        vals = [s.get(f"eval/rh5/rh5_h{off}_s{d}") for d in args.draws]
        if all(v is not None for v in vals):
            per[off] = sum(vals) / len(vals)
    if per:
        cells[key][int(m["seed"])] = per
    sv = [s.get(k) for k in s.keys() if k.startswith("eval/ckv5/rh5_h")]
    sel[key][int(m["seed"])] = s.get("eval/ckv5/mean_rh5_all")
print(f"# {n} finished eval runs matched /{args.tag_regex}/ in {args.project}\n")
hdr = "| tag | env/base | actor | n | " + " | ".join(f"h{o} median (per seed)" for o in args.offsets) + " |"
print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
for key in sorted(cells):
    seeds = sorted(cells[key])
    cols = []
    for off in args.offsets:
        v = [cells[key][s][off] for s in seeds if off in cells[key][s]]
        cols.append(f"**{st.median(v):.1f}** ({'/'.join(f'{x:.1f}' for x in v)})" if v else "—")
    print(f"| {key[0]} | {key[1]}/{key[2]} | {key[3]} | {len(seeds)} | " + " | ".join(cols) + " |")
