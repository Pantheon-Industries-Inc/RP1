#!/usr/bin/env python3
"""Collect RLP eval results straight from W&B (no cluster job needed).

TwoRoom/Cube jobs launched through scripts/sky/unigamma/tworoom_g98_rescue.yaml
log one eval run per (tag, config, seed) named
  <EXPERIMENT_TAG>_g_<env>_<base>_<config>_s<seed>
whose summary carries eval/rh5/rh5_h{25,100}_s{42,43,44} (report draws) and
eval/ckv5/rh5_h{25,100}_s{48..51} (selection draws, ES pass).

Reacher jobs (reacher_gamma.yaml) log NOTHING usable to the run summary; their
results/summary.csv is inside the `reacher_<stem>` W&B artifact (type
lip_actor), fetched with --reacher.

  python3 scripts/wandb_collect_rlp.py 'v2l05'            # tworoom + cube
  python3 scripts/wandb_collect_rlp.py 'v2l05' --reacher  # + reacher CSVs
"""
import argparse, collections, re, statistics as st, sys, tempfile, os
import wandb

ap = argparse.ArgumentParser()
ap.add_argument("tag_regex")
ap.add_argument("--entity", default="armin-sommer"); ap.add_argument("--project", default="RLP")
ap.add_argument("--reacher", action="store_true", help="also pull reacher summary.csv artifacts")
ap.add_argument("--reacher_regex", default=r"^reacher_rs_", help="display-name regex for reacher sweep runs")
ap.add_argument("--since", default="2026-09-01T00:00:00Z")
a = ap.parse_args()
api = wandb.Api(timeout=120)

# ---- tworoom / cube: one summary per (tag, env, base, cfg, seed)
runs = api.runs(f"{a.entity}/{a.project}", filters={"$and": [{"display_name": {"$regex": a.tag_regex}}, {"jobType": "eval"}]}, per_page=500)
cells = collections.defaultdict(dict)   # (env, base, cfg) -> seed -> {h: mean over report draws}
seen = set()
for r in runs:
    m = re.match(r"(?P<tag>.+)_g_(?P<env>tworoom|cube)_(?P<base>lejepa|lewm|pldm|dinowm)_(?P<cfg>.+)_s(?P<seed>\d+)$", r.name)
    if not m or r.name in seen: continue
    seen.add(r.name)
    s = r.summary
    per_h = {}
    for h in ("25", "100"):
        vals = [s[k] for k in (f"eval/rh5/rh5_h{h}_s{d}" for d in (42, 43, 44)) if k in s]
        if len(vals) == 3: per_h[h] = sum(vals) / 3
    if per_h:
        cells[(m["env"], m["base"], m["cfg"])][int(m["seed"])] = (m["tag"], per_h)
print(f"# {len(seen)} eval runs matched /{a.tag_regex}/")
for (env, base, cfg), seeds in sorted(cells.items()):
    tags = sorted({t for t, _ in seeds.values()})
    print(f"\n## {env} / {base} / {cfg}   tags: {', '.join(tags)}")
    hs = [h for h in ("25", "100") if any(h in v[1] for v in seeds.values())]
    print("| seed | " + " | ".join(f"h{h}" for h in hs) + " | tag |")
    for sd in sorted(seeds):
        t, ph = seeds[sd]
        print(f"| {sd} | " + " | ".join(f"{ph.get(h, float('nan')):.1f}" for h in hs) + f" | {t} |")
    meds = {h: st.median([v[1][h] for v in seeds.values() if h in v[1]]) for h in hs}
    print(f"| **median n={len(seeds)}** | " + " | ".join(f"**{meds[h]:.1f}**" for h in hs) + " | |")

# ---- reacher: summary.csv lives in the lip_actor artifact of each sweep run.
# The sweep run does not know its EXPERIMENT_TAG, but the actor_train run that
# precedes it (~1 h) carries it in config.wandb_run_id = "<tag>-<cfg>-<base>-s<seed>",
# so each sweep run is assigned to the nearest preceding training run of the
# same (cfg, base, seed), and rows are grouped by tag.
if a.reacher:
    flt = lambda jt, rx: {"$and": [{"display_name": {"$regex": rx}}, {"jobType": jt}, {"createdAt": {"$gte": a.since}}]}
    trains = collections.defaultdict(list)   # (cfg, base, seed) -> [(created_at, tag)]
    for r in api.runs(f"{a.entity}/{a.project}", filters=flt("actor_train", r"^reacher_train_rs_"), per_page=500):
        m = re.match(r"reacher_train_(?P<cfg>rs_.+)_(?P<base>lejepa|pldm)_s(?P<seed>\d+)$", r.name)
        rid = r.config.get("wandb_run_id", "")
        if not m or not rid: continue
        tag = rid.split(f"-{m['cfg']}-")[0]
        trains[(m["cfg"], m["base"], int(m["seed"]))].append((r.created_at, tag))
    rr = api.runs(f"{a.entity}/{a.project}", filters=flt("reacher_sweep", a.reacher_regex), per_page=500)
    rows = collections.defaultdict(dict)   # (tag, base, cfg) -> seed -> (date, {tau: mean over 42/43/44})
    tmp = tempfile.mkdtemp(prefix="wandb_reacher_")
    for r in rr:
        m = re.match(r"reacher_(?P<cfg>rs_.+)_(?P<base>lejepa|pldm)_s(?P<seed>\d+)$", r.name)
        if not m: continue
        key = (m["cfg"], m["base"], int(m["seed"]))
        arts = [x for x in r.logged_artifacts() if x.type == "lip_actor"]
        if not arts: continue
        art = arts[-1]
        csvs = [f for f in art.manifest.entries if f.endswith("summary.csv")]
        if not csvs: continue
        d = os.path.join(tmp, r.id); os.makedirs(d, exist_ok=True)
        # Authoritative tag: the training log shipped in the artifact prints
        # "wandb: setting up run <TAG>-<cfg>-<base>-s<seed>". The time-based
        # join below is only a fallback -- two jobs of the same cell running
        # concurrently (e.g. critB vs bandB on 2026-09-20) break it.
        tag = None
        logs = [f for f in art.manifest.entries if f.startswith("train_") and f.endswith(".log")]
        if logs:
            art.get_entry(logs[0]).download(root=d)
            txt = open(os.path.join(d, logs[0]), errors="ignore").read()
            mm = re.search(r"setting up run (\S+?)-" + re.escape(m["cfg"]) + "-" + m["base"] + "-s" + m["seed"], txt)
            if mm: tag = mm.group(1)
        if tag is None:
            prev = [t for t in trains.get(key, []) if t[0] <= r.created_at]
            tag = max(prev)[1] + "?" if prev else "?"
        if not re.search(a.tag_regex, tag): continue
        art.get_entry(csvs[0]).download(root=d)
        pre = f"{m['cfg']}_s{m['seed']}_final_e"
        vals = collections.defaultdict(list)
        for line in open(os.path.join(d, csvs[0])):
            if not line.startswith(pre): continue
            name, *kv = line.strip().split(",")
            if name[len(pre):] not in ("42", "43", "44"): continue
            for item in kv:
                k, v = item.split("=")
                if v != "FAIL": vals[k].append(float(v))
        if vals:
            rows[(tag, m["base"], m["cfg"])][int(m["seed"])] = (r.created_at[:16], {k: sum(v) / len(v) for k, v in vals.items() if len(v) == 3})
    bycell = collections.defaultdict(dict)   # (base, cfg) -> seed -> (tag, date, vals): pooled across tags matching the regex
    for (tag, base, cfg), seeds in sorted(rows.items()):
        print(f"\n## reacher / {base} / {cfg}   tag: {tag}")
        keys = sorted({k for v in seeds.values() for k in v[1]})
        print("| seed | " + " | ".join(keys) + " | eval run |")
        for sd in sorted(seeds):
            dt, ph = seeds[sd]
            print(f"| {sd} | " + " | ".join(f"{ph.get(k, float('nan')):.1f}" for k in keys) + f" | {dt} |")
            bycell[(base, cfg)][sd] = (tag, dt, ph)
        print(f"| **median n={len(seeds)}** | " + " | ".join(f"**{st.median([v[1][k] for v in seeds.values() if k in v[1]]):.1f}**" for k in keys) + " | |")
    for (base, cfg), seeds in sorted(bycell.items()):
        if len({v[0] for v in seeds.values()}) < 2: continue
        keys = sorted({k for v in seeds.values() for k in v[2]})
        print(f"\n## reacher / {base} / {cfg}   POOLED over tags matching /{a.tag_regex}/ (one row per seed, last tag wins)")
        print(f"| **median n={len(seeds)}** | " + " | ".join(f"**{st.median([v[2][k] for v in seeds.values() if k in v[2]]):.1f} {k}**" for k in keys) + " | |")
