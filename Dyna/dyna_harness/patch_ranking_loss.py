"""Add a planner-relevant pairwise RANKING (hinge) loss to the TD value learner.

Motivation (measured): on the comparison the planner actually consumes — fixed
goal g, two candidate states at true distances d_near < d_far — the deployed TD
value ranks correctly only 51.5% of the time at 50-vs-75 steps, while a network
trained with a margin-ranking loss on the same inputs reaches 98.9%. Distance
regression optimises absolute values and collapses toward the conditional mean
under ambiguity; the planner only ever consumes ORDER.

    L_rank = mean relu( margin + V(s_near, g) - V(s_far, g) )

with margin scaled by the true separation so the value keeps its "steps" units.
Combined: L = L_td + rank_weight * L_rank.  Idempotent.
"""
from pathlib import Path

td = Path("/workspace/code/stable-worldmodel/stable_worldmodel/trm/learners/td.py")
tm = Path("/workspace/code/stable-worldmodel/scripts/plan/train_metric.py")

t = td.read_text()
if "rank_weight" in t:
    print("td.py: already patched")
else:
    a = "    eikonal_weight: float = 0.0   # Eik-HIQL unit-gradient penalty (0 = off)"
    assert t.count(a) == 1, "td.py config anchor"
    t = t.replace(a, a + "\n"
        "    rank_weight: float = 0.0      # pairwise ranking hinge (0 = off)\n"
        "    rank_margin: float = 0.5      # margin per step of true separation\n"
        "    rank_max_delta: int = 200     # max goal distance for ranking triplets")

    a = "    g = cfg.gamma\n"
    assert t.count(a) >= 1, "td.py pre-loop anchor"
    t = t.replace(a, '''    g = cfg.gamma

    if cfg.rank_weight > 0:
        # planner-relevant triplets: ONE goal, two states at different true
        # distances from it. (The TD sampler gives one state + one goal, which
        # is the wrong shape for this comparison.)
        _rk_eps = cache.episodes()
        _rk_keys = [k for k in _rk_eps if len(_rk_eps[k]) > 3]
        _rk_rows = {k: np.asarray(_rk_eps[k]) for k in _rk_keys}
        _rk_rng = np.random.default_rng(cfg.seed + 1)

        def _rank_batch(bs):
            near = np.empty(bs, np.int64); far = np.empty(bs, np.int64)
            goal = np.empty(bs, np.int64); gap = np.empty(bs, np.float32)
            for b in range(bs):
                r = _rk_rows[_rk_keys[_rk_rng.integers(len(_rk_keys))]]
                L = len(r)
                hi = min(cfg.rank_max_delta, L - 1)
                d_far = int(_rk_rng.integers(2, hi + 1))
                d_near = int(_rk_rng.integers(1, d_far))
                gi = int(_rk_rng.integers(d_far, L))
                near[b], far[b], goal[b] = r[gi - d_near], r[gi - d_far], r[gi]
                gap[b] = d_far - d_near
            return (cache.z[near], cache.z[far], cache.z[goal],
                    torch.from_numpy(gap))

''')

    a = """        if cfg.eikonal_weight > 0:"""
    assert t.count(a) == 1, "td.py eikonal anchor"
    t = t.replace(a, """        if cfg.rank_weight > 0:
            zn, zf, zg_r, gap = _rank_batch(cfg.batch_size)
            zn, zf, zg_r, gap = zn.to(device), zf.to(device), zg_r.to(device), gap.to(device)
            v_near, v_far = value(zn, zg_r), value(zf, zg_r)
            margin = cfg.rank_margin * gap
            loss = loss + cfg.rank_weight * torch.relu(margin + v_near - v_far).mean()
        if cfg.eikonal_weight > 0:""")
    if "import numpy as np" not in t:
        t = t.replace("import torch", "import numpy as np\nimport torch", 1)
    td.write_text(t)
    print("td.py: ranking hinge added")

t = tm.read_text()
if "rank-weight" in t:
    print("train_metric.py: already patched")
else:
    a = '    p.add_argument("--num-components", type=int, default=8, help="iqe head only")'
    assert t.count(a) == 1, "train_metric anchor"
    t = t.replace(a, a + '\n'
        '    p.add_argument("--rank-weight", type=float, default=0.0,\n'
        '                   help="pairwise ranking hinge weight (0 = off)")\n'
        '    p.add_argument("--rank-margin", type=float, default=0.5,\n'
        '                   help="ranking margin per step of true separation")')
    a = "            eikonal_weight=args.eikonal_weight, num_components=args.num_components,"
    assert t.count(a) == 1, "train_metric TDConfig anchor"
    t = t.replace(a, a + "\n            rank_weight=args.rank_weight, rank_margin=args.rank_margin,")
    tm.write_text(t)
    print("train_metric.py: --rank-weight added")

print("RANK_PATCH_OK")
