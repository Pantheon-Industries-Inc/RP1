"""Add --td-weight so the TD regression term can be switched off (td_weight=0
=> pure ranking value, the control predicted to be scale-degenerate)."""
from pathlib import Path

td = Path("/workspace/code/stable-worldmodel/stable_worldmodel/trm/learners/td.py")
tm = Path("/workspace/code/stable-worldmodel/scripts/plan/train_metric.py")

t = td.read_text()
if "td_weight" in t:
    print("td.py: already patched")
else:
    a = "    rank_weight: float = 0.0      # pairwise ranking hinge (0 = off)"
    assert t.count(a) == 1, "td.py config anchor"
    t = t.replace(a, "    td_weight: float = 1.0        # 0 = pure ranking (no distance regression)\n" + a)
    a = "        loss = _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)"
    assert t.count(a) == 1, "td.py loss anchor"
    t = t.replace(a, "        loss = cfg.td_weight * _expectile_loss(pred - tgt, cfg.expectile, cfg.huber_beta)")
    td.write_text(t)
    print("td.py: td_weight added")

t = tm.read_text()
if "td-weight" in t:
    print("train_metric.py: already patched")
else:
    a = '    p.add_argument("--rank-weight", type=float, default=0.0,'
    assert t.count(a) == 1, "train_metric anchor"
    t = t.replace(a, '    p.add_argument("--td-weight", type=float, default=1.0,\n'
                     '                   help="weight on the TD distance-regression term (0 = pure ranking)")\n' + a)
    a = "            rank_weight=args.rank_weight, rank_margin=args.rank_margin,"
    assert t.count(a) == 1, "train_metric TDConfig anchor"
    t = t.replace(a, a + "\n            td_weight=args.td_weight,")
    tm.write_text(t)
    print("train_metric.py: --td-weight added")
print("TD_WEIGHT_PATCH_OK")
