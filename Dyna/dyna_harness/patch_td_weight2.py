"""Finish the --td-weight wiring in train_metric.py (TD branch only; the QRL
branch has its own config and must not receive td_weight)."""
from pathlib import Path

p = Path("/workspace/code/stable-worldmodel/scripts/plan/train_metric.py")
t = p.read_text()

if "td-weight" in t:
    print("already patched")
else:
    a = '    p.add_argument("--rank-weight", type=float, default=0.0,'
    assert t.count(a) == 1, "argparse anchor"
    t = t.replace(
        a,
        '    p.add_argument("--td-weight", type=float, default=1.0,\n'
        '                   help="weight on TD distance regression (0 = pure ranking)")\n' + a,
    )
    # TD branch only: QRLConfig has no n_step/gamma/expectile line
    a = ("            rank_weight=args.rank_weight, rank_margin=args.rank_margin,\n"
         "            n_step=args.n_step, gamma=args.gamma, expectile=args.expectile,")
    assert t.count(a) == 1, "TDConfig anchor"
    t = t.replace(
        a,
        "            rank_weight=args.rank_weight, rank_margin=args.rank_margin,\n"
        "            td_weight=args.td_weight,\n"
        "            n_step=args.n_step, gamma=args.gamma, expectile=args.expectile,",
    )
    p.write_text(t)
    print("train_metric.py patched")
print("TD_WEIGHT2_OK")
