"""Install the QRL learner: drop qrl.py into trm/learners/ and wire it through
the learner registry, the metric builder (so load_metric can rebuild it) and
train_metric.py's CLI. Idempotent."""
import shutil
from pathlib import Path

TRM = Path("/workspace/code/stable-worldmodel/stable_worldmodel/trm")
TM = Path("/workspace/code/stable-worldmodel/scripts/plan/train_metric.py")

shutil.copy("/workspace/qrl_learner.py", TRM / "learners" / "qrl.py")
print("installed learners/qrl.py")

# --- learners/__init__.py
p = TRM / "learners" / "__init__.py"
t = p.read_text()
if "qrl" not in t:
    t = t.replace("from . import contrastive, regression, td",
                  "from . import contrastive, qrl, regression, td")
    t = t.replace('__all__ = ["regression", "td", "contrastive"]',
                  '__all__ = ["regression", "td", "contrastive", "qrl"]')
    p.write_text(t)
    print("learners/__init__.py: qrl registered")
else:
    print("learners/__init__.py: already registered")

# --- io.py: qrl checkpoints rebuild as quasimetric heads
p = TRM / "io.py"
t = p.read_text()
if '"qrl"' not in t:
    a = 'if learner in ("regression", "td", "shuffled"):'
    assert t.count(a) == 1, "io.py learner tuple anchor"
    t = t.replace(a, 'if learner in ("regression", "td", "shuffled", "qrl"):')
    p.write_text(t)
    print("io.py: qrl added to builder")
else:
    print("io.py: already patched")

# --- train_metric.py: CLI choice + dispatch
t = TM.read_text()
if "QRLConfig" not in t:
    a = 'p.add_argument("--learner", required=True, choices=["regression", "td", "contrastive"])'
    assert t.count(a) == 1, "train_metric learner-choices anchor"
    t = t.replace(a, 'p.add_argument("--learner", required=True, choices=["regression", "td", "contrastive", "qrl"])\n'
                     '    p.add_argument("--qrl-eps", type=float, default=0.25, help="QRL constraint slack")\n'
                     '    p.add_argument("--qrl-spread-temp", type=float, default=100.0, help="QRL spreading temperature")\n'
                     '    p.add_argument("--qrl-step-cost", type=float, default=1.0, help="QRL cost of one env step")')

    a = "    else:  # contrastive"
    assert t.count(a) == 1, "train_metric dispatch anchor"
    t = t.replace(a, '''    elif args.learner == "qrl":
        cfg = QRLConfig(
            head=args.head, hidden_dim=args.hidden_dim, depth=args.depth,
            embed_dim=args.embed_dim, num_components=args.num_components,
            batch_size=args.batch_size, steps=args.steps, seed=args.seed,
            eps=args.qrl_eps, spread_temp=args.qrl_spread_temp, step_cost=args.qrl_step_cost,
            rank_weight=args.rank_weight, rank_margin=args.rank_margin,
        )
        module = learners.qrl.fit(cache, cfg, device)
        arch = {"head": args.head, "hidden_dim": args.hidden_dim, "depth": args.depth,
                "embed_dim": args.embed_dim, "softplus": True, "symmetric": False,
                "num_components": args.num_components}
    else:  # contrastive''')

    # import QRLConfig alongside the other configs
    for old, new in [
        ("from stable_worldmodel.trm.learners.td import TDConfig",
         "from stable_worldmodel.trm.learners.qrl import QRLConfig\nfrom stable_worldmodel.trm.learners.td import TDConfig"),
        ("from stable_worldmodel.trm.learners import td",
         "from stable_worldmodel.trm.learners import qrl, td"),
    ]:
        if old in t:
            t = t.replace(old, new, 1)
            break
    TM.write_text(t)
    print("train_metric.py: --learner qrl wired")
else:
    print("train_metric.py: already patched")

print("QRL_WIRING_OK")
