"""Add --pad-context to train_lip_ac.py: train under the DEPLOYED 1-frame
conditioning.

RECONSTRUCTED 2026-07-30 (original lost with volume 3cv5zzezm9).

THE CONFIG is the paper's: the policy sees ONE real frame, which the LIP solver
pads to [z0, z0, z0] with zero action history. But LIP actors train on REAL
3-consecutive-frame cache windows, so at deployment they run under a train/
deploy conditioning mismatch. Collapsing the training context to the deployed
form is worth roughly +5 held on the terminal critic (measured: pooled 44.0 ->
48.6, best actor 57.0).

Applied AFTER any replay merge on purpose: replayed (imagined) windows collapse
to their own last frame, which is exactly what a deployed replan would see.
"""

import ast

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()
if "--pad-context" in s:
    print("already patched")
    raise SystemExit

OLD_ARG = '    p.add_argument("--batch", type=int, default=128)'
NEW_ARG = ('    p.add_argument("--batch", type=int, default=128)\n'
           '    p.add_argument("--pad-context", action="store_true",\n'
           '                   help="train under the 1-frame deployment conditioning (THE paper "\n'
           '                        "config): context collapsed to [z0]*hs with zero action "\n'
           '                        "history, instead of real multi-frame cache windows")')
assert s.count(OLD_ARG) == 1, f"arg anchor x{s.count(OLD_ARG)}"
s = s.replace(OLD_ARG, NEW_ARG)

# collapse the context immediately before z0 is taken from it
OLD = "        z0 = zh[:, -1]"
NEW = ('        if a.pad_context:\n'
       '            # 1-frame deployment interface (THE config): only z0 is real,\n'
       '            # the solver pads the rest and zeroes the action history\n'
       '            zh = zh[:, -1:].expand_as(zh).contiguous()\n'
       '            ah = torch.zeros_like(ah)\n'
       '        z0 = zh[:, -1]')
assert s.count(OLD) == 1, f"context anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)
open(P, "w").write(s)
ast.parse(open(P).read())
print("train_lip_ac.py: --pad-context added; syntax OK")
