"""Make the --expand-weight path vframes-aware.

The value-expansion block feeds the actor's own rollout states to the critic:
    tgt_e = plan_cost + plan_disc * teacher(zTe, zge)
    loss += expand_weight * expectile(critic(z0e, zge) - tgt_e)
but it stores single 192-d frames, which cannot be scored by the 576-d 3-frame
window value -- the same shape error the deploy path had.

Fix: stack the expansion tuple the same way every other value query is stacked
(last `vframes` imagined frames, goal tiled), so the stored tuple is already in
the value's input space and critic_step needs no change.

CAVEAT worth carrying into the results: the trainer's own docstring warns that
the actor's imagined terminal and the critic can CO-EXPLOIT world-model error
under expansion. That is exactly the failure mode diagnosed on this env
(corr(E_final, HELD) = +0.583), so expand-weight could plausibly make things
worse here even though it helped on OGBench. Always compare against
expand-weight 0 on the same seeds.
"""

import ast

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()
if "_wpair(zh, zg)" in s:
    print("already patched")
    raise SystemExit

OLD = "        expand = (z0.detach(), zT.detach(), zg.detach())"
NEW = '''        if vframes > 1:
            # stack the expansion tuple into the window value's input space:
            # start-state window (pad-context makes this [z0]*hs), terminal
            # window from the rollout, goal tiled to match
            _e0, _eg = _wpair(zh, zg)
            _eT, _ = _wpair(tr, zg)
            expand = (_e0.detach(), _eT.detach(), _eg.detach())
        else:
            expand = (z0.detach(), zT.detach(), zg.detach())'''
assert s.count(OLD) == 1, f"expand anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)
open(P, "w").write(s)
ast.parse(open(P).read())
print("train_lip_ac.py: expand path is now vframes-aware; syntax OK")
