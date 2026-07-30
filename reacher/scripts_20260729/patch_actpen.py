"""Add --act-penalty to train_lip_ac.py: a terminal-action-magnitude term.

Why this specific lever for HELD-at-end. Every arm reaches the ball at ~90%
(ever-in-ball) but holds at 12-54%: the loss is entirely settling, not targeting.
The at-rest experiment proved "at rest" is NOT expressible in a single LeWM
latent (stopped vs moving separates at chance, AUC 0.496) and only weakly at the
deployable 5-step lag (AUC 0.61) -- so no purely latent-space value can ask for
it cleanly.

But the planner already HOLDS the action sequence it is about to execute, and
"end at rest" is trivial there: penalise the magnitude of the final action block.
This needs no velocity in the latent, no readout, and no extra rollout -- it is a
term on A itself. lam * mean(A[:, -1]**2) added to the actor loss.

Deploy-time equivalence note: the LIP solver picks its plan by the value alone,
so the penalty shapes only what the ACTOR learns to emit, which is exactly where
the under-actuation/overshoot trade-off lives.
"""

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()
if "--act-penalty" in s:
    print("already patched")
    raise SystemExit

OLD_ARG = '    p.add_argument("--bc-weight", type=float, default=0.0'
NEW_ARG = ('    p.add_argument("--act-penalty", type=float, default=0.0,\n'
           '                   help="weight on mean(A[:,-1]**2): asks the plan to END AT REST, "\n'
           '                        "the settling term held-at-end rewards and that no latent-space "\n'
           '                        "value can express (at-rest AUC 0.496 at 1 frame)")\n'
           '    p.add_argument("--bc-weight", type=float, default=0.0')
assert s.count(OLD_ARG) == 1, f"arg anchor x{s.count(OLD_ARG)}"
s = s.replace(OLD_ARG, NEW_ARG)

OLD_LOSS = """        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()"""
NEW_LOSS = """        loss = e_path[-1] + a.mean_weight * torch.stack(e_path).mean()
        if a.act_penalty > 0:
            # end-at-rest: the last planned action block should be small
            loss = loss + a.act_penalty * (A[:, -1] ** 2).mean()"""
assert s.count(OLD_LOSS) == 1, f"loss anchor x{s.count(OLD_LOSS)}"
s = s.replace(OLD_LOSS, NEW_LOSS)
open(P, "w").write(s)

import ast

ast.parse(open(P).read())
print("train_lip_ac.py: --act-penalty added; syntax OK")
