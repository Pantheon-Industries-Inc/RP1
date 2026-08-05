"""Let train_lip_ac.py distil a PARAMETER-FREE cost (e.g. the unlearned window-L2).

Motivation: at h25 the unlearned 3-frame window-L2 is the strongest cost measured
(89.7 vs 86.7 for the learned window quasimetric), but LIP has only ever been
trained against learned critics. LIP tracks its critic to ~1-2 points, so
distilling the better cost is the most direct route to lifting it. The window-L2
is differentiable, so grad_A V still flows -- the only blocker is bookkeeping:
AdamW raises on an empty parameter list, and the TD/EMA machinery has nothing to
update.

Patch: when the loaded value has no parameters, skip the critic optimizer, force
pretrain=0 and freeze_critic_frac=0 (so critic_step is never called), and make
the EMA a no-op. The actor path is untouched -- it still backprops through V.
"""

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()
if "PARAM-FREE critic" in s:
    print("already patched")
    raise SystemExit

OLD = "    c_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=a.critic_wd)"
NEW = '''    # PARAM-FREE critic (e.g. the unlearned window-L2): nothing to optimise, so
    # skip the optimizer/TD/EMA entirely and let the actor distil a fixed cost.
    _critic_params = [p for p in critic.parameters()]
    _fixed_critic = len(_critic_params) == 0
    if _fixed_critic:
        print("[fixed-critic] value has no parameters: TD/EMA disabled, actor "
              "distils it as a fixed cost", flush=True)
        a.pretrain = 0
        a.freeze_critic_frac = 0.0
        c_opt = None
    else:
        c_opt = torch.optim.AdamW(_critic_params, lr=a.critic_lr, weight_decay=a.critic_wd)'''
assert s.count(OLD) == 1, f"c_opt anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)

# critic_step must be inert if it is ever reached
OLD2 = """    def critic_step(expand=None, tau=None, lr=None):
        tau = a.expectile if tau is None else tau"""
NEW2 = """    def critic_step(expand=None, tau=None, lr=None):
        if c_opt is None:
            return float("nan")
        tau = a.expectile if tau is None else tau"""
assert s.count(OLD2) == 1, f"critic_step anchor x{s.count(OLD2)}"
s = s.replace(OLD2, NEW2)

# saving: a param-free module has an empty state_dict, which is fine, but keep
# the declared width so the eval hook still infers m=3
open(P, "w").write(s)

import ast

ast.parse(open(P).read())
print("train_lip_ac.py: param-free critic support added; syntax OK")
