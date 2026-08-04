"""Two fixes so held-at-end is fair at the published replanning cadence.

Both address the same finding: at receding_horizon=1 a single RLP cell fell from
100.0 to 86.0 @0.1 (and 66.0 -> 42.0 @0.05). Neither cause is "feedback hurts".

(1) RLP DISCARDS THE WARM START, CEM DOES NOT.
    policy.py already carries the unexecuted plan tail forward and passes it as
    solver(sliced, init_action=sliced_init). cem.py consumes it via
    prepare_init_action. lip.py accepts init_action in its signature (line 791)
    and never references it again -- _proposal_lip builds its plan from restart
    noise every time. So at rh=1 CEM warm-starts ten times per episode and RLP
    re-searches from scratch ten times. That asymmetry is ours, not the
    protocol's, and it penalises RLP specifically.

    Fix: seed restart 0 from the carried-forward tail (shifted and padded to the
    horizon) and leave the other restarts noisy, so exploration is unchanged.

(2) A TERMINAL COST NEEDS THE PLAN BOUNDARY ON THE GRADED STEP.
    held-at-end grades one instant: the state at eval_budget (50). The cost is
    the value of a plan's TERMINAL state, 5 blocks = 25 steps out. At
    receding_horizon=5 the boundaries fall at 0/25/50, so the second plan's
    terminal IS the graded state -- the optimiser optimises exactly what is
    scored. At rh=1 the boundaries are 0,5,...,45: the plan begun at 45
    optimises step 70, only its first block executes, and no plan ever targets
    step 50.

    The papers do not have this problem and so do not solve it: PLDM's success
    is "distance below 1.4 pixels" at ANY point in the episode, so there is no
    graded instant to align to, and neither PLDM nor DINO-WM shrinks the
    horizon. held-at-end is our own stricter choice, so the alignment is ours to
    handle.

    Fix: once the remaining budget is <= plan length, commit the whole plan
    instead of one block. Replanning stays frequent early; the final plan's
    terminal lands exactly on the graded step. The horizon is never shortened,
    so the actor is never run at a horizon it was not trained at (its positional
    embedding is sized to horizon=5).

Applied to policy.py, PlanConfig, eval_wm.py and lip.py. Every change is
inert unless plan_config.deadline is set or init_action is supplied.
"""

import ast
import re

# ---------------------------------------------------------------- 1. PlanConfig.deadline
P = "/workspace/swm_cem/stable_worldmodel/policy.py"
s = open(P).read()
assert "deadline" not in s, "policy.py already patched"

OLD_CFG = """    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True"""
NEW_CFG = """    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True
    # Step at which the episode is graded (held-at-end). When set, the policy
    # commits the whole plan once fewer than plan_len steps remain, so a
    # terminal-cost planner's final plan ends exactly on the graded step. None
    # reproduces the previous behaviour exactly.
    deadline: int | None = None"""
assert s.count(OLD_CFG) == 1, "PlanConfig anchor"
s = s.replace(OLD_CFG, NEW_CFG)

OLD_KEEP = """            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon"""
NEW_KEEP = """            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon
            # Deadline alignment: with a terminal cost, the last plan must end on
            # the graded step, otherwise it optimises a state beyond the episode
            # and only its first block is ever executed.
            _dl = getattr(self.cfg, 'deadline', None)
            if _dl:
                _rem = int(_dl) - int(max(self._step_ct[i] for i in replan_idx))
                if 0 < _rem <= self.cfg.plan_len:
                    keep_horizon = self.cfg.horizon
                    if not getattr(self, '_dl_announced', False):
                        print(f'[deadline] {_rem} steps left <= plan_len '
                              f'{self.cfg.plan_len}: committing the full plan so '
                              f'its terminal lands on step {_dl}', flush=True)
                        self._dl_announced = True"""
assert s.count(OLD_KEEP) == 1, "keep_horizon anchor"
s = s.replace(OLD_KEEP, NEW_KEEP)
open(P, "w").write(s)
ast.parse(open(P).read())
print("policy.py: PlanConfig.deadline + commit-full-plan-near-deadline")

# ---------------------------------------------------------------- 2. wire the deadline
E = "/workspace/swm_cem/scripts/plan/eval_wm.py"
e = open(E).read()
if "plan_config.deadline" in e or "deadline =" in e:
    print("eval_wm.py: already wired")
else:
    OLD_E = "    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget"
    NEW_E = ("    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget\n"
             "    # held-at-end grades the state at eval_budget; tell the policy so a\n"
             "    # terminal-cost plan can be made to end there (see PlanConfig.deadline)\n"
             "    if cfg.plan_config.get('deadline', None) is None:\n"
             "        cfg.plan_config.deadline = int(cfg.eval.eval_budget)")
    assert e.count(OLD_E) == 1, "eval_wm anchor"
    e = e.replace(OLD_E, NEW_E)
    open(E, "w").write(e)
    ast.parse(open(E).read())
    print("eval_wm.py: deadline = eval_budget")

# ---------------------------------------------------------------- 3. RLP warm start
L = "/workspace/swm_cem/stable_worldmodel/solver/lip.py"
l = open(L).read()
assert "_warm_seed" not in l, "lip.py already patched"

# thread init_action into the proposal
l2 = l.replace("            proposal = self._proposal_lip(info_dict, total_envs)",
               "            proposal = self._proposal_lip(info_dict, total_envs,\n"
               "                                          init_action=init_action)", 1)
assert l2 != l, "proposal call anchor"
l = l2

l2 = l.replace("    def _proposal_lip(self, info_dict, n_envs):",
               "    def _proposal_lip(self, info_dict, n_envs, init_action=None):", 1)
assert l2 != l, "proposal def anchor"
l = l2

# seed restart 0 from the carried-forward tail; leave other restarts noisy
OLD_A = re.search(
    r"( *)A = self\.restart_noise \* torch\.randn\(B \* R, self\.horizon, self\.action_dim,\n"
    r"[^\n]*\n", l)
assert OLD_A, "restart-noise anchor"
blk = OLD_A.group(0)
ind = OLD_A.group(1)
NEW_A = blk + (
    f"{ind}# Warm start: policy.py carries the unexecuted plan tail forward and\n"
    f"{ind}# passes it as init_action (cem.py consumes it; this path used to drop\n"
    f"{ind}# it). Seed ONLY restart 0 so exploration across restarts is unchanged.\n"
    f"{ind}if init_action is not None and getattr(self, 'use_warm_start', True):\n"
    f"{ind}    _w = init_action.to(device=self.device, dtype=A.dtype)\n"
    f"{ind}    if _w.dim() == 3 and _w.shape[0] == B and _w.shape[-1] == self.action_dim:\n"
    f"{ind}        _seed = A.new_zeros(B, self.horizon, self.action_dim)\n"
    f"{ind}        _k = min(_w.shape[1], self.horizon)\n"
    f"{ind}        _seed[:, :_k] = _w[:, :_k]          # shift: tail becomes the prefix\n"
    f"{ind}        if _k < self.horizon:               # pad by repeating the last block\n"
    f"{ind}            _seed[:, _k:] = _w[:, _k - 1:_k]\n"
    f"{ind}        _warm_seed = _seed\n"
    f"{ind}        A = A.view(B, R, self.horizon, self.action_dim)\n"
    f"{ind}        A[:, 0] = _warm_seed\n"
    f"{ind}        A = A.view(B * R, self.horizon, self.action_dim)\n")
l = l.replace(blk, NEW_A, 1)
open(L, "w").write(l)
ast.parse(open(L).read())
print("lip.py: _proposal_lip seeds restart 0 from init_action (warm start)")
print("all three files parse")
