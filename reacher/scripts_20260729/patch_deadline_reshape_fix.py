"""Fix two places that hardcode receding_horizon, breaking deadline alignment.

The deadline patch lets keep_horizon rise from receding_horizon to horizon for
the final plan. Two sites still assumed the smaller value:

  policy.py:558  plan.reshape(len(replan_idx), self.flatten_receding_horizon, -1)
      With rh=1 that is 5 primitive steps, but the committed plan holds 25, so
      the reshape produced 10-d entries instead of 2-d actions:
      "RuntimeError: The expanded size of the tensor (2) must match the existing
      size (10) at non-singleton dimension 0"

  policy.py:365  deque(maxlen=self.flatten_receding_horizon)
      maxlen 5 would have silently DROPPED 20 of the 25 committed steps -- worse
      than the crash, because it would have run and produced a plausible number.

Both now use the actual committed length. The deque is sized to plan_len (the
maximum any plan can contribute), which is a no-op when keep_horizon equals
receding_horizon.
"""

import ast

P = "/workspace/swm_cem/stable_worldmodel/policy.py"
s = open(P).read()

OLD_R = """            plan = plan.reshape(
                len(replan_idx), self.flatten_receding_horizon, -1
            )"""
NEW_R = """            # keep_horizon can exceed receding_horizon on the final,
            # deadline-aligned plan, so the flattened length must follow it.
            plan = plan.reshape(
                len(replan_idx), keep_horizon * self.cfg.action_block, -1
            )"""
assert s.count(OLD_R) == 1, f"reshape anchor x{s.count(OLD_R)}"
s = s.replace(OLD_R, NEW_R)

OLD_D = "            deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)"
NEW_D = ("            # plan_len, not flatten_receding_horizon: a deadline-aligned final\n"
         "            # plan commits up to the whole horizon, and a short maxlen would\n"
         "            # silently discard its leading steps.\n"
         "            deque(maxlen=self.cfg.plan_len) for _ in range(n_envs)")
assert s.count(OLD_D) == 1, f"deque anchor x{s.count(OLD_D)}"
s = s.replace(OLD_D, NEW_D)

open(P, "w").write(s)
ast.parse(open(P).read())
print("policy.py: reshape uses keep_horizon*action_block; deque sized to plan_len")
