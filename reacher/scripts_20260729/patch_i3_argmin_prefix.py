"""I3: argmin-over-prefix readout. Eval-only, no retraining.

THE PROBLEM. The cost is the value of a plan's TERMINAL state, 5 chunks (25
primitive steps) out. held-at-end grades one instant: eval_budget. At
receding_horizon=5 with budget 50 the plan boundaries fall at 0/25/50, so the
terminal IS the graded state. At rh=1 the plans start at 0,5,...,45 and the ones
from step 30 on have terminals past step 50 -- the planner optimises a state the
episode never reaches. Measured: one RLP cell 100.0 -> 86.0 @0.1 unaligned.

I3. Fixed clock until the plan would overshoot the deadline; then score the best
still-reachable point on the plan instead of the fixed terminal:

    cost = min over j of V(z_j, z_g),   j <= min(H, chunks_remaining) - 1

Away from the deadline chunks_remaining >= H, the cap is inert, and the readout
is the terminal exactly as before -- so this cannot change any rh=5 number
(where 50 = 2 x 25 means chunks_remaining is never below H, and I3 provably
never fires).

WHY THIS IS ONLY A COST CHANGE. At rh=1 the executed length is one chunk by
construction, so min(rh, j*+1) = 1 and the execution rule is untouched. I3
changes only which point the planner optimises. That also means it needs nothing
inside RLP's refinement loop -- the actor keeps consuming its own value the way
it was trained to; only the scored index moves.

WINDOWING. With vframes=3, scoring prefix point j needs frames j-2, j-1, j. For
j<2 the window clamps at the start of the imagined sequence, which is the same
convention train_window.py uses at episode starts ("window F=3 lag=5:
100000/2010000 rows clamp at episode starts"). So no encoder history is needed.

FRAME AVAILABILITY. LeWM/PLDM already publish the whole rollout
(lewm.py: info['predicted_emb'] = pred_rollout), so all H chunks are present.
Only DinoWMTokens truncates to keep_pred_frames, which is why that is raised
here too.

Enabled by plan_config.deadline being set (already opt-in). Unset -> inert.
"""

import ast

# ---------------------------------------------------------------- 1. policy: publish the clock
P = "/workspace/swm_cem/stable_worldmodel/policy.py"
s = open(P).read()
assert "i3_chunks_remaining" not in s, "policy already patched"

OLD = "            outputs = self.solver(sliced, init_action=sliced_init)"
NEW = ('''            # I3: publish chunks-remaining so the cost can cap its prefix
            # search at the deadline. Only meaningful when a deadline is set;
            # otherwise left as None and the readout stays terminal.
            _dl = getattr(self.cfg, 'deadline', None)
            _mdl = getattr(self.solver, 'model', None)
            if _mdl is not None:
                if _dl:
                    _step = int(max(self._step_ct[i] for i in replan_idx))
                    _mdl.i3_chunks_remaining = max(
                        1, (int(_dl) - _step) // self.cfg.action_block)
                else:
                    _mdl.i3_chunks_remaining = None

            outputs = self.solver(sliced, init_action=sliced_init)''')
assert s.count(OLD) == 1, f"solver-call anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)
open(P, "w").write(s)
ast.parse(open(P).read())
print("policy.py: publishes model.i3_chunks_remaining before each solve")

# ---------------------------------------------------------------- 2. hook: argmin over prefix
E = "/workspace/swm_cem/scripts/plan/eval_wm.py"
e = open(E).read()
assert "i3" not in e.lower() or "i3_chunks_remaining" not in e, "eval_wm already patched"

OLD_H = """                    elif _m >= 3:
                        # window metric: last m imagined frames, goal tiled
                        pred = torch.cat(
                            [p_raw[..., -_m + j, :] for j in range(_m)], dim=-1)
                        goal = torch.cat([goal] * _m, dim=-1)"""
NEW_H = """                    elif _m >= 3:
                        # window metric: last m imagined frames, goal tiled
                        pred = torch.cat(
                            [p_raw[..., -_m + j, :] for j in range(_m)], dim=-1)
                        goal = torch.cat([goal] * _m, dim=-1)
                        # I3: if the plan would overshoot the deadline, score the
                        # best still-reachable chunk instead of the terminal.
                        # Windows clamp at the start of the imagined sequence,
                        # matching train_window.py's episode-start convention.
                        _cr = getattr(self.base, 'i3_chunks_remaining', None)
                        _T = p_raw.shape[-2]
                        if _cr is not None and int(_cr) < _T:
                            _cands = []
                            for _j in range(int(_cr)):
                                _cols = [p_raw[..., max(0, _j - _m + 1 + k), :]
                                         for k in range(_m)]
                                _cands.append(torch.cat(_cols, dim=-1))
                            pred = torch.stack(_cands, dim=0)
                            goal = goal.unsqueeze(0).expand_as(pred)
                            if not getattr(self, '_i3_said', False):
                                print(f'[I3] {_cr} of {_T} chunks reachable: '
                                      f'argmin over prefix', flush=True)
                                self._i3_said = True"""
assert e.count(OLD_H) == 1, f"hook anchor x{e.count(OLD_H)}"
e = e.replace(OLD_H, NEW_H)

# the cost must reduce over the new leading candidate axis
OLD_C = "                    if goal.ndim < pred.ndim:"
NEW_C = """                    _i3_stack = (pred.ndim == goal.ndim
                                 and getattr(self.base, 'i3_chunks_remaining', None)
                                 is not None
                                 and pred.shape[0] != p_raw.shape[0]
                                 and pred.dim() > 2)
                    if goal.ndim < pred.ndim:"""
assert e.count(OLD_C) == 1, f"cost anchor x{e.count(OLD_C)}"
e = e.replace(OLD_C, NEW_C, 1)

open(E, "w").write(e)
ast.parse(open(E).read())
print("eval_wm.py: argmin-over-prefix readout when the plan overshoots")
print("NOTE: inert unless plan_config.deadline is set; provably inert at rh=5")
