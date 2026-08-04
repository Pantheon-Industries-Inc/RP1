#!/usr/bin/env python3
"""I3, second pass: fix the frame axis, the crash, and the readout semantics.

MEASURED (logs/i3fix, RLP lejepa s0, seed 42):
  A rh=5 + deadline   66.0 / 100.0, no [I3]      -- inert, reported cell intact
  B rh=1 + deadline   24.0 /  86.0, [I3] fired   -- WORSE @0.05 than
  C rh=1 no deadline  42.0 /  86.0               -- the unaligned control
  D rh=1 + deadline, Latent+CEM: CRASH in _MetricCost's I3 branch

Three things follow.

1. SEMANTICS. min-over-reachable-prefix is not the objective the metric grades.
   held-at-end asks where the arm IS on the graded step; the min rewards merely
   passing near the goal at any reachable chunk, and 42.0 -> 24.0 @0.05 is what
   that costs. Score the chunk that LANDS on the deadline instead (index cr-1).
   Both are kept behind ``I3_MODE`` (default ``deadline``) so the choice is
   measured rather than argued -- ``min`` reproduces the previous behaviour.

2. FRAME AXIS. ``p_raw`` is (B, S, T, D) with T = off + P: the WM re-predicts
   the conditioning history, so with 3 history frames and P=5 plan blocks the
   axis is 6 long, not 5. The plan chunks are the LAST P. The old guard compared
   chunks_remaining against T, which fires I3 one replan EARLY -- at rh=5 the
   count reaches 5 at the second replan, and 5 < 6 would have moved reported
   numbers had a deadline ever been set on that protocol. Compare against P,
   taken from ``action_candidates.shape[-2]``, and offset every index by T - P.
   LIPSolver._V needs no offset: its trajectory is the P plan chunks exactly.

3. CRASH. The old branch built goal as ``unsqueeze(0).expand(J, *goal.shape)``
   and then hit the generic ``goal.unsqueeze(1)`` normalisation below, giving
   (J,1,B,D) against pred (J,B,S,D). Normalise against one candidate first, then
   broadcast over the prefix axis.
"""
import ast
import shutil
import sys

LIP = "/workspace/swm_cem/stable_worldmodel/solver/lip.py"
EVW = "/workspace/swm_cem/scripts/plan/eval_wm.py"

# ---------------------------------------------------------------- lip.py mode
LIP_OLD_HEAD = '''_LIP_PROBE_N = 0
'''
LIP_NEW_HEAD = '''_LIP_PROBE_N = 0

# I3 readout (see LIPSolver._V). 'deadline' scores the chunk that lands on the
# graded step; 'min' scores the best still-reachable chunk. Both are inert
# without plan_config.deadline. Env var, not config, so the sampling path in
# eval_wm.py reads the same switch.
_I3_MODE = os.environ.get("I3_MODE", "deadline").lower()
'''

LIP_OLD_V = '''            _cr = getattr(self, "i3_chunks_remaining", None)
            _H = x.shape[1]
            if _cr is not None and 0 < int(_cr) < _H:
                _n = int(_cr)
                if not getattr(self, "_i3_said", False):
                    print(f"[I3] {_n}/{_H} chunks reachable -> argmin over "
                          f"prefix (vframes={self.vframes})", flush=True)
                    self._i3_said = True
                return torch.stack(
                    [self._V_traj(x[:, :j + 1], zg) for j in range(_n)],
                    dim=0).min(dim=0).values
            return self._V_traj(x, zg)
'''
LIP_NEW_V = '''            _cr = getattr(self, "i3_chunks_remaining", None)
            _H = x.shape[1]
            if _cr is not None and 0 < int(_cr) < _H:
                _n = int(_cr)
                # 'deadline': the one chunk that lands on the graded step.
                # 'min': the best still-reachable chunk (measured worse @0.05).
                _js = [_n - 1] if _I3_MODE == "deadline" else list(range(_n))
                if not getattr(self, "_i3_said", False):
                    print(f"[I3] mode={_I3_MODE} {_n}/{_H} chunks reachable -> "
                          f"chunks {_js} (vframes={self.vframes})", flush=True)
                    self._i3_said = True
                return torch.stack(
                    [self._V_traj(x[:, :j + 1], zg) for j in _js],
                    dim=0).min(dim=0).values
            return self._V_traj(x, zg)
'''

# ------------------------------------------------------------- eval_wm.py I3
EVW_OLD = '''                        _cr = getattr(self, "i3_chunks_remaining", None)
                        _T = p_raw.shape[-2]
                        if not getattr(self, "_i3_dbg", False):
                            print(f"[I3-dbg] p_raw {tuple(p_raw.shape)} "
                                  f"chunk_axis(-2)={_T} chunks_remaining={_cr} "
                                  f"_m={_m}", flush=True)
                            self._i3_dbg = True
                        _i3 = _cr is not None and int(_cr) < _T
                        if _i3:
                            pred = torch.stack([
                                torch.cat([p_raw[..., max(0, _j - _m + 1 + k), :]
                                           for k in range(_m)], dim=-1)
                                for _j in range(int(_cr))], dim=0)
                            goal = goal.unsqueeze(0).expand(pred.shape[0], *goal.shape)
                            if not getattr(self, "_i3_said", False):
                                print(f"[I3] {_cr}/{_T} chunks reachable -> argmin over prefix", flush=True)
                                self._i3_said = True
                        else:
                            _i3 = False
                    if goal.ndim < pred.ndim:
                        goal = goal.unsqueeze(1)
                    goal = goal.expand_as(pred)
'''
EVW_NEW = '''                        # The frame axis is NOT the plan. The WM re-predicts
                        # its conditioning history, so p_raw is T = off + P
                        # frames long for P plan blocks (6 vs 5 here) and the
                        # plan chunks are the LAST P: chunk j sits at off + j.
                        # chunks_remaining counts PLAN chunks, so it must be
                        # compared against P, never T -- the T comparison fires
                        # I3 one replan early, inside the reported rh=5
                        # protocol (count 5, T 6).
                        _cr = getattr(self, "i3_chunks_remaining", None)
                        _T = p_raw.shape[-2]
                        _P = int(action_candidates.shape[-2])
                        _off = _T - _P
                        if not getattr(self, "_i3_dbg", False):
                            print(f"[I3-dbg] p_raw {tuple(p_raw.shape)} T={_T} "
                                  f"plan={_P} off={_off} chunks_remaining={_cr} "
                                  f"_m={_m}", flush=True)
                            self._i3_dbg = True
                        _i3 = _cr is not None and 0 < int(_cr) < _P
                        if _i3:
                            _n = int(_cr)
                            _js = ([_n - 1] if _I3_MODE == 'deadline'
                                   else list(range(_n)))
                            pred = torch.stack([
                                torch.cat(
                                    [p_raw[..., max(0, _off + _j - _m + 1 + k), :]
                                     for k in range(_m)], dim=-1)
                                for _j in _js], dim=0)
                            if not getattr(self, "_i3_said", False):
                                print(f"[I3] mode={_I3_MODE} {_n}/{_P} chunks "
                                      f"reachable (T={_T} off={_off}) -> frames "
                                      f"{[_off + _j for _j in _js]}", flush=True)
                                self._i3_said = True
                    if locals().get("_i3", False):
                        # pred gained a leading prefix axis: normalise the goal
                        # against ONE candidate, then broadcast over that axis.
                        if goal.ndim < pred.ndim - 1:
                            goal = goal.unsqueeze(1)
                        goal = goal.unsqueeze(0).expand_as(pred)
                    else:
                        if goal.ndim < pred.ndim:
                            goal = goal.unsqueeze(1)
                        goal = goal.expand_as(pred)
'''

EVW_OLD_MIN = '''                    if locals().get("_i3", False):
                        # min over prefix candidates: the best reachable point
                        mcost = mcost.min(dim=0).values
'''
EVW_NEW_MIN = '''                    if locals().get("_i3", False):
                        # collapse the prefix axis. 'deadline' stacked a single
                        # candidate, so this is the identity there; 'min' takes
                        # the best still-reachable chunk.
                        mcost = mcost.min(dim=0).values
'''

EVW_OLD_IMP = '''            class _MetricCost(torch.nn.Module):
'''
EVW_NEW_IMP = '''            # I3 readout, shared with LIPSolver._V (see stable_worldmodel/
            # solver/lip.py): 'deadline' scores the chunk landing on the graded
            # step, 'min' the best still-reachable chunk.
            _I3_MODE = os.environ.get('I3_MODE', 'deadline').lower()

            class _MetricCost(torch.nn.Module):
'''


def patch(path, old, new, label, backup=True):
    src = open(path).read()
    if old not in src:
        if new.strip().splitlines()[0].strip() in src:
            print(f"[skip] {label}: already patched")
            return
        sys.exit(f"[FATAL] {label}: anchor not found in {path}")
    if backup:
        shutil.copy(path, path + ".pre_i3sem")
    open(path, "w").write(src.replace(old, new, 1))
    print(f"[ok] {label}")


patch(LIP, LIP_OLD_HEAD, LIP_NEW_HEAD, "lip.py: I3_MODE switch")
patch(LIP, LIP_OLD_V, LIP_NEW_V, "lip.py: _V mode-aware", backup=False)
patch(EVW, EVW_OLD_IMP, EVW_NEW_IMP, "eval_wm.py: I3_MODE switch")
patch(EVW, EVW_OLD, EVW_NEW, "eval_wm.py: axis + goal broadcast", backup=False)
patch(EVW, EVW_OLD_MIN, EVW_NEW_MIN, "eval_wm.py: collapse comment", backup=False)

for p in (LIP, EVW):
    ast.parse(open(p).read())
    assert "import os" in open(p).read(), f"{p}: os not imported"
print("[ok] both files parse, os available")

lsrc, esrc = open(LIP).read(), open(EVW).read()
assert "_I3_MODE" in lsrc and "_I3_MODE" in esrc
assert "int(_cr) < _P" in esrc and "int(_cr) < _T" not in esrc, "axis guard not fixed"
assert lsrc.count("def _V(") == 1 and lsrc.count("def _V_traj(") == 1
print("[ok] guard compares against the plan length; both readouts mode-aware")
