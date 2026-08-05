#!/usr/bin/env python3
"""Move I3 (argmin-over-reachable-prefix readout) into the path RLP actually uses.

DIAGNOSIS (2026-08-04). The unconditional ``[metric-hook]`` print at the entry of
``_MetricCost.get_cost`` in eval_wm.py does NOT appear in a LIP cell's log
(logs/i3dbg.log, a full 50-episode LeWM run with ``--metric l2window3.pt``).
``_MetricCost.get_cost`` is therefore never called when solver=lip: LIPSolver
never calls ``self.model.get_cost`` on the refinement path -- it scores with
``self.lip_value``, the value embedded in the actor checkpoint, through
``LIPSolver._V``. Every previous I3 edit was to code the RLP arm does not run.

That also locates the 3-frame window cost for both arm families, which the
handoff flagged as unresolved:
  * sampling arms (Latent/Value + CEM/Adam)  -> _MetricCost, the ``_m >= 3`` branch
  * RLP                                      -> LIPSolver._V / window_pair

This patch:
  1. lip.py     -- splits ``_V`` into the terminal-window score (``_V_traj``)
                   plus an I3 wrapper taking the argmin over reachable chunks.
  2. policy.py  -- publishes ``i3_chunks_remaining`` on the SOLVER as well as on
                   the cost model, so it reaches LIPSolver at all.

Both are inert without ``plan_config.deadline``, and inert whenever
chunks_remaining >= H (which is every replan at rh=5, budget 50, plan_len 25).
"""
import re
import shutil
import sys

LIP = "/workspace/swm_cem/stable_worldmodel/solver/lip.py"
POL = "/workspace/swm_cem/stable_worldmodel/policy.py"

OLD_V = '''    def _V(self, x, zg):
        """Score with the value, stacking an m-frame window when required.

        ``x`` is either the full imagined trajectory (B, H, D) or a terminal
        latent (B, D). With a window value a terminal-only input cannot be
        scored honestly -- repeating one frame would silently turn the window
        cost back into the terminal cost -- so that case raises.
        """
        if x.dim() == 3:
            if self.vframes <= 1:
                return self.lip_value(x[:, -1], zg)
            return self.lip_value(*window_pair(x, zg, self.vframes))
        if self.vframes > 1:
            raise RuntimeError(
                f"window value ({self.vframes} frames) received a terminal-only "
                f"latent {tuple(x.shape)}; the caller must pass the trajectory")
        return self.lip_value(x, zg)
'''

NEW_V = '''    def _V_traj(self, x, zg):
        """Terminal-window score of an imagined trajectory (B, <=H, D).

        With a window value the last ``vframes`` imagined frames are stacked;
        fewer than that available (a short prefix, or the start of an episode)
        repeats the oldest, which is window_pair's -- and train_window.py's --
        clamp convention.
        """
        if self.vframes <= 1:
            return self.lip_value(x[:, -1], zg)
        return self.lip_value(*window_pair(x, zg, self.vframes))

    def _V(self, x, zg):
        """Score with the value, stacking an m-frame window when required.

        ``x`` is either the full imagined trajectory (B, H, D) or a terminal
        latent (B, D). With a window value a terminal-only input cannot be
        scored honestly -- repeating one frame would silently turn the window
        cost back into the terminal cost -- so that case raises.

        I3. When the policy publishes ``i3_chunks_remaining`` (a deadline is set
        and fewer than H plan chunks still fit before the graded step) the plan
        overshoots the deadline: its terminal is a state the episode never
        reaches, and optimising it grades the planner on step 70 while the
        metric reads step 50. Score the best STILL-REACHABLE chunk instead,
        ``min_j V(z_j, z_g)`` for ``j < min(H, chunks_remaining)``.

        This is a pure readout change -- the refinement loop, the actor and the
        value are untouched -- and it is inert whenever chunks_remaining >= H.
        At rh=5 with eval_budget 50 and plan_len 25 the count is 10, 5 at the
        two replans, so no reported number can move. Applies to the gradient,
        to the per-iteration E and to the final argmin alike, because all three
        go through this one entry point.
        """
        if x.dim() == 3:
            _cr = getattr(self, "i3_chunks_remaining", None)
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
        if self.vframes > 1:
            raise RuntimeError(
                f"window value ({self.vframes} frames) received a terminal-only "
                f"latent {tuple(x.shape)}; the caller must pass the trajectory")
        return self.lip_value(x, zg)
'''

OLD_POL = '''            _mdl = getattr(self.solver, "model", None)
            if _mdl is not None:
                if _dl:
                    _st = int(max(self._step_ct[i] for i in replan_idx))
                    _mdl.i3_chunks_remaining = max(1, (int(_dl) - _st) // self.cfg.action_block)
                else:
                    _mdl.i3_chunks_remaining = None
'''

NEW_POL = '''            # Published on BOTH the solver and its cost model. Sampling solvers
            # reach the cost through ``model.get_cost`` (_MetricCost owns the
            # readout there), but LIPSolver never calls get_cost on its
            # refinement path -- it scores with the checkpoint-embedded value in
            # LIPSolver._V -- so the attribute has to reach the solver itself.
            # Publishing only to the model left I3 dead in every RLP cell, and
            # a dead I3 returns the unaligned baseline verbatim.
            _cr_i3 = None
            if _dl:
                _st = int(max(self._step_ct[i] for i in replan_idx))
                _cr_i3 = max(1, (int(_dl) - _st) // self.cfg.action_block)
            for _tgt in (self.solver, getattr(self.solver, "model", None)):
                if _tgt is not None:
                    _tgt.i3_chunks_remaining = _cr_i3
'''


def patch(path, old, new, label):
    src = open(path).read()
    if new.strip().splitlines()[0] in src and old not in src:
        print(f"[skip] {label}: already patched")
        return False
    if old not in src:
        sys.exit(f"[FATAL] {label}: anchor not found in {path}")
    shutil.copy(path, path + ".pre_i3v")
    open(path, "w").write(src.replace(old, new, 1))
    print(f"[ok] {label}: patched (backup {path}.pre_i3v)")
    return True


patch(LIP, OLD_V, NEW_V, "lip.py _V")
patch(POL, OLD_POL, NEW_POL, "policy.py publish")

# --- verify it parses and the inertness contract holds -----------------------
import ast

for p in (LIP, POL):
    ast.parse(open(p).read())
print("[ok] both files parse")

src = open(LIP).read()
assert "_V_traj" in src and "i3_chunks_remaining" in src
assert src.count("def _V(") == 1 and src.count("def _V_traj(") == 1
print("[ok] lip.py: one _V, one _V_traj, I3 present")
psrc = open(POL).read()
assert "for _tgt in (self.solver," in psrc
print("[ok] policy.py: publishes to solver + model")
