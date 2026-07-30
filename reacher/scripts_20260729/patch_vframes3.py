"""vframes=3 support: let LIP train against and deploy a 3-FRAME WINDOW value.

RECONSTRUCTED 2026-07-30 after losing volume 3cv5zzezm9. The original applied
cleanly to both pods and its smoke gate ("gradient alive: E_first -> E_final")
passed; this version re-derives the same contract from the window value's
declared interface rather than from the lost file, so VERIFY THE SMOKE GATE
before trusting any actor trained with it.

Contract (must match train_window.py and eval_wm.py's metric hook):
  * the value declares latent_dim = 3*D, so the hook infers _m = 3 and feeds the
    LAST 3 IMAGINED frames of the plan with the goal TILED 3x;
  * inside LIP the same stacking must be applied wherever V is queried, using
    the trajectory's own consecutive latents [z_{T-2}, z_{T-1}, z_T];
  * at the episode edge (fewer than 3 imagined frames yet) the oldest available
    frame is repeated, mirroring train_window.py's step-0 clamp and the
    planner's own padded starts.

train_lip_ac.py already detects vframes from the init-value's declared width
(1 or 2 previously); this widens that to 3 and stacks the TD cache the same way.
"""

import re

# ---------------------------------------------------------------- solver/lip.py
P1 = "/workspace/swm_cem/stable_worldmodel/solver/lip.py"
s = open(P1).read()
if "goal tiled" in s:
    print("lip.py: already patched")
else:
    OLD = "def stack_for_metric("
    assert OLD in s, "stack_for_metric not found -- vframes 1/2 support missing?"
    # widen the existing helper: add the m==3 branch (goal tiled 3x)
    m = re.search(r"def stack_for_metric\((.*?)\n(?=\ndef |\nclass )", s, re.S)
    assert m, "could not isolate stack_for_metric"
    body = m.group(0)
    assert "m == 3" not in body
    NEW_BRANCH = '''    if m == 3:
        # 3-frame window: [z_{T-2}, z_{T-1}, z_T] against the goal tiled 3x.
        # z_prev2 falls back to z_prev (and z_prev to z_term) at the plan edge,
        # matching train_window.py's episode-start clamp.
        p1 = z_prev if z_prev is not None else z_term
        p2 = z_prev2 if z_prev2 is not None else p1
        return (torch.cat([p2, p1, z_term], dim=-1),
                z_goal.repeat(*([1] * (z_goal.dim() - 1)), 3))
'''
    # insert the branch before the final return of the helper
    lines = body.rstrip().split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().startswith("return"):
            lines.insert(i, NEW_BRANCH.rstrip())
            break
    newbody = "\n".join(lines) + "\n"
    s = s.replace(body, newbody)
    # the helper needs a z_prev2 parameter (default None keeps every existing caller valid)
    s = re.sub(r"def stack_for_metric\(([^)]*?)\)",
               lambda mm: ("def stack_for_metric(" + mm.group(1) + ", z_prev2=None)"
                           if "z_prev2" not in mm.group(1) else mm.group(0)),
               s, count=1)
    open(P1, "w").write(s)
    print("lip.py: stack_for_metric m=3 branch + z_prev2 arg added")

# ---------------------------------------------------------------- train_lip_ac.py
P2 = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
t = open(P2).read()
if "vframes == 3" in t:
    print("train_lip_ac.py: already patched")
else:
    OLD_ASSERT = "assert _cd and _ld % _cd == 0 and _ld // _cd in (1, 2), ("
    NEW_ASSERT = "assert _cd and _ld % _cd == 0 and _ld // _cd in (1, 2, 3), ("
    assert t.count(OLD_ASSERT) == 1, "vframes assert anchor"
    t = t.replace(OLD_ASSERT, NEW_ASSERT)
    t = t.replace('f"init-value per-side width {_ld} is neither 1x nor 2x the TD cache "',
                  'f"init-value per-side width {_ld} is not 1x/2x/3x the TD cache "')

    # TD-side stacking for the window value: [z_{t-2L}, z_{t-L}, z_t], clamped
    OLD_TD = '''    def td_side(idx):
        """Stack a batch of TD-cache rows into the value's per-side input."""
        zc = Ztd[idx]
        return zc if prev_td is None else torch.cat([zc, zc - Ztd[prev_td[idx]]], -1)'''
    NEW_TD = '''    def td_side(idx):
        """Stack a batch of TD-cache rows into the value's per-side input."""
        zc = Ztd[idx]
        if vframes == 3:
            # window: oldest -> newest, episode-start clamp exactly as
            # train_window.py builds the training rows
            return torch.cat([Ztd[prev2_td[idx]], Ztd[prev_td[idx]], zc], -1)
        return zc if prev_td is None else torch.cat([zc, zc - Ztd[prev_td[idx]]], -1)'''
    assert t.count(OLD_TD) == 1, "td_side anchor"
    t = t.replace(OLD_TD, NEW_TD)

    # build prev2_td alongside prev_td
    OLD_PREV = '''        prev_td = torch.from_numpy(_pv).long()'''
    NEW_PREV = '''        prev_td = torch.from_numpy(_pv).long()
    prev2_td = None
    if vframes == 3:
        import numpy as _np
        _n = len(Ztd)
        _lag = atrest_lag if atrest_lag else fs
        for _k, _name in ((1, "prev_td"), (2, "prev2_td")):
            _off = _k * _lag
            _p = _np.arange(_n) - _off
            _first = c_td.step_idx.numpy() < _off
            _p[_first] = (_np.arange(_n) - c_td.step_idx.numpy())[_first]
            if _k == 1:
                prev_td = torch.from_numpy(_p).long()
            else:
                prev2_td = torch.from_numpy(_p).long()
        print(f"[window] TD cache stacked 3 frames at lag {_lag}", flush=True)'''
    assert t.count(OLD_PREV) == 1, "prev_td anchor"
    t = t.replace(OLD_PREV, NEW_PREV)

    # V() must pass the second-previous imagined frame
    t = t.replace("def prev_frame(", "def prev_frame2(traj, z0):\n"
                  "    \"\"\"Second-previous imagined frame, clamped at the plan start.\"\"\"\n"
                  "    if traj.shape[1] >= 3:\n"
                  "        return traj[:, -3]\n"
                  "    return z0\n\n\ndef prev_frame(", 1)
    open(P2, "w").write(t)
    print("train_lip_ac.py: vframes=3 wired (assert, td_side, prev2_td, prev_frame2)")

import ast

for p in (P1, P2):
    ast.parse(open(p).read())
print("syntax OK -- NOW RUN THE 20-STEP SMOKE AND CONFIRM THE GRADIENT IS ALIVE")
