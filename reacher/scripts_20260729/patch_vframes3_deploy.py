"""Fix the DEPLOY path for m-frame window values in LIPSolver.

Bug found 2026-07-31: training stacked correctly, but only 2 of ~9 deploy call
sites were routed through the window helper, so evaluation died with
`mat1 and mat2 shapes cannot be multiplied (50x192 and 576x256)` -- a 192-d
terminal frame handed to a 576-d window value. Every wave-A card read 0.0.
The 27 trained actors are UNAFFECTED (their training path was correct); only
the solver needed fixing, so this costs an eval re-run, not a retrain.

Two changes, both structural rather than per-site so a missed site cannot recur:

1. `rollT` defaults to `rollout_traj` (which `rollout_terminal` was already just
   slicing with `[:, -1]`), so the full trajectory reaches every scoring site.
2. One adapter, `self._V(x, zg)`, replaces every `self.lip_value(...)` call:
   3-D input is window-stacked with the goal tiled; 2-D input passes through
   when the value is single-frame, and raises when it is not -- silence there
   would mean a window value being fed a terminal frame, which is exactly the
   bug this patch exists to remove.

vframes is now read from the value blob's declared `window_frames` (written by
train_window.py) instead of being inferred from checkpoint fields that may be
absent.
"""

import ast
import re

P = "/workspace/swm_cem/stable_worldmodel/solver/lip.py"
s = open(P).read()
if "def _V(" in s:
    print("already patched")
    raise SystemExit

# ---------------------------------------------------------------- 1. vframes from the blob
OLD_DET = re.search(r"\n\s*_vd = int\(getattr\(self\.lip_value.*?\n(?:.*?\n)*?.*?vframes.*?\n(?:.*?flush=True\)\n)?", s)
if OLD_DET:
    s = s.replace(OLD_DET.group(0), "\n")
OLD_LOAD = 'self.lip_value = load_metric(ck["value"], device=self.device)'
NEW_LOAD = '''self.lip_value = load_metric(ck["value"], device=self.device)
        # m-frame window values declare window_frames in their arch (see
        # train_window.py). Read it from the blob rather than inferring widths.
        _vb = torch.load(ck["value"], map_location="cpu", weights_only=False)
        self.vframes = int((_vb.get("arch") or {}).get("window_frames", 1))
        if self.vframes > 1:
            print(f"[vframes] window value: {self.vframes} frames", flush=True)'''
assert s.count(OLD_LOAD) == 1, f"load anchor x{s.count(OLD_LOAD)}"
s = s.replace(OLD_LOAD, NEW_LOAD)

# ---------------------------------------------------------------- 2. the adapter
ADAPTER = '''
    def _V(self, x, zg):
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
m = re.search(r"\n(    def [a-z_]+\(self[^\n]*\n)", s[s.index("class LIPSolver"):])
assert m, "no method found in LIPSolver"
ins = s.index("class LIPSolver") + m.start(1)
s = s[:ins] + ADAPTER + s[ins:]

# ---------------------------------------------------------------- 3. rollT returns trajectories
s = s.replace("rollT = rollT or (lambda zh, ah, p: rollout_terminal(wm, zh, ah, p))",
              "# full trajectory (rollout_terminal was only its [:, -1] slice), so window\n"
              "        # values can stack the last m frames at every scoring site\n"
              "        rollT = rollT or (lambda zh, ah, p: rollout_traj(wm, zh, ah, p))")

# ---------------------------------------------------------------- 4. route every call site
s = s.replace("E = self.lip_value(term.float(), zg.repeat_interleave(C, dim=0)).view(B, C)",
              "E = self._V(term.float(), zg.repeat_interleave(C, dim=0)).view(B, C)")
s = s.replace("self.lip_value(*window_pair(traj, zg_r, self.vframes)).sum(), A_in)",
              "self._V(traj, zg_r).sum(), A_in)")
s = s.replace("E = self.lip_value(*window_pair(traj_f, zg_r, self.vframes))",
              "E = self._V(traj_f, zg_r)")
s = s.replace("scores = scores + self.lip_value(rollT(zh_c, ah_c, pert), zg_c)",
              "scores = scores + self._V(rollT(zh_c, ah_c, pert), zg_c)")
s = s.replace("Ef = self.lip_value(rollT(zh_c, ah_c, cf), zg_c).view(B, R * C)",
              "Ef = self._V(rollT(zh_c, ah_c, cf), zg_c).view(B, R * C)")
s = s.replace("e_imag = self.lip_value(z_imag, zg)", "e_imag = self._V(z_imag, zg)")

open(P, "w").write(s)
ast.parse(open(P).read())
left = len(re.findall(r"self\.lip_value\(", s))
print(f"lip.py deploy patched; remaining direct lip_value calls: {left} "
      f"(dino path + the vtraj per-step site are expected)")
print("VERIFY: a window-value LIP eval must now print [vframes] and produce a "
      "non-FAIL success_rate")
