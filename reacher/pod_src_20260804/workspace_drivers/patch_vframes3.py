"""vframes=3: let LIP train against and deploy a 3-FRAME WINDOW value.

AUTHORED FRESH 2026-07-30 (the original was lost with volume 3cv5zzezm9; the
local repo's train_lip_ac.py/lip.py are the pristine single-frame versions, so
this is written against them rather than being a re-derivation).

Contract, fixed by train_window.py and eval_wm.py's metric hook:
  * the value declares latent_dim = 3*D; the hook infers _m = 3 and feeds the
    LAST 3 IMAGINED frames with the goal TILED 3x;
  * so everywhere LIP queries the value it must pass the same thing:
    [z_{T-2}, z_{T-1}, z_T] against z_goal repeated 3x;
  * at the plan edge (fewer than 3 imagined frames) the oldest available frame
    repeats -- mirroring train_window.py's episode-start clamp and the
    planner's own padded starts.

Both files get a single helper and every value call site routes through it, so
there is exactly one place where the stacking convention lives.

SMOKE GATE (run after patching, before trusting any actor):
  the 20-step run must print a live gradient AND the value must respond to
  window CONTENT -- a value that ignores the two older frames would silently
  reduce to the terminal cost and every "window" result would be a lie.
"""

import ast
import re

D_HELPER = '''

def window_pair(traj, z_goal, frames, z0=None):
    """(stacked_state, tiled_goal) for an m-frame window value.

    ``traj`` is (B, H, D) imagined latents, newest last. Fewer than ``frames``
    available => repeat the oldest (train_window.py's clamp). ``frames==1``
    returns the terminal frame unchanged, so single-frame values are untouched.
    """
    if frames <= 1:
        return traj[:, -1], z_goal
    cols = []
    for k in range(frames - 1, -1, -1):
        idx = traj.shape[1] - 1 - k
        if idx >= 0:
            cols.append(traj[:, idx])
        else:
            cols.append(traj[:, 0] if traj.shape[1] else z0)
    return torch.cat(cols, dim=-1), z_goal.repeat(*([1] * (z_goal.dim() - 1)), frames)
'''

# ---------------------------------------------------------------- solver/lip.py
P1 = "/workspace/swm_cem/stable_worldmodel/solver/lip.py"
s = open(P1).read()
if "def window_pair(" in s:
    print("lip.py: already patched")
else:
    # helper after the imports
    m = re.search(r"^(import|from)[^\n]*\n(?:(?:import|from)[^\n]*\n|\s*\n)*", s, re.M)
    assert m, "no import block"
    s = s[: m.end()] + D_HELPER + s[m.end():]

    # discover the value's frame count once, at load
    OLD_LOAD = 'self.lip_value = load_metric(ck["value"], device=self.device)'
    NEW_LOAD = ('self.lip_value = load_metric(ck["value"], device=self.device)\n'
                '        # m-frame window values declare latent_dim = m*D; every query below\n'
                '        # must stack that many imagined frames and tile the goal\n'
                '        _vd = int(getattr(self.lip_value, "latent_dim", 0) or 0)\n'
                '        self.vframes = max(1, _vd // int(ck.get("latent_dim", _vd or 1)))\n'
                '        if self.vframes > 1:\n'
                '            print(f"[vframes] window value: {self.vframes} frames", flush=True)')
    assert s.count(OLD_LOAD) == 1, f"load anchor x{s.count(OLD_LOAD)}"
    s = s.replace(OLD_LOAD, NEW_LOAD)

    # route every trajectory-based query through the helper
    SITES = [
        ("(gA,) = torch.autograd.grad(self.lip_value(traj[:, -1], zg_r).sum(), A_in)",
         "(gA,) = torch.autograd.grad(\n"
         "                    self.lip_value(*window_pair(traj, zg_r, self.vframes)).sum(), A_in)"),
        ("E = self.lip_value(traj_f[:, -1], zg_r)",
         "E = self.lip_value(*window_pair(traj_f, zg_r, self.vframes))"),
    ]
    for old, new in SITES:
        if old in s:
            s = s.replace(old, new)
    open(P1, "w").write(s)
    print("lip.py: window_pair helper + vframes detection + traj call sites")

# ---------------------------------------------------------------- train_lip_ac.py
P2 = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
t = open(P2).read()
if "vframes" in t:
    print("train_lip_ac.py: already patched")
else:
    # 1. accept a 3x-wide init value
    OLD_A = 'assert blob["latent_dim"] == c_td.latent_dim, "init-value latent dim mismatch"'
    NEW_A = ('_ld, _cd = int(blob["latent_dim"]), int(c_td.latent_dim)\n'
             '        assert _cd and _ld % _cd == 0 and _ld // _cd in (1, 3), (\n'
             '            f"init-value width {_ld} is neither 1x nor 3x the cache dim {_cd}")\n'
             '        vframes = _ld // _cd\n'
             '        if vframes == 3:\n'
             '            _wl = int(blob["arch"].get("window_lag", 5))\n'
             '            assert _wl == fs, (\n'
             '                f"window lag {_wl} != action_block {fs}: consecutive imagined "\n'
             '                "latents are action_block apart, so the deployed window would "\n'
             '                "not match the trained one")\n'
             '            print(f"[vframes] 3-frame window value, lag {_wl}", flush=True)')
    assert t.count(OLD_A) == 1, f"assert anchor x{t.count(OLD_A)}"
    t = t.replace(OLD_A, NEW_A)
    t = t.replace("        critic = build_metric(\"td\", c_td.latent_dim, arch).to(dev)",
                  "        vframes = 1\n        critic = build_metric(\"td\", c_td.latent_dim, arch).to(dev)")

    # 2. helpers: window stacking for trajectories and for TD-cache rows
    HELP = '''
    def _wpair(traj, zg):
        """Stack the last `vframes` imagined frames; tile the goal to match."""
        if vframes <= 1:
            return traj[:, -1], zg
        cols = []
        for k in range(vframes - 1, -1, -1):
            i = traj.shape[1] - 1 - k
            cols.append(traj[:, i] if i >= 0 else traj[:, 0])
        return torch.cat(cols, dim=-1), zg.repeat(*([1] * (zg.dim() - 1)), vframes)

    def _wrow(idx):
        """Window-stack TD-cache rows exactly as train_window.py builds them."""
        if vframes <= 1:
            return Ztd[idx]
        cols = []
        for k in range(vframes - 1, -1, -1):
            off = k * fs
            j = idx - off
            j = torch.where(torch.as_tensor(st_td)[idx] < off,
                            idx - torch.as_tensor(st_td)[idx], j)
            cols.append(Ztd[j])
        return torch.cat(cols, dim=-1)
'''
    ANCH = "    def critic_step("
    assert t.count(ANCH) == 1, "critic_step anchor"
    t = t.replace(ANCH, HELP + "\n" + ANCH)

    # 3. Ztd / st_td handles
    t = t.replace("    c_td = LatentCache.load(a.cache_td)",
                  "    c_td = LatentCache.load(a.cache_td)\n"
                  "    Ztd, st_td = c_td.z, c_td.step_idx.numpy()")

    # 4. route the value calls
    for old, new in [
        ("d_next = teacher(z_tn, z_g)", "d_next = teacher(_wrow(b['tn_idx']).to(dev), _wrow(b['g_idx']).to(dev)) if vframes > 1 else teacher(z_tn, z_g)"),
        ("(gA,) = torch.autograd.grad(teacher(traj[:, -1], zg).sum(), A_in)",
         "(gA,) = torch.autograd.grad(teacher(*_wpair(traj, zg)).sum(), A_in)"),
        ("E_feat = teacher(traj_f[:, -1], zg).detach()",
         "E_feat = teacher(*_wpair(traj_f, zg)).detach()"),
        ("e_path.append(teacher(zT, zg).mean())",
         "e_path.append(teacher(*_wpair(tr, zg)).mean())"),
    ]:
        if old in t:
            t = t.replace(old, new)

    # 5. critic trains on stacked rows too
    t = t.replace("loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)",
                  "loss = _expectile_loss(\n"
                  "            (critic(_wrow(b['t_idx']).to(dev), _wrow(b['g_idx']).to(dev))\n"
                  "             if vframes > 1 else critic(z_t, z_g)) - tgt, tau, a.huber_beta)")

    # 6. declare the right width on save
    t = t.replace('save_metric(teacher.cpu(), "td", c_td.latent_dim, arch, a.out_value)',
                  'save_metric(teacher.cpu(), "td", c_td.latent_dim * vframes, arch, a.out_value)')
    open(P2, "w").write(t)
    print("train_lip_ac.py: vframes wired (assert, _wpair, _wrow, call sites, save width)")

for p in (P1, P2):
    ast.parse(open(p).read())
print("syntax OK")
print("NEXT: run the smoke gate -- gradient must be alive AND the value must "
      "respond to window content (else it silently collapsed to the terminal cost)")
