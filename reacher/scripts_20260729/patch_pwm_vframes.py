"""Give train_pwm_ac.py the same 3-frame window-value support train_lip_ac.py has.

PWM cannot currently join the baseline table: it hard-asserts

    assert blob['latent_dim'] == c_td.latent_dim

which fails 576 != 192 against the 3-frame window value every other arm uses,
and it then calls teacher()/critic() on bare single frames at five sites. Point
it at a 1-frame value instead and it runs, but it is no longer the same cost
function as LIP -- so it would not be a baseline, it would be a different
experiment. Hence the port.

This mirrors train_lip_ac.py exactly rather than inventing a second convention:

  _wrow(idx)      window-stacks TD-CACHE rows the way train_window.py built
                  them (real windows for both state and goal, clamped at
                  episode starts). Used in critic_step.
  _wpair(traj,zg) stacks the last `vframes` IMAGINED frames and TILES the goal.
                  Used in the planning rollout, where only one goal frame
                  exists. train_window.py prints a [tiled-goal diag] measuring
                  exactly this train/deploy mismatch (7.02 against a 82.19
                  random-pair baseline), so the asymmetry is deliberate and
                  quantified, not an oversight.

NOT ported, because PWM already has it: pad-context. PWM builds
    z_hist = z0.unsqueeze(1).expand(-1, hs, -1);  a_hist = zeros
which is precisely the 1-frame deployment interface LIP's --pad-context
constructs. Adding a flag would have been a no-op.

The window lag is taken from the value's own arch and checked against the
action block, because consecutive imagined latents are action_block apart --
a lag mismatch would silently deploy a different window than was trained.
"""

import ast

P = "/workspace/swm_cem/scripts/plan/train_pwm_ac.py"
s = open(P).read()

if "_wpair" in s:
    print("already patched")
    raise SystemExit

# ---------------------------------------------------------------- 1. detect vframes
OLD = """    blob = torch.load(a.init_value, map_location='cpu', weights_only=False)
    assert blob['latent_dim'] == c_td.latent_dim, (
        f"init-value latent_dim {blob['latent_dim']} != cache {c_td.latent_dim}")
    arch = blob['arch']
    critic = load_metric(a.init_value, device=dev)"""

NEW = """    blob = torch.load(a.init_value, map_location='cpu', weights_only=False)
    _ld, _cd = int(blob['latent_dim']), int(c_td.latent_dim)
    assert _cd and _ld % _cd == 0 and _ld // _cd in (1, 3), (
        f"init-value width {_ld} is neither 1x nor 3x the cache dim {_cd}")
    vframes = _ld // _cd
    _wlag = int(blob['arch'].get('window_lag', 5)) if vframes > 1 else 1
    if vframes == 3:
        print(f'[vframes] 3-frame window value, lag {_wlag}', flush=True)
    arch = blob['arch']
    critic = load_metric(a.init_value, device=dev)
    Ztd, st_td = c_td.z, c_td.step_idx.numpy()

    def _wpair(traj, zg):
        \"\"\"Stack the last `vframes` imagined frames; tile the goal to match.\"\"\"
        if vframes <= 1:
            return traj[:, -1], zg
        cols = []
        for k in range(vframes - 1, -1, -1):
            i = traj.shape[1] - 1 - k
            cols.append(traj[:, i] if i >= 0 else traj[:, 0])
        return torch.cat(cols, dim=-1), zg.repeat(*([1] * (zg.dim() - 1)), vframes)

    def _wrow(idx):
        \"\"\"Window-stack TD-cache rows exactly as train_window.py builds them.\"\"\"
        if vframes <= 1:
            return Ztd[idx]
        st = torch.as_tensor(st_td)
        cols = []
        for k in range(vframes - 1, -1, -1):
            off = k * _wlag
            j = idx - off
            j = torch.where(st[idx] < off, idx - st[idx], j)
            cols.append(Ztd[j])
        return torch.cat(cols, dim=-1)"""

assert s.count(OLD) == 1, f"init-value anchor x{s.count(OLD)}"
s = s.replace(OLD, NEW)

# ---------------------------------------------------------------- 2. critic step
OLD_D = """        with torch.no_grad():
            d_next = teacher(z_tn, z_g)"""
NEW_D = """        with torch.no_grad():
            d_next = (teacher(_wrow(b['tn_idx']).to(dev), _wrow(b['g_idx']).to(dev))
                      if vframes > 1 else teacher(z_tn, z_g))"""
assert s.count(OLD_D) == 1, f"d_next anchor x{s.count(OLD_D)}"
s = s.replace(OLD_D, NEW_D)

OLD_L = "        loss = _expectile_loss(critic(z_t, z_g) - tgt, tau, a.huber_beta)"
NEW_L = """        loss = _expectile_loss(
            (critic(_wrow(b['t_idx']).to(dev), _wrow(b['g_idx']).to(dev))
             if vframes > 1 else critic(z_t, z_g)) - tgt, tau, a.huber_beta)"""
assert s.count(OLD_L) == 1, f"critic-loss anchor x{s.count(OLD_L)}"
s = s.replace(OLD_L, NEW_L)

# ---------------------------------------------------------------- 3. planning rollout
OLD_J1 = "                J = J - disc * teacher(z, zg)"
NEW_J1 = "                J = J - disc * teacher(*_wpair(traj, zg))"
assert s.count(OLD_J1) == 1, f"dense-J anchor x{s.count(OLD_J1)}"
s = s.replace(OLD_J1, NEW_J1)

OLD_J2 = "            J = -disc * teacher(z, zg)"
NEW_J2 = "            J = -disc * teacher(*_wpair(traj, zg))"
assert s.count(OLD_J2) == 1, f"terminal-J anchor x{s.count(OLD_J2)}"
s = s.replace(OLD_J2, NEW_J2)

OLD_E = "                E = teacher(z, zg).mean().item()"
NEW_E = "                E = teacher(*_wpair(traj, zg)).mean().item()"
assert s.count(OLD_E) == 1, f"E-diag anchor x{s.count(OLD_E)}"
s = s.replace(OLD_E, NEW_E)

# ---------------------------------------------------------------- 4. saved width
OLD_S = "    save_metric(teacher.cpu(), 'td', c_td.latent_dim, arch, a.out_value)"
NEW_S = "    save_metric(teacher.cpu(), 'td', c_td.latent_dim * vframes, arch, a.out_value)"
assert s.count(OLD_S) == 1, f"save_metric anchor x{s.count(OLD_S)}"
s = s.replace(OLD_S, NEW_S)

open(P, "w").write(s)
ast.parse(open(P).read())
print("train_pwm_ac.py: vframes-aware (critic rows, planning rollout, saved "
      "width); syntax OK")
