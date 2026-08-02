"""Give LIP a dense per-timestep path term, mirroring PWM's --dense.

WHY. PWM's ablation isolated the objective as the single largest lever on this
task. At matched budget (1000 steps), switching PWM from dense to LIP's terminal
objective cost 16.8 points on lejepa and 64.9 on pldm; at 8000 steps terminal
collapsed outright (0.3 / 1.8) while dense improved (85.9 / 99.0). LIP is
terminal with respect to the ROLLOUT at every refinement, and LIP also degrades
with more steps (1000 beat 3000 in all 12 pairings) -- the same signature.

WHAT IS AND IS NOT ALREADY THERE. LIP's mean_weight averages e_path over
REFINEMENT ITERATIONS k -- "be good after every refinement" -- and each element
is still the TERMINAL cost of that iteration's rollout. So LIP has no
per-timestep supervision at all. It does compute per-step values (vtraj), but
they are .detach()ed and fed to the learned update rule as a feature: LIP
observes the path costs without ever optimising them.

WHAT THIS ADDS.

    dense = sum_t gamma^(t+1) V(window_t, z_g) / sum_t gamma^(t+1)
    loss  = e_path[-1] + mean_weight * <path over k>  +  dense_weight * dense

with dense_weight defaulting to 0.0, so the current behaviour is bit-identical
unless the flag is passed.

WINDOWING IS THE SUBTLETY. With vframes=3 the value consumes a 3-frame window,
so a per-step cost cannot just score tr[:, t]. _wsteps builds the window at step
t from the last `vframes` frames of [history, imagined[:t+1]] -- exactly how the
window fills at deploy as the rollout proceeds. With pad-context the history is
z0 repeated, so step 0 scores [z0, z0, z_1], which is precisely what the
deployed solver sees on its first block. No padding branch is needed because
hs=3 and vframes=3 make the earliest index hs-2 = 1.

The goal is tiled to match, consistent with _wpair and with the tiled-goal
diagnostic train_window.py already reports.
"""

import ast

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()

if "dense_weight" in s:
    print("already patched")
    raise SystemExit

# ---------------------------------------------------------------- flag
OLD_ARG = '    p.add_argument("--mean-weight", type=float'
NEW_ARG = ('    p.add_argument("--dense-weight", type=float, default=0.0,\n'
           '                   help="weight on a per-TIMESTEP path cost over the rollout "\n'
           '                        "(PWM-style dense objective). 0.0 = terminal-only, the "\n'
           '                        "historical behaviour. Distinct from --mean-weight, "\n'
           '                        "which averages terminal cost over refinement iters.")\n'
           '    p.add_argument("--mean-weight", type=float')
assert s.count(OLD_ARG) == 1, f"mean-weight arg anchor x{s.count(OLD_ARG)}"
s = s.replace(OLD_ARG, NEW_ARG)

# ---------------------------------------------------------------- helper next to _wpair
OLD_H = '''    def _wrow(idx):
        """Window-stack TD-cache rows exactly as train_window.py builds them."""'''
NEW_H = '''    def _wsteps(zh_, tr_, zg_):
        """Per-timestep windowed cost over the imagined rollout -> (B, H).

        The window at step t is the last `vframes` frames of
        [history, imagined[:t+1]], which is how the window actually fills at
        deploy as the rollout proceeds. hs=3 and vframes=3 make the earliest
        index hs-2=1, so no padding branch is needed.
        """
        S = torch.cat([zh_, tr_], dim=1)
        hs_, H_ = zh_.shape[1], tr_.shape[1]
        outs = []
        for t in range(H_):
            if vframes <= 1:
                w, g = S[:, hs_ + t], zg_
            else:
                cols = [S[:, hs_ + t - k] for k in range(vframes - 1, -1, -1)]
                w = torch.cat(cols, dim=-1)
                g = zg_.repeat(*([1] * (zg_.dim() - 1)), vframes)
            outs.append(teacher(w, g))
        return torch.stack(outs, dim=1)

    def _wrow(idx):
        """Window-stack TD-cache rows exactly as train_window.py builds them."""'''
assert s.count(OLD_H) == 1, f"_wrow anchor x{s.count(OLD_H)}"
s = s.replace(OLD_H, NEW_H)

# ---------------------------------------------------------------- dense term
OLD_ST = "        _stack = torch.stack(e_path)"
NEW_ST = '''        _dense = torch.zeros((), device=dev)
        if a.dense_weight > 0:
            _ps = _wsteps(zh, tr, zg)                     # (B, H) per-timestep cost
            _dw = torch.tensor([a.gamma ** (t + 1) for t in range(_ps.shape[1])],
                               device=dev, dtype=_ps.dtype)
            _dense = a.dense_weight * (_ps * _dw).sum(1).mean() / _dw.sum()
        _stack = torch.stack(e_path)'''
assert s.count(OLD_ST) == 1, f"_stack anchor x{s.count(OLD_ST)}"
s = s.replace(OLD_ST, NEW_ST)

n = 0
for old, new in [
    ("loss = e_path[-1] + a.mean_weight * _stack.mean()",
     "loss = e_path[-1] + a.mean_weight * _stack.mean() + _dense"),
    ("loss = e_path[-1] + a.mean_weight * (_wt * _stack).sum()",
     "loss = e_path[-1] + a.mean_weight * (_wt * _stack).sum() + _dense"),
]:
    if old in s:
        s = s.replace(old, new)
        n += 1
assert n >= 1, "no loss assignment matched"

open(P, "w").write(s)
ast.parse(open(P).read())
print(f"train_lip_ac.py: --dense-weight added, {n} loss branch(es) patched, "
      f"_wsteps windows per timestep; syntax OK")
print("default 0.0 -> behaviour unchanged unless the flag is passed")
