"""Add --align-mode to train_lip_ac.py: four ways to handle target alignment.

WHY. RLP's loss scores only the plan terminal, V(z_H, z_g). Deployed at
receding_horizon=1 only the FIRST block executes, and a terminal cost does not
constrain the path -- so the executed prefix is the least-determined part of the
plan. Measured consequence at rh=1 without deadline alignment: one RLP cell fell
100.0 -> 86.0 @0.1 and 58.0 -> 38.0 @0.05.

All four modes reduce to a single question -- which timesteps of the final
refined rollout enter the loss:

  terminal      z_H only. The control, so a 6000-step run isolates the step-count
                change from the alignment change.
  dense         sum_t gamma^(t+1) V(z_t, z_g) / sum gamma^(t+1), all H blocks.
                Every executed prefix gets direct supervision. PWM's dense-vs-
                terminal ablation was worth +16.8 (lejepa) and +64.9 (pldm) at
                matched budget, and terminal-only collapsed to 0.3 at 8x steps
                while dense improved -- so this is the best-evidenced of the four.
  prefix        V(z_1, z_g), the block that actually executes at rh=1, plus the
                terminal. Targeted version of dense with one extra term.
  randhorizon   V(z_h, z_g) with h ~ U{1..H} drawn per sample, plus the terminal.
                Teaches every prefix to land, so any truncation point is valid.
                Needs no positional-embedding change: the actor still emits H
                blocks, only the scored index moves.

WINDOWING. With vframes=3 the value consumes a 3-frame window, so a per-step
cost cannot score tr[:, t] directly. _wsteps builds the window at step t from the
last vframes frames of [history, imagined[:t+1]] -- exactly how the window fills
at deploy as the rollout proceeds. With pad-context the history is z0 repeated,
so step 0 scores [z0, z0, z_1], which is what the deployed solver sees on its
first block. hs=3 and vframes=3 make the earliest index hs-2=1, so no padding
branch is needed.

Default align-mode=terminal and align-weight=0 leave behaviour bit-identical.
"""

import ast

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()

if "align_mode" in s:
    print("already patched")
    raise SystemExit

# ---------------------------------------------------------------- flags
OLD_ARG = '    p.add_argument("--mean-weight", type=float'
NEW_ARG = ('    p.add_argument("--align-mode", default="terminal",\n'
           '                   choices=["terminal", "dense", "prefix", "randhorizon"],\n'
           '                   help="which rollout timesteps enter the loss; see the "\n'
           '                        "patch docstring. terminal = historical behaviour.")\n'
           '    p.add_argument("--align-weight", type=float, default=0.0,\n'
           '                   help="weight on the alignment term (0 = off)")\n'
           '    p.add_argument("--mean-weight", type=float')
assert s.count(OLD_ARG) == 1, f"mean-weight anchor x{s.count(OLD_ARG)}"
s = s.replace(OLD_ARG, NEW_ARG)

# ---------------------------------------------------------------- per-step windowed cost
OLD_H = '''    def _wrow(idx):
        """Window-stack TD-cache rows exactly as train_window.py builds them."""'''
NEW_H = '''    def _wsteps(zh_, tr_, zg_):
        """Per-timestep windowed cost over the imagined rollout -> (B, H).

        The window at step t is the last `vframes` frames of
        [history, imagined[:t+1]], matching how it fills at deploy.
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

# ---------------------------------------------------------------- the alignment term
OLD_ST = "        _stack = torch.stack(e_path)"
NEW_ST = '''        _align = torch.zeros((), device=dev)
        if a.align_weight > 0 and a.align_mode != "terminal":
            _ps = _wsteps(zh, tr, zg)                      # (B, H) per-timestep
            _H = _ps.shape[1]
            if a.align_mode == "dense":
                _w = torch.tensor([a.gamma ** (t + 1) for t in range(_H)],
                                  device=dev, dtype=_ps.dtype)
                _align = (_ps * _w).sum(1).mean() / _w.sum()
            elif a.align_mode == "prefix":
                _align = _ps[:, 0].mean()                  # the block rh=1 executes
            else:                                          # randhorizon
                _h = torch.randint(0, _H, (_ps.shape[0],), device=dev)
                _align = _ps.gather(1, _h[:, None]).squeeze(1).mean()
            _align = a.align_weight * _align
        _stack = torch.stack(e_path)'''
assert s.count(OLD_ST) == 1, f"_stack anchor x{s.count(OLD_ST)}"
s = s.replace(OLD_ST, NEW_ST)

n = 0
for old, new in [
    ("loss = e_path[-1] + a.mean_weight * _stack.mean()",
     "loss = e_path[-1] + a.mean_weight * _stack.mean() + _align"),
    ("loss = e_path[-1] + a.mean_weight * (_wt * _stack).sum()",
     "loss = e_path[-1] + a.mean_weight * (_wt * _stack).sum() + _align"),
]:
    if old in s:
        s = s.replace(old, new)
        n += 1
assert n >= 1, "no loss assignment matched"

open(P, "w").write(s)
ast.parse(open(P).read())
print(f"train_lip_ac.py: --align-mode {{terminal,dense,prefix,randhorizon}} + "
      f"--align-weight, {n} loss branch(es); syntax OK")
print("defaults (terminal, 0.0) keep behaviour bit-identical")
