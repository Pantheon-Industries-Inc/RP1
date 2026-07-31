"""Add --ckpt-every to train_lip_ac.py: periodic actor snapshots so a selection
pass can EARLY-STOP ON HELD-AT-END rather than on the training objective.

Why: corr(E_final, HELD) = +0.583 over 20 actors -- driving the imagined cost
lower reliably makes real performance worse, because the world model's 25-step
error (~0.11 rad) exceeds the 0.05 rad tolerance. So neither "train to
convergence" nor "take the lowest loss" is a valid stopping rule; the only
trustworthy signal is held-at-end itself.

The snapshot must be byte-compatible with the final save or the solver cannot
load it, so this REUSES the exact dict the script writes at the end (same keys,
same arch fields) rather than an approximation. Two deviations from the final
save, both deliberate:
  * `net.state_dict()` is deep-copied to CPU instead of calling `net.cpu()` --
    the final save moves the module permanently, which mid-training would put
    the model on the wrong device;
  * `net.eval()`/`train()` is not toggled, since the planner net has no
    train-mode-dependent layers here and toggling mid-loop risks side effects.
"""

import ast
import re

P = "/workspace/swm_cem/scripts/plan/train_lip_ac.py"
s = open(P).read()
if "--ckpt-every" in s:
    print("already patched")
    raise SystemExit

OLD_ARG = '    p.add_argument("--steps", type=int, default=8000, help="actor steps")'
NEW_ARG = '''    p.add_argument("--steps", type=int, default=8000, help="actor steps")
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="snapshot the actor every N steps to <out>.step{N}.pt so a "
                        "selection pass can early-stop on HELD-at-end rather than on "
                        "the training objective (which anti-predicts it, r=+0.58)")'''
assert s.count(OLD_ARG) == 1, f"arg anchor x{s.count(OLD_ARG)}"
s = s.replace(OLD_ARG, NEW_ARG)

# lift the final save dict verbatim and reuse it for snapshots
m = re.search(r"    torch\.save\(\{\"kind\": kind.*?\}, a\.out\)\n", s, re.S)
assert m, "final save block not found"
final = m.group(0)
snap_body = (final
             .replace('net.cpu().state_dict()',
                      '{k: v.detach().cpu() for k, v in net.state_dict().items()}')
             .replace('}, a.out)', '}, _snap)')
             .replace('    torch.save(', '            torch.save('))
# re-indent the continuation lines of the dict literal to match
snap_body = "\n".join(
    ("        " + ln if ln.startswith("                ") else ln)
    for ln in snap_body.split("\n"))

OLD_LOOP = "        e_first, e_final, expand = actor_step()"
NEW_LOOP = ('        e_first, e_final, expand = actor_step()\n'
            '        if a.ckpt_every and (step + 1) % a.ckpt_every == 0 \\\n'
            '                and (step + 1) < a.steps:\n'
            '            _snap = f"{a.out}.step{step + 1}.pt"\n'
            '            kind = ("lip3" if a.arch == "traj" else\n'
            '                    "lip4r" if a.arch == "v4r" else\n'
            '                    "lip4" if a.arch == "v4" else\n'
            '                    ("lip" if a.feed == "none" else "lip2"))\n'
            + snap_body)
assert s.count(OLD_LOOP) == 1, f"loop anchor x{s.count(OLD_LOOP)}"
s = s.replace(OLD_LOOP, NEW_LOOP)
open(P, "w").write(s)
ast.parse(open(P).read())
print("train_lip_ac.py: --ckpt-every added, snapshot dict lifted from the final save; syntax OK")
