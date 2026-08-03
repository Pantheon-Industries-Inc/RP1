"""Fix the DINO LIP action convention at both interfaces.

MEASURED (exp_action_convention.py, 192 contexts, held-out + train episodes):
the predictor's one-step-to-5-step error against the TRUE next latent is

    h:      1        2        3        4        5
    raw   0.1117   0.1480   0.1699   0.1735   0.1859     (+66% over the horizon)
    norm  0.0604   0.0653   0.0661   0.0669   0.0678     (+13%, and 2.7x lower at h=5)
    zero  0.1718   0.2300   0.2631   0.2613   0.2720

So the action embedder was trained on Z-SCORED blocks. Raw actions are barely
better than feeding ZERO actions -- at that scale the action pathway is nearly
inert, which is why every LIP cell sat at 70-72 regardless of amax / lr / steps.

Two sites, one false premise ("this WM wants raw actions", asserted in
train_lip_dino.py's docstring and inherited by my ports):

  1. rollout input: `rollout_*_dino(wm, toks, A * ast + amu)` -> pass A directly.
  2. return value: `_proposal_lip_dino` returned `A * ast + amu`, but
     policy.py:303 does `action = process['action'].inverse_transform(action)`,
     i.e. the policy un-normalizes the solver's output. LeWM's known-good
     `_proposal_lip` returns A un-scaled for exactly this reason; the dino path
     was double-scaling.

After this fix the dino path is convention-identical to LeWM's. amu5/ast5 stay
in the checkpoints (harmless provenance) but are no longer applied.

All LIP actors trained before this fix are invalid and must be retrained.
"""

import subprocess
import sys
from pathlib import Path

LIP = Path('/workspace/code/stable-worldmodel/stable_worldmodel/solver/lip.py')
TRS = [Path('/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac_dino.py'),
       Path('/workspace/code/stable-worldmodel/scripts/plan/train_lip_dino_lance.py')]


def apply(text, edits, tag):
    for old, new, n in edits:
        c = text.count(old)
        if c != n:
            sys.exit(f'[{tag}] anchor x{c} (want {n}):\n{old[:150]}')
        text = text.replace(old, new)
    return text


# ------------------------------------------------------------------ solver
lip = LIP.read_text()
if 'ACTION CONVENTION FIX' not in lip:
    lip = apply(lip, [
        # rollout inputs inside the dino proposal: feed normalized A
        ('                zT = rollout_terminal_dino(wm, toks, A_in * self._ast + self._amu)',
         '                # ACTION CONVENTION FIX: the predictor was trained on\n'
         '                # z-scored action blocks (measured: 2.7x lower rollout\n'
         '                # error at h=5 vs raw, and raw ~= zero-action). Feed the\n'
         '                # normalized plan straight through, as the LeWM path does.\n'
         '                zT = rollout_terminal_dino(wm, toks, A_in)', 1),
        ('                E = self.lip_value(\n'
         '                    rollout_terminal_dino(wm, toks, A * self._ast + self._amu), zg)',
         '                E = self.lip_value(rollout_terminal_dino(wm, toks, A), zg)', 1),
        # return normalized: policy.py inverse_transform's it for the env
        ('        return (A * self._ast + self._amu).to(self.dtype)            # raw action units',
         '        # policy.py applies process["action"].inverse_transform to the\n'
         '        # solver output, so return NORMALIZED units (as _proposal_lip does).\n'
         '        # Returning raw here double-scaled every action sent to the env.\n'
         '        return A.to(self.dtype)', 1),
    ], 'solver')
    LIP.write_text(lip)
    print('solver: fixed (rollout inputs + return)')
else:
    print('solver: already fixed')

# ----------------------------------------------------------------- trainers
for TR in TRS:
    if not TR.exists():
        print(f'{TR.name}: absent, skip')
        continue
    t = TR.read_text()
    if 'ACTION CONVENTION FIX' in t:
        print(f'{TR.name}: already fixed')
        continue
    n_before = t.count('* ast5 + amu5')
    t = t.replace('rollout_traj_dino(wm, toks, A_in * ast5 + amu5)',
                  'rollout_traj_dino(wm, toks, A_in)  # ACTION CONVENTION FIX: normalized')
    t = t.replace('rollout_traj_dino(wm, toks, A * ast5 + amu5,\n'
                  '                                           return_toks=True)',
                  'rollout_traj_dino(wm, toks, A, return_toks=True)  # normalized')
    t = t.replace('rollout_traj_dino(wm, toks, A * ast5 + amu5)',
                  'rollout_traj_dino(wm, toks, A)  # normalized')
    t = t.replace('rollout_terminal_dino(wm, toks, A_in * ast5 + amu5)',
                  'rollout_terminal_dino(wm, toks, A_in)  # ACTION CONVENTION FIX')
    t = t.replace('rollout_terminal_dino(wm, toks, A * ast5 + amu5)',
                  'rollout_terminal_dino(wm, toks, A)  # normalized')
    n_after = t.count('* ast5 + amu5')
    TR.write_text(t)
    print(f'{TR.name}: {n_before} scaled call(s) -> {n_after} remaining')

for f in [LIP] + [p for p in TRS if p.exists()]:
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(f)],
                       capture_output=True, text=True)
    print(f'{f.name}: compile {"OK" if r.returncode == 0 else "FAIL " + r.stderr[-300:]}')
