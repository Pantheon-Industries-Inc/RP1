"""Port --replay-prob to the DINO LIPv4 trainer (token-space replay buffer).

The stock replay curriculum stores the previous actor step's final imagined
window (3 pooled latents) and substitutes it for a fraction of the next step's
contexts, so the refiner learns to CONTINUE from its own mid-task states -- the
exact query the deployed receding-horizon replan makes. My dino port asserted
replay off, because the dino actor's context is a TOKEN window
(B, 3, 196, 394), not pooled latents.

Two edits make it work:
  1. solver: rollout_traj_dino gains `return_toks`, so the rollout can hand back
     the last 3 imagined TOKEN frames alongside the pooled trajectory (it already
     tracks them internally; they were being discarded).
  2. trainer: the buffer stores those token frames with the ACTION SLICE ZEROED.
     That mirrors the stock version's `ah[idx] = 0.0`: at deploy the WM re-encodes
     real pixels with zero action history, so a replayed context must not carry
     the injected action embedding from imagination, or the actor trains on a
     token layout it never sees at plan time.

Critic targets stay expert-anchored (inputs only), exactly as upstream.
"""

import subprocess
import sys
from pathlib import Path

LIP = Path('/workspace/code/stable-worldmodel/stable_worldmodel/solver/lip.py')
TR = Path('/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac_dino.py')


def apply(text, edits, tag):
    for old, new, n in edits:
        c = text.count(old)
        if c != n:
            sys.exit(f'[{tag}] anchor x{c} (want {n}):\n{old[:140]}')
        text = text.replace(old, new)
    return text


lip = LIP.read_text()
if 'return_toks' not in lip:
    lip = apply(lip, [
        ('def rollout_traj_dino(wm, toks_hist, plan, pix_dim=384, act_emb_dim=10):',
         'def rollout_traj_dino(wm, toks_hist, plan, pix_dim=384, act_emb_dim=10,\n'
         '                      return_toks=False):', 1),
        ('        outs.append(nxt[..., :pix_dim].mean(dim=1))\n'
         '    return torch.stack(outs, dim=1)\n',
         '        outs.append(nxt[..., :pix_dim].mean(dim=1))\n'
         '    traj = torch.stack(outs, dim=1)\n'
         '    if return_toks:\n'
         '        # last 3 imagined token frames -> replay contexts\n'
         '        return traj, torch.stack(toks[-3:], dim=1)\n'
         '    return traj\n', 1),
    ], 'solver')
    LIP.write_text(lip)
    print('solver: return_toks added')
else:
    print('solver: already has return_toks')

t = TR.read_text()
if 'replay_buf["toks"]' not in t:
    t = apply(t, [
        # drop the assertion
        ('    assert a.replay_prob == 0, "dino port: replay needs token buffers (not ported)"\n',
         '', 1),
        # buffer holds token windows now
        ('    replay_buf = {"zh": None, "zg": None}   # previous step\'s final imagined windows',
         '    replay_buf = {"toks": None, "zg": None}  # previous step\'s imagined TOKEN window', 1),
        # substitution on token contexts
        ('        toks, zg, aref = sample(a.batch)\n'
         '        if a.replay_prob > 0 and replay_buf["zh"] is not None:\n'
         '            pick = torch.rand(a.batch, device=dev) < a.replay_prob\n'
         '            idx = torch.nonzero(pick).squeeze(1)\n'
         '            if idx.numel():\n'
         '                take = torch.randint(0, replay_buf["zh"].shape[0], (idx.numel(),), device=dev)\n'
         '                zh, ah, zg = zh.clone(), ah.clone(), zg.clone()\n'
         '                zh[idx] = replay_buf["zh"][take]\n'
         '                zg[idx] = replay_buf["zg"][take]\n'
         '                ah[idx] = 0.0        # deployed replans query with zero action history\n',
         '        toks, zg, aref = sample(a.batch)\n'
         '        if a.replay_prob > 0 and replay_buf["toks"] is not None:\n'
         '            pick = torch.rand(a.batch, device=dev) < a.replay_prob\n'
         '            idx = torch.nonzero(pick).squeeze(1)\n'
         '            if idx.numel():\n'
         '                take = torch.randint(0, replay_buf["toks"].shape[0], (idx.numel(),),\n'
         '                                     device=dev)\n'
         '                toks, zg = toks.clone(), zg.clone()\n'
         '                toks[idx] = replay_buf["toks"][take]\n'
         '                zg[idx] = replay_buf["zg"][take]\n', 1),
        # final rollout returns tokens; stash them zero-actioned
        ('            tr = rollout_traj_dino(wm, toks, A * ast5 + amu5)\n',
         '            tr, tr_toks = rollout_traj_dino(wm, toks, A * ast5 + amu5,\n'
         '                                           return_toks=True)\n', 1),
        ('        if a.replay_prob > 0:\n'
         '            replay_buf["zh"] = tr[:, -3:].detach()\n'
         '            replay_buf["zg"] = zg.detach()\n',
         '        if a.replay_prob > 0:\n'
         '            # zero the action slice: deploy re-encodes real pixels with zero\n'
         '            # action history, so replayed contexts must match that layout\n'
         '            rb = tr_toks.detach().clone()\n'
         '            rb[..., -10:] = 0.0\n'
         '            replay_buf["toks"] = rb\n'
         '            replay_buf["zg"] = zg.detach()\n', 1),
    ], 'trainer')
    TR.write_text(t)
    print('trainer: token replay wired')
else:
    print('trainer: already wired')

for f in (LIP, TR):
    r = subprocess.run([sys.executable, '-m', 'py_compile', str(f)],
                       capture_output=True, text=True)
    print(f'{f.name}: compile {"OK" if r.returncode == 0 else "FAIL " + r.stderr[-300:]}')
