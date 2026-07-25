"""Reapply the blocks() episode-end clamp to train_lip_ac.py (idempotent).

Without it: action blocks at the episode tail silently bleed into the next
episode's actions, and hard-crash (ragged np.stack) on the dataset's final
episode. Discovered 2026-07-23 on the mixture ladder.
"""
from pathlib import Path

p = Path("/workspace/code/stable-worldmodel/scripts/plan/train_lip_ac.py")
t = p.read_text()

if "ep_len_h5" in t:
    print("already patched — nothing to do")
    raise SystemExit(0)

a1 = (
    '    with h5py.File(a.h5, "r") as h:\n'
    '        act = h["action"][:]\n'
    '        ep_off = h["ep_offset"][:]'
)
n1 = (
    '    with h5py.File(a.h5, "r") as h:\n'
    '        act = h["action"][:]\n'
    '        ep_off = h["ep_offset"][:]\n'
    '        ep_len_h5 = h["ep_len"][:] if "ep_len" in h else None'
)

a2 = (
    "    def blocks(e, t):\n"
    "        h0 = int(ep_off[e] + fs * t)\n"
    "        return act_n[h0:h0 + fs].reshape(-1)"
)
n2 = (
    "    def blocks(e, t):\n"
    "        # clamp the block inside the episode: never bleed into the next\n"
    "        # episode (and never run off the array end on the final one)\n"
    "        s = fs * t\n"
    "        if ep_len_h5 is not None:\n"
    "            s = min(s, int(ep_len_h5[e]) - fs)\n"
    "        h0 = int(ep_off[e] + s)\n"
    "        return act_n[h0:h0 + fs].reshape(-1)"
)

for name, a in (("a1", a1), ("a2", a2)):
    c = t.count(a)
    print(name, "count:", c)
    assert c == 1, f"{name} anchor not unique/missing"

t = t.replace(a1, n1).replace(a2, n2)
p.write_text(t)
print("train_lip_ac.py blocks() clamp applied")
