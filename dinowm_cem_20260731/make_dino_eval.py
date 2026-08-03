"""Create eval_wm_dino.py: eval_wm.py with one line made conditional.

The stock script does an unconditional

    model.interpolate_pos_encoding = True

right after loading. That assumes a ViT-style encoder whose forward accepts the
kwarg. `create_backbone('dinov2_small')` returns a HF Dinov2Model, which does
NOT accept it (Dinov2 interpolates position encodings unconditionally inside
its embeddings), so PreJEPA passes an argument that raises TypeError.

A private copy is used instead of editing eval_wm.py because that file is
shared with other sessions' running campaigns.
"""

import sys
from pathlib import Path

src = Path('/workspace/code/stable-worldmodel/scripts/plan/eval_wm.py')
dst = Path('/workspace/code/stable-worldmodel/scripts/plan/eval_wm_dino.py')

OLD = '        model.interpolate_pos_encoding = True\n'
NEW = (
    '        # Honour an explicit False from the checkpoint config: HF\n'
    '        # Dinov2Model.forward rejects `interpolate_pos_encoding` (it always\n'
    '        # interpolates), unlike the ViTModel this line was written for.\n'
    "        if getattr(model, 'interpolate_pos_encoding', None) is not False:\n"
    '            model.interpolate_pos_encoding = True\n'
)

text = src.read_text()
if text.count(OLD) != 1:
    sys.exit(f'expected exactly one occurrence, found {text.count(OLD)}')
dst.write_text(text.replace(OLD, NEW))
print(f'wrote {dst} ({dst.stat().st_size} bytes)')

# the only difference must be that one line
import difflib

diff = [
    l for l in difflib.unified_diff(
        text.splitlines(), dst.read_text().splitlines(), lineterm='', n=0
    )
    if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))
]
print('diff lines:', len(diff))
for l in diff:
    print(' ', l)
