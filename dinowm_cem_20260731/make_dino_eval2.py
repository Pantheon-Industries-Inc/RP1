"""Rebuild eval_wm_dino.py from pristine eval_wm.py with two fixes.

Fix 1 -- interpolate_pos_encoding.
  The stock script force-sets `model.interpolate_pos_encoding = True`, which
  assumes a ViT encoder whose forward takes that kwarg. HF Dinov2Model does not
  (it always interpolates), so any DINOv2-backed WM dies with TypeError.

Fix 2 -- patch-token latents in the metric hook.
  `_MetricCost.get_cost` reads `info_dict['predicted_emb']` / `['goal_emb']`
  and then takes `[..., -1, :]` for the last imagined FRAME. Both assumptions
  are LeWM-specific:
    * PreJEPA/DINO-WM names them `predicted_embedding` and publishes an
      already-split pixels-only view `predicted_pixels_emb` / `pixels_goal_emb`
      (`predicted_emb` does not exist at all -> KeyError);
    * those tensors are (B, N, T, P, D), so `[..., -1, :]` would index the last
      PATCH rather than the last frame.
  So: prefer the LeWM keys, fall back to the pixels-only PreJEPA keys, and mean
  -pool the patch axis -- matching cache_latents_dino.py, which trained the
  metric on mean-pooled `pixels_emb`. Gated on `+metric_pool_dim=<D>` so
  behaviour for every existing LeWM checkpoint is bit-identical.
"""

import subprocess
import sys
from pathlib import Path

PLAN = Path('/workspace/code/stable-worldmodel/scripts/plan')
SRC, DST = PLAN / 'eval_wm.py', PLAN / 'eval_wm_dino.py'

EDITS = [
    (
        '        model.interpolate_pos_encoding = True\n',
        '        # Honour an explicit False from the checkpoint config: HF\n'
        '        # Dinov2Model.forward rejects `interpolate_pos_encoding` (it\n'
        '        # always interpolates), unlike the ViTModel this was written for.\n'
        "        if getattr(model, 'interpolate_pos_encoding', None) is not False:\n"
        '            model.interpolate_pos_encoding = True\n'
    ),
    (
        "                    p_raw = info_dict['predicted_emb']\n"
        "                    g_raw = info_dict['goal_emb']\n",
        '                    # Patch-token world models (PreJEPA / DINO-WM) use\n'
        '                    # different key names and emit (B, N, T, P, D).\n'
        '                    # They publish an already-split pixels-only view,\n'
        '                    # which is what the metric was trained on; pool the\n'
        '                    # patch axis so the `[..., -1, :]` below selects the\n'
        '                    # last imagined FRAME and not the last PATCH.\n'
        "                    if 'predicted_emb' in info_dict:\n"
        "                        p_raw = info_dict['predicted_emb']\n"
        "                        g_raw = info_dict['goal_emb']\n"
        '                    else:\n'
        "                        p_raw = info_dict['predicted_pixels_emb']\n"
        "                        g_raw = info_dict['pixels_goal_emb']\n"
        "                    if int(cfg.get('metric_pool_dim', 0) or 0):\n"
        '                        if p_raw.ndim >= 5:\n'
        '                            p_raw = p_raw.mean(dim=-2)\n'
        '                        if g_raw.ndim >= 5:\n'
        '                            g_raw = g_raw.mean(dim=-2)\n'
    ),
]

text = SRC.read_text()
for old, new in EDITS:
    n = text.count(old)
    if n != 1:
        sys.exit(f'expected 1 occurrence, found {n} of:\n{old[:160]}')
    text = text.replace(old, new, 1)
DST.write_text(text)

r = subprocess.run([sys.executable, '-m', 'py_compile', str(DST)],
                   capture_output=True, text=True)
print(f'wrote {DST}: compile {"OK" if r.returncode == 0 else "FAILED " + r.stderr[:300]}')

import difflib
diff = [l for l in difflib.unified_diff(
    SRC.read_text().splitlines(), DST.read_text().splitlines(), lineterm='', n=0)
    if l.startswith(('+', '-')) and not l.startswith(('+++', '---'))]
print(f'diff: {len(diff)} lines changed')
