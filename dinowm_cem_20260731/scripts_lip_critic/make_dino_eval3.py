"""Rebuild eval_wm_dino.py from pristine eval_wm.py with two fixes.

Fix 1 -- interpolate_pos_encoding.
  eval_wm.py force-sets it True, which assumes a ViT encoder accepting that
  kwarg. HF Dinov2Model.forward rejects it (Dinov2 always interpolates), so any
  DINOv2-backed WM dies with TypeError.

Fix 2 -- patch-token latents in the metric hook.
  `_MetricCost.get_cost` reads LeWM's `predicted_emb`/`goal_emb` and then takes
  `[..., -1, :]` for the last imagined FRAME. For PreJEPA/DINO-WM both are wrong:
    * the keys don't exist -- it publishes `predicted_embedding` plus an
      already-split pixels-only view `predicted_pixels_emb`/`pixels_goal_emb`;
    * the tensors are (B, N, T, P, D), so `[..., -1, :]` would index the last
      PATCH, not the last frame.

  The reduction is AUTO-DETECTED from the metric's own input width rather than
  configured, because getting it wrong is silent:
      metric_in == P*D  -> flatten the patch axis   (full-token teacher, 75264)
      metric_in == D    -> mean-pool the patch axis (pooled teacher, 384)
  A mismatch raises instead of quietly feeding the head the wrong space.
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
        '        # Dinov2Model.forward rejects `interpolate_pos_encoding`.\n'
        "        if getattr(model, 'interpolate_pos_encoding', None) is not False:\n"
        '            model.interpolate_pos_encoding = True\n'
    ),
    (
        "                    p_raw = info_dict['predicted_emb']\n"
        "                    g_raw = info_dict['goal_emb']\n",
        '                    # Patch-token WMs (PreJEPA / DINO-WM): different key\n'
        '                    # names, and (B, N, T, P, D) latents. Reduce the patch\n'
        '                    # axis to match how the metric was TRAINED, chosen by\n'
        '                    # the head width so a mismatch cannot pass silently.\n'
        "                    if 'predicted_emb' in info_dict:\n"
        "                        p_raw = info_dict['predicted_emb']\n"
        "                        g_raw = info_dict['goal_emb']\n"
        '                    else:\n'
        "                        p_raw = info_dict['predicted_pixels_emb']\n"
        "                        g_raw = info_dict['pixels_goal_emb']\n"
        '                    if p_raw.ndim >= 5:\n'
        '                        _P, _D = p_raw.shape[-2], p_raw.shape[-1]\n'
        '                        if self.metric_in % (_P * _D) == 0:\n'
        '                            p_raw = p_raw.flatten(-2, -1)\n'
        '                            g_raw = g_raw.flatten(-2, -1)\n'
        '                            _mode_note = f"flatten {_P}x{_D}"\n'
        '                        elif self.metric_in % _D == 0:\n'
        '                            p_raw = p_raw.mean(dim=-2)\n'
        '                            g_raw = g_raw.mean(dim=-2)\n'
        '                            _mode_note = f"mean-pool {_P} tokens"\n'
        '                        else:\n'
        '                            raise ValueError(\n'
        "                                f'metric expects {self.metric_in} dims but '\n"
        "                                f'latents are {_P}x{_D}: neither flatten '\n"
        "                                f'nor pool matches'\n"
        '                            )\n'
        '                        if not self._shapes_printed[0]:\n'
        "                            print(f'[metric-hook] {_mode_note} -> pred '\n"
        "                                  f'{tuple(p_raw.shape)} goal '\n"
        "                                  f'{tuple(g_raw.shape)} '\n"
        "                                  f'(metric_in={self.metric_in})',\n"
        '                                  flush=True)\n'
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
print(f'wrote {DST}: compile {"OK" if r.returncode == 0 else "FAIL " + r.stderr[:300]}')
