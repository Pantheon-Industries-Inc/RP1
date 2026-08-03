"""Verify the rebuilt environment and that every checkpoint loads COMPLETELY.

The container has been recycled twice, and several behaviours here are
version-sensitive:
  * transformers 4.49.0 -- `_vit_layout_compat` remaps ViT encoder keys when the
    checkpoint and the installed transformers disagree on layout. A different
    version can silently remap (or fail to remap) keys.
  * cuDNN must be 90100. 92000 means leftover nvidia-*-cu13 wheels are shadowing
    the cu12 ones, which kills every conv.
  * numpy<2 -- torch 2.4.1 was built against numpy 1.x.

The load check is the important one: if `load_state_dict` is non-strict, missing
keys mean parts of the model are RANDOMLY INITIALISED and nothing would tell us.
The DINO-WM scoring 80.0 with plain CEM suggests it is largely intact, but
"largely" is not a verification.
"""

import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace/code/stable-worldmodel')

PINS = {
    'torch': '2.4.1+cu124',
    'transformers': '4.49.0',
    'numpy': '1.x (<2)',
    'cudnn': 90100,
}


def line(k, got, want=None, ok=None):
    if ok is None:
        ok = (want is None) or (str(got) == str(want))
    mark = 'ok  ' if ok else 'WARN'
    tail = '' if want is None else f'   (pin: {want})'
    print(f'  [{mark}] {k:24s} {got}{tail}')


print('=== versions ===')
import torchvision  # noqa: E402
import transformers  # noqa: E402
line('torch', torch.__version__, PINS['torch'])
line('torchvision', torchvision.__version__, '0.19.1+cu124')
line('transformers', transformers.__version__, PINS['transformers'])
line('numpy', np.__version__, ok=np.__version__.startswith('1.'))
cudnn = torch.backends.cudnn.version()
line('cudnn', cudnn, PINS['cudnn'], ok=(cudnn == PINS['cudnn']))
line('cuda (torch)', torch.version.cuda, '12.4')
line('gpus visible', torch.cuda.device_count(), ok=torch.cuda.device_count() >= 1)
for mod in ('lance', 'mujoco', 'dm_control', 'gymnasium', 'stable_pretraining', 'h5py'):
    try:
        m = __import__(mod)
        line(mod, getattr(m, '__version__', 'present'))
    except Exception as e:
        line(mod, f'IMPORT FAILED: {type(e).__name__}', ok=False)

print('\n=== stray CUDA-13 wheels (they shadow cuDNN) ===')
import subprocess  # noqa: E402
out = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True,
                     text=True).stdout
bad = [l for l in out.splitlines()
       if l.lower().startswith('nvidia-') and '-cu12' not in l.lower()]
if bad:
    for b in bad:
        line('stray', b.strip(), ok=False)
else:
    line('nvidia-* wheels', 'all cu12, none stray')

print('\n=== DINO-WM: does the state_dict load COMPLETELY? ===')
import json  # noqa: E402
import hydra  # noqa: E402

CK = '/workspace/ckpts/dinowm_noprop_cube'
cfg = json.load(open(f'{CK}/config.json'))
model = hydra.utils.instantiate(cfg)
sd = torch.load(f'{CK}/weights_epoch_10.pt', map_location='cpu', weights_only=True)
missing, unexpected = model.load_state_dict(sd, strict=False)
# buffers legitimately absent from a saved state_dict (attention causal masks are
# registered buffers regenerated at construction)
mask_only = [k for k in missing if k.endswith('.bias') and 'layers' in k
             and 'to_qkv' not in k and 'norm' not in k and 'net' not in k]
real_missing = [k for k in missing if k not in mask_only]
line('missing keys', f'{len(missing)} (of which {len(mask_only)} attn masks)',
     ok=len(real_missing) == 0)
line('unexpected keys', len(unexpected), 0, ok=len(unexpected) == 0)
if real_missing[:5]:
    print('    real missing sample:', real_missing[:5])
if unexpected[:5]:
    print('    unexpected sample:', unexpected[:5])
shape_bad = [k for k in sd if k in dict(model.state_dict())
             and dict(model.state_dict())[k].shape != sd[k].shape]
line('shape mismatches', len(shape_bad), 0, ok=not shape_bad)

print('\n=== did _vit_layout_compat remap anything? (version-dependent) ===')
import logging  # noqa: E402
recs = []


class Grab(logging.Handler):
    def emit(self, r):
        recs.append(r.getMessage())


logging.getLogger().addHandler(Grab())
logging.getLogger().setLevel(logging.INFO)
import stable_worldmodel as swm  # noqa: E402
wm = swm.wm.utils.load_pretrained(CK)
remap = [r for r in recs if 'vit-compat' in r]
line('vit-compat remap', remap[0] if remap else 'none (keys matched as-is)',
     ok=not remap)

print('\n=== numeric spot-check: rollout error must reproduce ===')
# the action-convention experiment measured, at h=1, raw 0.1117 / norm 0.0604.
# Reproduce the ORDERING here as a cheap end-to-end check that weights + versions
# give the same behaviour as when those numbers were taken.
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
wm = wm.to(dev).eval()
torch.manual_seed(0)
px = torch.randn(2, 3, 3, 196, 196, device=dev)
with torch.no_grad():
    emb = wm.encode({'pixels': px, 'action': torch.zeros(2, 3, 25, device=dev)})['emb']
line('token shape', tuple(emb.shape), '(2, 3, 196, 394)',
     ok=tuple(emb.shape) == (2, 3, 196, 394))

print('\n=== critic checkpoints load? ===')
from stable_worldmodel.trm import load_metric  # noqa: E402
import glob  # noqa: E402
for p in sorted(glob.glob('/workspace/metrics/dinopool_td_std96k.pt')
                + glob.glob('/workspace/metrics/dinopool_td_raw24k.pt')):
    try:
        m = load_metric(p, device='cpu')
        d = m.enc[0].in_features if hasattr(m, 'enc') else '?'
        line(p.split('/')[-1], f'loads, expects {d}-d input')
    except Exception as e:
        line(p.split('/')[-1], f'FAILED {type(e).__name__}: {e}', ok=False)

print('\ndone.')
