"""Add INIT_WEIGHTS env-gated warm-start to scripts/train/lewm_expert.py.

INIT_WEIGHTS=<path to weights .pt> loads the state dict into the freshly
instantiated model before training — used to fine-tune from v2WM. Idempotent.
"""
from pathlib import Path

p = Path("/workspace/code/stable-worldmodel/scripts/train/lewm_expert.py")
t = p.read_text()

if "INIT_WEIGHTS" in t:
    print("already patched — nothing to do")
    raise SystemExit(0)

a = "    world_model = hydra.utils.instantiate(cfg.model)"
n = """    world_model = hydra.utils.instantiate(cfg.model)
    _init_w = os.environ.get('INIT_WEIGHTS')
    if _init_w:
        import torch as _torch
        _sd = _torch.load(_init_w, map_location='cpu')
        world_model.load_state_dict(_sd)
        print(f'[init-weights] warm-started from {_init_w}', flush=True)"""

c = t.count(a)
print("anchor count:", c)
assert c == 1
p.write_text(t.replace(a, n))
print("lewm_expert.py INIT_WEIGHTS patch applied")
