"""Convert the official lewm-pusht weights.pt (transformers-4 HF ViT naming)
to the transformers-5 naming this pod's vit_hf produces, then smoke-test.

Writes /workspace/swm_home/checkpoints/lewm_pusht_official/{weights.pt,config.json}.
"""

import json
import re
import shutil
from pathlib import Path

import torch
from hydra.utils import instantiate

SRC = Path("/workspace/swm_home/checkpoints/models--quentinll--lewm-pusht")
DST = Path("/workspace/swm_home/checkpoints/lewm_pusht_official")

sd = torch.load(SRC / "weights.pt", map_location="cpu")
config = json.loads((SRC / "config.json").read_text())
model = instantiate(config)
want = set(model.state_dict().keys())
have = set(sd.keys())

RULES = [
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_before\.", r"encoder.layers.\1.layernorm_before."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_after\.", r"encoder.layers.\1.layernorm_after."),
]

new_sd = {}
for k, v in sd.items():
    nk = k
    for pat, rep in RULES:
        nk2 = re.sub(pat, rep, nk)
        if nk2 != nk:
            nk = nk2
            break
    new_sd[nk] = v

missing = sorted(want - set(new_sd.keys()))
unexpected = sorted(set(new_sd.keys()) - want)
print(f"after remap: missing={len(missing)} unexpected={len(unexpected)}")
for k in missing[:15]:
    print("  MISS", k)
for k in unexpected[:15]:
    print("  UNEX", k)

if missing or unexpected:
    # show remaining raw checkpoint encoder keys to extend rules if needed
    raise SystemExit("remap incomplete")

model.load_state_dict(new_sd)
DST.mkdir(parents=True, exist_ok=True)
torch.save(new_sd, DST / "weights.pt")
shutil.copy(SRC / "config.json", DST / "config.json")

model = model.eval()
x = torch.rand(1, 3, 3, 224, 224)
with torch.no_grad():
    emb = model.encode({"pixels": x})["emb"]
print("converted OK, emb:", tuple(emb.shape), "->", DST)
