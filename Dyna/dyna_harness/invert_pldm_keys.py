"""Invert the July PLDM key conversion, in place on the pod.

previous/pldm_lip_20260715/convert_pldm_encoder.py renamed the ViT keys
old-HF-ViTModel -> transformers-5.x (encoder.encoder.layer.N.attention.
attention.query -> encoder.layers.N.attention.q_proj, ...) for the OLD pod.
THIS pod runs the older transformers whose LeWM encoder expects the
ViTModel layout, so the uploaded PLDM_OgBench_lewm/weights.pt fails
strict load (Missing encoder.encoder.layer.*, Unexpected encoder.layers.*).
The conversion was keys-only and numerically validated at the time; its
inverse is exact.

Gates: every renamed key must round-trip uniquely; strict load through the
pod's own config must pass; a CPU forward must run. Original weights are
backed up OUTSIDE the model dir (the loader requires exactly one .pt inside).
"""
import json
import re
import shutil
import sys

import torch

# Generalized 2026-07-31: the same transformers-5.x -> 4.49 ViT inversion is
# needed for EVERY archived checkpoint on this pod, not just PLDM. v2WM has the
# identical problem (HYPERPARAMS sec 1: "the archived v2WM is in transformers-5.x
# ViT key layout; pods run 4.49.0").
import sys as _sys
P = _sys.argv[1] if len(_sys.argv) > 1 else "/workspace/models/PLDM_OgBench_lewm"
WEIGHTS = _sys.argv[2] if len(_sys.argv) > 2 else "weights.pt"
BACKUP = f"/workspace/models/{P.rstrip('/').split('/')[-1]}_transformers5_backup.pt"

INV = [
    (r"^encoder\.layers\.(\d+)\.attention\.q_proj\.", r"encoder.encoder.layer.\1.attention.attention.query."),
    (r"^encoder\.layers\.(\d+)\.attention\.k_proj\.", r"encoder.encoder.layer.\1.attention.attention.key."),
    (r"^encoder\.layers\.(\d+)\.attention\.v_proj\.", r"encoder.encoder.layer.\1.attention.attention.value."),
    (r"^encoder\.layers\.(\d+)\.attention\.o_proj\.", r"encoder.encoder.layer.\1.attention.output.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc1\.", r"encoder.encoder.layer.\1.intermediate.dense."),
    (r"^encoder\.layers\.(\d+)\.mlp\.fc2\.", r"encoder.encoder.layer.\1.output.dense."),
    (r"^encoder\.layers\.(\d+)\.layernorm_before\.", r"encoder.encoder.layer.\1.layernorm_before."),
    (r"^encoder\.layers\.(\d+)\.layernorm_after\.", r"encoder.encoder.layer.\1.layernorm_after."),
]


def invert(k):
    for pat, rep in INV:
        k2, n = re.subn(pat, rep, k)
        if n:
            return k2
    return k


def main():
    sd = torch.load(f"{P}/{WEIGHTS}", map_location="cpu", weights_only=True)
    sd2 = {invert(k): v for k, v in sd.items()}
    assert len(sd2) == len(sd), "key collision during inversion"
    n_ren = sum(invert(k) != k for k in sd)
    print(f"renamed {n_ren} of {len(sd)} keys")
    if n_ren == 0:
        print("nothing to invert -- weights already in ViTModel layout?")
        sys.exit(0)

    from hydra.utils import instantiate
    cfg = json.load(open(f"{P}/config.json"))
    m = instantiate(cfg)
    m.load_state_dict(sd2, strict=True)   # the gate that failed at 00:11
    m.eval()
    print("strict load through the pod's LeWM: OK")
    with torch.no_grad():
        x = torch.randn(1, 1, 3, 224, 224)
        e = m.encode({"pixels": x})["emb"]
    print(f"CPU encode forward OK, emb shape {tuple(e.shape)}")

    shutil.copy(f"{P}/{WEIGHTS}", BACKUP)
    torch.save(sd2, f"{P}/{WEIGHTS}")
    print(f"backup (transformers-5.x layout) -> {BACKUP}")
    print("INVERT_OK")


if __name__ == "__main__":
    main()
