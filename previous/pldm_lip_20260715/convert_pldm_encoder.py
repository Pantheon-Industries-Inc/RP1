"""Convert PLDM_OgBench weights from old HF ViT key layout -> transformers-5.x layout,
and numerically validate with a from-scratch reference forward of the OLD ViT semantics.

Writes checkpoints/PLDM_OgBench_lewm/{weights.pt, config.json} (LeWM-target config).
"""
import json
import math
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = "/Users/arminsommer/SynologyDrive/1privat/Value_Metric_LeWM/stable-worldmodel/checkpoints"
SRC = os.path.join(ROOT, "PLDM_OgBench")
DST = os.path.join(ROOT, "PLDM_OgBench_lewm")

RENAMES = [
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_before\.", r"encoder.layers.\1.layernorm_before."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_after\.", r"encoder.layers.\1.layernorm_after."),
]


def convert_key(k):
    for pat, rep in RENAMES:
        k2, n = re.subn(pat, rep, k)
        if n:
            return k2
    return k


def reference_vit_forward(sd, x, n_layers=12, n_heads=3, eps=1e-12):
    """Faithful re-implementation of the old HF ViTModel forward (pre-LN, exact gelu)."""
    g = lambda k: sd["encoder." + k]
    B = x.shape[0]
    h = F.conv2d(x, g("embeddings.patch_embeddings.projection.weight"),
                 g("embeddings.patch_embeddings.projection.bias"), stride=14)
    h = h.flatten(2).transpose(1, 2)                          # (B, N, D)
    cls = g("embeddings.cls_token").expand(B, -1, -1)
    h = torch.cat([cls, h], dim=1) + g("embeddings.position_embeddings")
    D = h.shape[-1]
    hd = D // n_heads
    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        y = F.layer_norm(h, (D,), g(p + "layernorm_before.weight"), g(p + "layernorm_before.bias"), eps)
        q = F.linear(y, g(p + "attention.attention.query.weight"), g(p + "attention.attention.query.bias"))
        k = F.linear(y, g(p + "attention.attention.key.weight"), g(p + "attention.attention.key.bias"))
        v = F.linear(y, g(p + "attention.attention.value.weight"), g(p + "attention.attention.value.bias"))
        def split(t):
            return t.view(B, -1, n_heads, hd).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, -1, D)
        o = F.linear(o, g(p + "attention.output.dense.weight"), g(p + "attention.output.dense.bias"))
        h = h + o
        y = F.layer_norm(h, (D,), g(p + "layernorm_after.weight"), g(p + "layernorm_after.bias"), eps)
        y = F.linear(y, g(p + "intermediate.dense.weight"), g(p + "intermediate.dense.bias"))
        y = F.gelu(y)
        y = F.linear(y, g(p + "output.dense.weight"), g(p + "output.dense.bias"))
        h = h + y
    return F.layer_norm(h, (D,), sd["encoder.layernorm.weight"], sd["encoder.layernorm.bias"], eps)


def main():
    sd_old = torch.load(os.path.join(SRC, "weights.pt"), map_location="cpu", weights_only=True)
    sd_new = {convert_key(k): v for k, v in sd_old.items()}
    assert len(sd_new) == len(sd_old)

    from hydra.utils import instantiate
    cfg = json.load(open(os.path.join(SRC, "config_lewm_target.json")))
    m = instantiate(cfg)
    m.load_state_dict(sd_new, strict=True)
    m.eval()
    print("strict load through LeWM wrapper: OK")

    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = reference_vit_forward(sd_old, x)
        out = m.encoder(x, interpolate_pos_encoding=True).last_hidden_state
    diff = (ref - out).abs().max().item()
    rel = diff / ref.abs().max().item()
    print(f"reference-vs-converted max abs diff {diff:.3e} (rel {rel:.3e})")
    assert diff < 1e-4, "conversion mismatch"

    # also compare full encode() path (projector on cls token) between PLDM and LeWM classes
    cfg_pldm = json.load(open(os.path.join(SRC, "config.json")))
    mp = instantiate(cfg_pldm)
    mp.load_state_dict(sd_new, strict=True)
    mp.eval()
    with torch.no_grad():
        e1 = m.encode({"pixels": x.unsqueeze(1)})["emb"]
        e2 = mp.encode({"pixels": x.unsqueeze(1)})["emb"]
    d2 = (e1 - e2).abs().max().item()
    print(f"LeWM.encode vs PLDM.encode max abs diff {d2:.3e}")
    assert d2 < 1e-5

    os.makedirs(DST, exist_ok=True)
    torch.save(sd_new, os.path.join(DST, "weights.pt"))
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)
    print(f"wrote {DST}/weights.pt + config.json")


if __name__ == "__main__":
    main()
