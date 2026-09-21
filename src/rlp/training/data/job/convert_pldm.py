"""Rename a PLDM checkpoint's keys to the LeWM module layout.

The authors' PLDM export predates a stable_pretraining ``vit_hf`` refactor:
its encoder is stored under HF ViTModel names
(``encoder.encoder.layer.N.attention.attention.query`` ...) while the current
LeWM class expects ``encoder.layers.N.attention.q_proj``. Both are the SAME
architecture (vit_hf tiny / patch14 / 224) — verified 2026-08-06: 303 keys map
1:1 with identical shapes. All non-encoder components already share names.

The output pairs with a LeWM-target ``config.json`` (the tracked
``assets/core/world_model/pldm_cube/config.json`` is one) and then loads
through every LeWM code path unchanged.

Example::

    pixi run prepare job=convert_pldm preparation.src=PLDM_OgBench/weights.pt \
        dst=pldm_cube/weights.pt
"""

import re

import torch
from omegaconf import DictConfig

from rlp.utils.config import phase_config
from rlp.utils.logging import logger


def remap_key(key: str) -> str:
    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.(.*)", key)
    if not match:
        return key
    layer, rest = match.groups()
    rest = (
        rest.replace("attention.attention.query", "attention.q_proj")
        .replace("attention.attention.key", "attention.k_proj")
        .replace("attention.attention.value", "attention.v_proj")
        .replace("attention.output.dense", "attention.o_proj")
        .replace("intermediate.dense", "mlp.fc1")
        .replace("output.dense", "mlp.fc2")
    )
    return f"encoder.layers.{layer}.{rest}"


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")
    checkpoint = torch.load(args.src, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    remapped = {remap_key(key): value for key, value in state_dict.items()}
    if len(remapped) != len(state_dict):
        raise ValueError("key collision during remap")
    renamed = sum(1 for key in state_dict if remap_key(key) != key)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint["state_dict"] = remapped
        torch.save(checkpoint, args.dst)
    else:
        torch.save(remapped, args.dst)
    logger.success(f"Converted PLDM checkpoint: keys={len(remapped)} renamed={renamed} -> {args.dst}")
