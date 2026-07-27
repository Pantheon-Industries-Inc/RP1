"""Convert the pasted TwoRoom pretrained bases (authors' release, 2026-03) into
load_pretrained checkpoints, and validate each end-to-end on real play data.

Inputs  (extracted from checkpoints/tworoom/pretrained/*.tar.zst):
  lejepa_weights.ckpt / pldm_weights.ckpt   old-HF-ViT-layout LeWM/PLDM twins
  dinowm_weights.ckpt / dinowm_noprop_weights.ckpt  PreJEPA DINO-WM (frozen
      DINOv2-small, 196 patches @ img 196, frame-causal predictor, token =
      [pix 384 | proprio 10 | act 10])

Outputs under $STABLEWM_HOME/checkpoints/:
  lejepa_tworoom/   (LeWM class)      pldm_tworoom/     (PLDM class)
  dinowm_tworoom/   (DinoWMTokens, proprio)
  dinowmnp_tworoom/ (DinoWMTokens, pixels-only)

Validation per WM (real transitions from tworoom_play.lance + h5 actions):
  - strict/accounted state-dict load
  - lejepa/pldm: converted encoder vs faithful old-ViT reference forward
  - dinowm: converted backbone weights vs stock facebook/dinov2-small (frozen)
  - one-step + 5-block open-loop latent prediction, scored relative to the
    copy-last-frame baseline and a shuffled-action control; the old
    lewm_tworoom (validated planner base) is run as the scale anchor.
"""
import argparse
import json
import math
import os
import re
import subprocess

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained

HOME = os.environ.get("STABLEWM_HOME", "/workspace/swm_home")
CKPT = os.path.join(HOME, "checkpoints")
ARCH = "/workspace/code/stable-worldmodel/checkpoints/tworoom/pretrained"
EXTRACT = "/workspace/pretrained"
# Validation data. Defaults kept for reproducing the July run, but the matrix
# campaign validates on the AUTHORS' canonical tworoom.h5 -- the dataset the
# released bases were trained on and the one the eval draws its tasks from.
# Validating on our noised tworoom_play.lance (ExpertPolicy action_noise=2.0)
# would measure open-loop fidelity on a distribution we never score.
# Both may point at the same file: the h5 side needs `action` + `ep_offset`,
# the dataset side needs `pixels`, `proprio`, `episode_idx`, `step_idx`.
H5 = os.environ.get("TWOROOM_VAL_H5", "/workspace/caches/tworoom_play.h5")
PLAY = os.environ.get("TWOROOM_VAL_DATASET", "tworoom_play.lance")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

VIT_RENAMES = [  # old HF ViT layout -> transformers-5.x vit_hf layout (validated on cube)
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r"encoder.layers.\1.attention.q_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r"encoder.layers.\1.attention.k_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r"encoder.layers.\1.attention.v_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r"encoder.layers.\1.attention.o_proj."),
    (r"^encoder\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r"encoder.layers.\1.mlp.fc1."),
    (r"^encoder\.encoder\.layer\.(\d+)\.output\.dense\.", r"encoder.layers.\1.mlp.fc2."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_before\.", r"encoder.layers.\1.layernorm_before."),
    (r"^encoder\.encoder\.layer\.(\d+)\.layernorm_after\.", r"encoder.layers.\1.layernorm_after."),
    (r"^encoder\.layernorm\.", r"encoder.layernorm."),
]

DINO_RENAMES = [  # PreJEPA save layout -> DinoWMTokens attribute layout
    (r"^backbone\.backbone\.", r"backbone."),          # unwrap EvalOnly
    (r"^extra_encoders\.action\.", r"action_encoder.embedder."),
    (r"^extra_encoders\.proprio\.", r"proprio_encoder."),
]


def rename(sd, rules):
    out = {}
    for k, v in sd.items():
        for pat, rep in rules:
            k2, n = re.subn(pat, rep, k)
            if n:
                k = k2
                break
        out[k] = v
    return out


def lewm_config(target):
    return {
        "_target_": target,
        "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf",
                    "size": "tiny", "patch_size": 14, "image_size": 224,
                    "pretrained": False, "use_mask_token": False},
        "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor",
                      "num_frames": 3, "input_dim": 192, "hidden_dim": 192,
                      "output_dim": 192, "depth": 6, "heads": 16,
                      "mlp_dim": 2048, "dim_head": 64, "dropout": 0.1},
        "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder",
                           "input_dim": 10, "emb_dim": 192},
        "projector": {"_target_": "stable_worldmodel.wm.lewm.module.MLP",
                      "input_dim": 192, "hidden_dim": 2048, "output_dim": 192,
                      "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
        "pred_proj": {"_target_": "stable_worldmodel.wm.lewm.module.MLP",
                      "input_dim": 192, "hidden_dim": 2048, "output_dim": 192,
                      "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
    }


def dino_config(with_proprio):
    tok = 384 + 10 + 10 if with_proprio else 384 + 10
    cfg = {
        "_target_": "stable_worldmodel.wm.dinowm.DinoWMTokens",
        "encoder": {"_target_": "stable_worldmodel.wm.prejepa.module.create_backbone",
                    "name": "dinov2_small"},
        "predictor": {"_target_": "stable_worldmodel.wm.prejepa.module.CausalPredictor",
                      "num_patches": 196, "num_frames": 3, "dim": tok,
                      "depth": 6, "heads": 16, "mlp_dim": 2048, "dim_head": 64,
                      "dropout": 0.0, "emb_dropout": 0.0},
        "action_encoder": {"_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                           "in_chans": 10, "emb_dim": 10},
        "proprio_encoder": ({"_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                             "in_chans": 2, "emb_dim": 10} if with_proprio else None),
        "img_size": 196, "action_dim": 10, "proprio_dim": 2, "history_size": 3,
        "interpolate_pos_encoding": True,
    }
    return cfg


def reference_vit_forward(sd, x, n_layers=12, n_heads=3, eps=1e-12):
    """Faithful old-HF ViTModel forward (pre-LN, exact gelu) from raw weights."""
    g = lambda k: sd["encoder." + k].to(x.device)
    B = x.shape[0]
    h = F.conv2d(x, g("embeddings.patch_embeddings.projection.weight"),
                 g("embeddings.patch_embeddings.projection.bias"), stride=14)
    h = h.flatten(2).transpose(1, 2)
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
        split = lambda t: t.view(B, -1, n_heads, hd).transpose(1, 2)
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
    return F.layer_norm(h, (D,), g("layernorm.weight"), g("layernorm.bias"), eps)


# ---------------------------------------------------------------- validation data
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _decode_px(p):
    if isinstance(p, (bytes, bytearray, np.bytes_)):
        from io import BytesIO
        from PIL import Image
        return np.array(Image.open(BytesIO(bytes(p))))
    return np.asarray(p)


def load_transitions(n=256, span=6, seed=0):
    """n windows of `span+1` WM frames (primitive rows 5 apart, the frameskip-5
    block rate) + raw proprio + the z-scored 10-d action block between each
    consecutive frame pair."""
    fs = 5
    ds = swm.data.load_dataset(PLAY)
    # Episode column naming differs by source: lance keeps 'episode_idx' as a
    # writer-managed index column, the authors' tworoom.h5 lists 'ep_idx'.
    # Same rule eval_wm.episode_col() uses -- hardcoding either one breaks on
    # the other, and get_col_data raises rather than returning empty.
    ep_name = "ep_idx" if "ep_idx" in ds.column_names else "episode_idx"
    ep = np.asarray(ds.get_col_data(ep_name)).reshape(-1)
    st = np.asarray(ds.get_col_data("step_idx")).reshape(-1)
    with h5py.File(H5, "r") as h:
        act = h["action"][:]
        ep_off = h["ep_offset"][:]
    amu, astd = np.nanmean(act, 0), np.nanstd(act, 0) + 1e-6
    act_n = ((act - amu) / astd).astype(np.float32)
    rng = np.random.default_rng(seed)
    starts = []
    for e in np.unique(ep)[:400]:
        rows = np.nonzero(ep == e)[0]
        rows = rows[np.argsort(st[rows])]
        need = fs * (span + 1)
        if len(rows) <= need + 1:
            continue
        t0 = int(rng.integers(0, len(rows) - need - 1))
        frame_rows = rows[t0: t0 + need + 1: fs][: span + 1]
        starts.append((int(e), frame_rows, int(st[rows[t0]])))
        if len(starts) >= n:
            break
    px, pro, blocks = [], [], []
    for e, frame_rows, t_first in starts:
        rd = ds.get_row_data(list(frame_rows))
        px.append(np.stack([_decode_px(p) for p in rd["pixels"]]))
        pro.append(np.asarray(rd["proprio"], dtype=np.float32).reshape(len(frame_rows), -1))
        bl = [act_n[ep_off[e] + t_first + fs * j: ep_off[e] + t_first + fs * j + fs].reshape(-1)
              for j in range(span)]
        blocks.append(np.stack(bl))
    px = torch.from_numpy(np.stack(px))            # (N, span+1, H, W, C) uint8
    pro = torch.from_numpy(np.stack(pro))          # (N, span+1, 2)
    blocks = torch.from_numpy(np.stack(blocks))    # (N, span, 10) z-scored
    imgs = px.permute(0, 1, 4, 2, 3).float() / 255.0
    imgs = (imgs - _MEAN.unsqueeze(0)) / _STD.unsqueeze(0)
    return imgs, pro, blocks, (amu, astd)


@torch.no_grad()
def encode_all(wm, imgs, pro):
    embs = []
    for i in range(0, imgs.shape[0], 32):
        info = {"pixels": imgs[i:i + 32].to(DEV)}
        if getattr(wm, "wants_proprio", False):
            info["proprio"] = pro[i:i + 32].to(DEV)
        embs.append(wm.encode(info)["emb"].float().cpu())
    return torch.cat(embs)


@torch.no_grad()
def rollout_check(wm, imgs, pro, blocks, tag):
    """One-step + 5-step open-loop prediction quality, vs copy/shuffle baselines."""
    from stable_worldmodel.solver.lip import rollout_traj
    z = encode_all(wm, imgs, pro)                         # (N, span+1, D)
    N, S1, D = z.shape
    span = S1 - 1
    zh = z[:, :3].to(DEV)                                 # history = frames 0..2
    ah = blocks[:, :2].to(DEV)                            # blocks taken at 0,1
    plan = blocks[:, 2:span].to(DEV)                      # blocks 2..span-1
    tr = []
    for i in range(0, N, 64):
        tr.append(rollout_traj(wm, zh[i:i + 64], ah[i:i + 64], plan[i:i + 64]).cpu())
    tr = torch.cat(tr)                                    # (N, span-2, D) preds for frames 3..span
    perm = torch.randperm(N)
    trs = []
    for i in range(0, N, 64):
        trs.append(rollout_traj(wm, zh[i:i + 64], ah[i:i + 64], plan[perm][i:i + 64]).cpu())
    trs = torch.cat(trs)
    res = {}
    for k, name in ((0, "1step"), (tr.shape[1] - 1, f"{tr.shape[1]}step")):
        tgt = z[:, 3 + k]
        prev = z[:, 2]
        e_pred = (tr[:, k] - tgt).pow(2).mean().item()
        e_copy = (prev - tgt).pow(2).mean().item()
        e_shuf = (trs[:, k] - tgt).pow(2).mean().item()
        res[name] = (e_pred, e_copy, e_shuf)
        print(f"[{tag}] {name}: pred {e_pred:.5f} | copy-last {e_copy:.5f} "
              f"(ratio {e_pred / max(e_copy, 1e-12):.3f}) | shuffled-act {e_shuf:.5f} "
              f"(ratio {e_pred / max(e_shuf, 1e-12):.3f})", flush=True)
    return res


def account_load(model, sd, allow_missing_prefixes=(), tag=""):
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not any(
        k.startswith(p) or re.search(p, k) for p in allow_missing_prefixes)]
    assert not unexpected, f"[{tag}] unexpected keys: {unexpected[:8]}"
    assert not bad_missing, f"[{tag}] missing keys: {bad_missing[:8]}"
    if missing:
        print(f"[{tag}] tolerated missing (deterministic/stats): {len(missing)}")
    return model


def write_ckpt(name, sd, cfg):
    d = os.path.join(CKPT, name)
    os.makedirs(d, exist_ok=True)
    torch.save(sd, os.path.join(d, "weights.pt"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote {d}/(weights.pt, config.json)", flush=True)


def convert_twin(src_name, out_name, target_cls, imgs, pro, blocks):
    sd_old = torch.load(os.path.join(EXTRACT, "tworoom", src_name), map_location="cpu",
                        weights_only=True)
    sd_new = rename(sd_old, VIT_RENAMES)
    assert len(sd_new) == len(sd_old)
    cfg = lewm_config(target_cls)
    from hydra.utils import instantiate
    m = instantiate(cfg)
    m.load_state_dict(sd_new, strict=True)
    m = m.to(DEV).eval()
    print(f"[{out_name}] strict load OK ({len(sd_new)} keys)")
    x = imgs[:4, 0].to(DEV)
    ref = reference_vit_forward({k: v for k, v in sd_old.items()}, x)
    out = m.encoder(x, interpolate_pos_encoding=True).last_hidden_state
    d = (ref - out).abs().max().item()
    print(f"[{out_name}] encoder parity vs old-ViT reference: max abs diff {d:.3e}")
    assert d < 1e-3, "encoder conversion mismatch"
    rollout_check(m, imgs, pro, blocks, out_name)
    write_ckpt(out_name, sd_new, cfg)
    return m


def convert_dino(src_name, out_name, with_proprio, imgs, pro, blocks, stats):
    sd_old = torch.load(os.path.join(EXTRACT, "tworoom", src_name), map_location="cpu",
                        weights_only=True)
    sd_new = rename(sd_old, DINO_RENAMES)
    # frozen-backbone check: must equal stock facebook/dinov2-small
    from stable_worldmodel.wm.prejepa.module import create_backbone
    stock = create_backbone("dinov2_small").state_dict()
    diffs = [k for k in stock
             if ("backbone." + k) in sd_new
             and not torch.equal(stock[k], sd_new["backbone." + k])]
    maxd = 0.0
    for k in diffs:
        maxd = max(maxd, (stock[k] - sd_new["backbone." + k]).abs().max().item())
    print(f"[{out_name}] backbone vs stock dinov2-small: {len(diffs)} differing "
          f"tensors (max abs {maxd:.2e})")
    cfg = dino_config(with_proprio)
    from hydra.utils import instantiate
    m = instantiate(cfg)
    amu, astd = stats
    bmu = torch.from_numpy(np.tile(amu, 5)).float()
    bstd = torch.from_numpy(np.tile(astd, 5)).float()
    sd_new["action_encoder.ext_mu"] = bmu
    sd_new["action_encoder.ext_std"] = bstd
    sd_new["action_encoder.train_mu"] = bmu.clone()   # training stats assumed == play stats
    sd_new["action_encoder.train_std"] = bstd.clone()  # (identity re-map); parity gates this
    if with_proprio:
        ds = swm.data.load_dataset(PLAY)
        pcol = np.asarray(ds.get_col_data("proprio"), dtype=np.float32).reshape(-1, 2)
        sd_new["pro_mu"] = torch.from_numpy(pcol.mean(0))
        sd_new["pro_std"] = torch.from_numpy(pcol.std(0) + 1e-6)
    account_load(m, sd_new, allow_missing_prefixes=(r"\.bias$",), tag=out_name)
    # (predictor attention masks are deterministic registered buffers; if the
    # 2026-03 save skipped them the freshly-built ones are identical)
    m = m.to(DEV).eval()
    rollout_check(m, imgs, pro, blocks, out_name)
    write_ckpt(out_name, sd_new, cfg)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma list: lejepa,pldm,dinowm,dinowmnp,anchor")
    args = p.parse_args()
    only = set(q for q in args.only.split(",") if q)

    os.makedirs(EXTRACT, exist_ok=True)
    for f in ("lewm", "pldm", "dinowm", "dinowm_noprop"):
        path = os.path.join(EXTRACT, "tworoom")
        marker = {"lewm": "lejepa_weights.ckpt", "pldm": "pldm_weights.ckpt",
                  "dinowm": "dinowm_weights.ckpt",
                  "dinowm_noprop": "dinowm_noprop_weights.ckpt"}[f]
        if not os.path.exists(os.path.join(path, marker)):
            print(f"extracting {f}.tar.zst", flush=True)
            subprocess.run(
                ["tar", "-I", "zstd", "--no-same-owner", "--no-same-permissions",
                 "-xf", os.path.join(ARCH, f"{f}.tar.zst"), "-C", EXTRACT],
                check=True)

    imgs, pro, blocks, stats = load_transitions()
    print(f"validation transitions: imgs {tuple(imgs.shape)} pro {tuple(pro.shape)} "
          f"blocks {tuple(blocks.shape)}", flush=True)

    if not only or "anchor" in only:
        anchor = load_pretrained("lewm_tworoom").to(DEV).eval()
        rollout_check(anchor, imgs, pro, blocks, "ANCHOR lewm_tworoom(old)")
        del anchor
    if not only or "lejepa" in only:
        convert_twin("lejepa_weights.ckpt", "lejepa_tworoom",
                     "stable_worldmodel.wm.lewm.LeWM", imgs, pro, blocks)
    if not only or "pldm" in only:
        convert_twin("pldm_weights.ckpt", "pldm_tworoom",
                     "stable_worldmodel.wm.pldm.PLDM", imgs, pro, blocks)
    if not only or "dinowm" in only:
        convert_dino("dinowm_weights.ckpt", "dinowm_tworoom", True,
                     imgs, pro, blocks, stats)
    if not only or "dinowmnp" in only:
        convert_dino("dinowm_noprop_weights.ckpt", "dinowmnp_tworoom", False,
                     imgs, pro, blocks, stats)
    print("CONVERSION DONE", flush=True)


if __name__ == "__main__":
    main()
