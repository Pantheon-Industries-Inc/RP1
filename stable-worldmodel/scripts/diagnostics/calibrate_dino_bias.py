"""Calibrate the reacher DINO-WM predictor's constant output offset.

The released PreJEPA predictor (``DinoWMTokens``, flat 196x384 = 75264-d
tokens) lands in a SHIFTED copy of the encoder's feature space: for every
transition,

    predict(encode(hist), actions) ~= encode(next_frame) + c

with ``c`` fixed (|c| ~ 214, cosine 0.999 across independent data splits).
Raw 1-step pred/copy-last MSE ratio is ~2.6 (worse than not moving at all);
subtracting ``c`` drops it below 1. This script measures ``c`` on HALF the
transitions, reports the ratio on the HELD-OUT half, and writes a calibrated
checkpoint that is byte-identical to the original except for the new
``pred_bias`` buffer.

    python3 /workspace/calibrate_dino_bias.py
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, "/workspace/stable-worldmodel/scripts/plan")
sys.path.insert(0, "/workspace/stable-worldmodel")

import stable_worldmodel as swm  # noqa: E402
from convert_reacher_bases import load_transitions  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HOME = os.environ.get("STABLEWM_HOME", "/workspace/swm_home")
CKPT = os.path.join(HOME, "checkpoints")


@torch.no_grad()
def encode_all(wm, imgs, bs=8):
    """(N, T, C, H, W) -> (N, T, D) flat-token latents."""
    out = []
    for i in range(0, imgs.shape[0], bs):
        out.append(wm.encode({"pixels": imgs[i : i + bs].to(DEV)})["emb"].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def rollout(wm, zh, ah, plan, bs=16):
    """Real-history open-loop unroll; returns (N, plan_len, D) on cpu."""
    out = []
    for i in range(0, zh.shape[0], bs):
        out.append(
            wm.rollout_traj(
                zh[i : i + bs].to(DEV), plan[i : i + bs].to(DEV), ah[i : i + bs].to(DEV)
            ).cpu()
        )
    return torch.cat(out)


def report(wm, z, blocks, tag):
    """1-step and multi-step pred vs copy-last, with the current pred_bias."""
    span = blocks.shape[1]
    zh, ah, plan = z[:, :3], blocks[:, :2], blocks[:, 2:span]
    tr = rollout(wm, zh, ah, plan)
    res = {}
    for k, name in ((0, "1step"), (tr.shape[1] - 1, f"{tr.shape[1]}step")):
        tgt, prev = z[:, 3 + k], z[:, 2]
        e_pred = (tr[:, k] - tgt).pow(2).mean().item()
        e_copy = (prev - tgt).pow(2).mean().item()
        res[name] = e_pred / max(e_copy, 1e-12)
        print(
            f"[{tag}] {name}: pred {e_pred:.5f} | copy-last {e_copy:.5f} "
            f"| ratio {res[name]:.4f}",
            flush=True,
        )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="dinowmnp_reacher")
    p.add_argument("--out", default="dinowmnp_cal_reacher")
    p.add_argument("--dataset", default="dmc/reacher_random.lance")
    p.add_argument("--h5", default="/workspace/caches/reacher_random_train.h5")
    p.add_argument("--n", type=int, default=400, help="transition windows")
    p.add_argument("--span", type=int, default=6)
    args = p.parse_args()

    wm = swm.wm.utils.load_pretrained(args.src).to(DEV).eval()
    wm.requires_grad_(False)
    assert float(wm.pred_bias.abs().max()) == 0.0, "source checkpoint is already calibrated"

    imgs, blocks, _ = load_transitions(args.dataset, args.h5, n=args.n, span=args.span)
    print(f"transitions: imgs {tuple(imgs.shape)} blocks {tuple(blocks.shape)}", flush=True)
    z = encode_all(wm, imgs)  # (N, span+1, D)
    N = z.shape[0]
    half = N // 2
    fit, hold = slice(0, half), slice(half, N)
    print(f"fit on {half} transitions, held-out {N - half}", flush=True)

    # ---- baseline (pred_bias = 0)
    raw_fit = report(wm, z[fit], blocks[fit], "raw/fit")
    raw_hold = report(wm, z[hold], blocks[hold], "raw/held-out")

    # ---- c = mean 1-step residual on the FIT half only
    pred1 = rollout(wm, z[fit, :3], blocks[fit, :2], blocks[fit, 2:3])[:, 0]
    resid = pred1 - z[fit, 3]
    c = resid.mean(dim=0)
    # sanity: is the offset really constant? split the fit half in two.
    c_a = resid[: half // 2].mean(dim=0)
    c_b = resid[half // 2 :].mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(c_a, c_b, dim=0).item()
    print(
        f"c: |c| {c.norm().item():.2f} mean {c.mean().item():+.4f} "
        f"std {c.std().item():.4f} | split-half cosine {cos:.4f} "
        f"| residual std/|c| {resid.std(dim=0).norm().item() / c.norm().item():.3f}",
        flush=True,
    )

    # ---- calibrated
    wm.pred_bias.copy_(c.to(DEV))
    cal_fit = report(wm, z[fit], blocks[fit], "cal/fit")
    cal_hold = report(wm, z[hold], blocks[hold], "cal/held-out")

    # ---- write the calibrated checkpoint (original weights + pred_bias)
    src_dir = os.path.join(CKPT, args.src)
    out_dir = os.path.join(CKPT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    sd = torch.load(os.path.join(src_dir, "weights.pt"), map_location="cpu")
    n_before = len(sd)
    sd["pred_bias"] = c.cpu().clone()
    torch.save(sd, os.path.join(out_dir, "weights.pt"))
    shutil.copyfile(
        os.path.join(src_dir, "config.json"), os.path.join(out_dir, "config.json")
    )
    print(f"wrote {out_dir} ({n_before} -> {len(sd)} keys, config.json copied)", flush=True)

    # ---- verify it reloads and still beats copy-last on the held-out half
    del wm
    torch.cuda.empty_cache()
    wm2 = swm.wm.utils.load_pretrained(args.out).to(DEV).eval()
    wm2.requires_grad_(False)
    assert torch.allclose(wm2.pred_bias.cpu(), c.cpu()), "pred_bias did not round-trip"
    rl_hold = report(wm2, z[hold], blocks[hold], "reloaded/held-out")
    assert rl_hold["1step"] < 1.0, "calibrated 1-step ratio is not below copy-last"
    assert abs(rl_hold["1step"] - cal_hold["1step"]) < 1e-6

    summary = {
        "raw_fit": raw_fit,
        "raw_heldout": raw_hold,
        "cal_fit": cal_fit,
        "cal_heldout": cal_hold,
        "reloaded_heldout": rl_hold,
        "c_norm": c.norm().item(),
        "split_half_cosine": cos,
        "n_fit": half,
        "n_heldout": N - half,
    }
    print(json.dumps(summary, indent=2), flush=True)
    with open("/workspace/results/dino_bias_calibration.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("CALIBRATION DONE", flush=True)


if __name__ == "__main__":
    main()
