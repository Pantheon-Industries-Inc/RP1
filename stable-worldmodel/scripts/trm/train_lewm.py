"""Train a LeWM world model on TwoRoom pixels (custom loop, faithful hparams).

The repo's scripts/train/lewm.py relies on a stable-pretraining transforms
pipeline incompatible with the installed spt 0.1.7. This trainer reuses the
repo's ``LeWM`` module + ``SIGReg`` loss but with a plain (GPU-vectorised)
torchvision data pipeline, and matches the repo's tworoom LeWM hyperparameters:

  encoder ViT-tiny/p14/d192, predictor d6/h16/mlp2048, history=3 num_preds=1,
  SIGReg weight 0.09 (knots 17, num_proj 1024), AdamW lr 5e-5 wd 1e-3,
  batch 128, LinearWarmup+Cosine, frameskip 5 (action encoder sees 5*action_dim),
  and the JEPA predictor loss WITHOUT stop-grad on the target (SIGReg prevents
  collapse). Periodic logging + checkpoints for visibility.

Example::

    python scripts/trm/train_lewm.py --dataset tworoom_pixels.lance \
        --run-name lewm_tworoom --raw-action-dim 2 --frameskip 5 --steps 20000 --device cuda
"""

import argparse
import math
from io import BytesIO

import numpy as np
import torch
from hydra.utils import instantiate
from loguru import logger as logging
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import stable_worldmodel as swm
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def lewm_config(action_dim, embed_dim=192, patch=14, img=224):
    return OmegaConf.create({
        "_target_": "stable_worldmodel.wm.lewm.LeWM",
        "encoder": {"_target_": "stable_pretraining.backbone.utils.vit_hf",
                    "size": "tiny", "patch_size": patch, "image_size": img,
                    "pretrained": False, "use_mask_token": False},
        "predictor": {"_target_": "stable_worldmodel.wm.lewm.module.Predictor",
                      "num_frames": 3, "input_dim": embed_dim, "hidden_dim": embed_dim,
                      "output_dim": embed_dim, "depth": 6, "heads": 16, "mlp_dim": 2048,
                      "dim_head": 64, "dropout": 0.1, "emb_dropout": 0.0},
        "action_encoder": {"_target_": "stable_worldmodel.wm.lewm.module.Embedder",
                           "input_dim": action_dim, "emb_dim": embed_dim},
        "projector": {"_target_": "stable_worldmodel.wm.lewm.module.MLP",
                      "input_dim": embed_dim, "output_dim": embed_dim, "hidden_dim": 2048,
                      "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
        "pred_proj": {"_target_": "stable_worldmodel.wm.lewm.module.MLP",
                      "input_dim": embed_dim, "output_dim": embed_dim, "hidden_dim": 2048,
                      "norm_fn": {"_target_": "torch.nn.BatchNorm1d", "_partial_": True}},
    })


def decode_frames(dataset):
    ep = np.asarray(dataset.get_col_data("episode_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    n = len(ep)
    frames = np.empty((n, 224, 224, 3), np.uint8)
    acts = []
    for start in tqdm(range(0, n, 512), desc="decode"):
        idx = list(range(start, min(start + 512, n)))
        rows = dataset.get_row_data(idx)
        for j, p in enumerate(rows["pixels"]):
            frames[start + j] = np.array(Image.open(BytesIO(bytes(p)))) if isinstance(p, (bytes, bytearray, np.bytes_)) else np.asarray(p)
        acts.append(np.nan_to_num(np.asarray(rows["action"]).reshape(len(idx), -1).astype(np.float32)))
    acts = np.concatenate(acts, 0)
    episodes = {int(e): np.nonzero(ep == e)[0][np.argsort(st[ep == e])] for e in np.unique(ep)}
    return frames, acts, episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--raw-action-dim", type=int, default=2)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--n-preds", type=int, default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--sigreg-weight", type=float, default=0.09)
    p.add_argument("--warmup-frac", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=4000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dataset = swm.data.load_dataset(args.dataset)
    frames, acts, episodes = decode_frames(dataset)
    logging.info(f"decoded {len(frames)} frames, {len(episodes)} episodes")

    fs, win = args.frameskip, args.history + args.n_preds
    action_dim = fs * args.raw_action_dim  # LeWM action encoder sees frameskip actions
    span = win * fs                         # env-steps spanned by one training window
    valid = [e for e, r in episodes.items() if len(r) >= span]
    logging.info(f"frameskip={fs} action_dim={action_dim} span={span} valid_eps={len(valid)}")

    model = instantiate(lewm_config(action_dim)).to(args.device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_frac * args.steps))

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = (s - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))  # cosine to 0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    mean = torch.tensor(IMAGENET_MEAN, device=args.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=args.device).view(1, 1, 3, 1, 1)

    def sample_batch():
        fb = np.empty((args.batch_size, win, 224, 224, 3), np.uint8)
        ab = np.empty((args.batch_size, win, action_dim), np.float32)
        for b in range(args.batch_size):
            e = valid[rng.integers(0, len(valid))]
            rows = episodes[e]
            t0 = rng.integers(0, len(rows) - span + 1)
            for k in range(win):
                base = t0 + k * fs
                fb[b, k] = frames[rows[base]]
                ab[b, k] = acts[rows[base: base + fs]].reshape(-1)  # concat fs raw actions
        return fb, ab

    def to_gpu_imgs(fb):
        x = torch.from_numpy(fb).to(args.device).float().div_(255.0)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return (x - mean) / std

    ctx, npred, lam = args.history, args.n_preds, args.sigreg_weight
    model.train()
    pbar = tqdm(range(args.steps), desc="lewm")
    for step in pbar:
        fb, ab = sample_batch()
        imgs = to_gpu_imgs(fb)
        a = torch.as_tensor(ab, device=args.device)
        out = model.encode({"pixels": imgs, "action": a})
        emb, act_emb = out["emb"], out["act_emb"]
        pred = model.predict(emb[:, :ctx], act_emb[:, :ctx])
        tgt = emb[:, npred:]
        pred_loss = (pred - tgt).pow(2).mean()       # JEPA: NO stop-grad (SIGReg prevents collapse)
        sig_loss = sigreg(emb.transpose(0, 1))
        loss = pred_loss + lam * sig_loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % args.log_every == 0:
            logging.info(f"step {step}/{args.steps} pred={pred_loss.item():.4f} "
                         f"sigreg={sig_loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}")
        if step > 0 and step % args.save_every == 0:
            cfg = OmegaConf.to_container(lewm_config(action_dim), resolve=True)
            save_pretrained(model, run_name=args.run_name, config=cfg)
    model.eval()
    cfg = OmegaConf.to_container(lewm_config(action_dim), resolve=True)
    save_pretrained(model.cpu(), run_name=args.run_name, config=cfg)
    logging.success(f"LeWM saved as '{args.run_name}' (frameskip={fs}, action_dim={action_dim})")


if __name__ == "__main__":
    main()
