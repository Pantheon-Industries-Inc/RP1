"""Train DINO-WM on pixels: frozen DINOv2 encoder + learned latent predictor.

Because DINOv2 is frozen, we encode every frame ONCE (GPU) into a feature cache,
then train only the small action-conditioned predictor on the cached feature
sequences. The same feature cache is dumped for the TRM metric learners, so the
expensive encoder runs a single time for the whole pipeline.

Example::

    python scripts/trm/train_dinowm.py --dataset tworoom_pixels.lance \
        --run-name dinowm_tworoom --action-dim 2 --steps 6000 \
        --cache-out caches/dinowm_tworoom.pt --device cuda
"""

import argparse
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger as logging
from PIL import Image
from torchvision.transforms import v2 as T
from tqdm import tqdm


def _decode_pixel(p):
    """Decode a stored pixel frame: JPEG/PNG bytes -> HWC uint8 array, else passthrough."""
    if isinstance(p, (bytes, bytearray, np.bytes_)):
        return np.array(Image.open(BytesIO(bytes(p))))
    return np.asarray(p)

import stable_worldmodel as swm
from stable_worldmodel.trm import LatentCache
from stable_worldmodel.wm.dinowm import DinoWM
from stable_worldmodel.wm.utils import save_pretrained

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pick_device(name):
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def precompute_features(dataset, wm, device, img_size=224, batch_size=128):
    """Encode every frame with frozen DINOv2 -> (N, D) features + actions + idx."""
    tf = T.Compose([
        T.ToImage(), T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), T.Resize(size=img_size),
    ])
    ep = np.asarray(dataset.get_col_data("episode_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    n = len(ep)
    feats, acts = [], []
    for start in tqdm(range(0, n, batch_size), desc="dino encode"):
        idx = list(range(start, min(start + batch_size, n)))
        rows = dataset.get_row_data(idx)
        imgs = torch.stack([tf(_decode_pixel(p)) for p in rows["pixels"]]).to(device)
        with torch.no_grad():
            f = wm._encode_pixels(imgs).float().cpu()
        feats.append(f)
        acts.append(np.nan_to_num(np.asarray(rows["action"]).reshape(len(idx), -1).astype(np.float32)))
    feats = torch.cat(feats, 0)
    acts = torch.from_numpy(np.concatenate(acts, 0))
    return feats, acts, torch.from_numpy(ep), torch.from_numpy(st)


def episode_index(ep, st):
    out = {}
    ep = ep.numpy(); st = st.numpy()
    for e in np.unique(ep):
        rows = np.nonzero(ep == e)[0]
        out[int(e)] = rows[np.argsort(st[rows])]
    return out


def sample_windows(episodes, feats, acts, win, batch, rng):
    valid = [e for e, r in episodes.items() if len(r) >= win]
    fdim, adim = feats.shape[1], acts.shape[1]
    fo = torch.empty((batch, win, fdim)); ao = torch.empty((batch, win, adim))
    for b in range(batch):
        e = valid[rng.integers(0, len(valid))]
        rows = episodes[e]
        t = rng.integers(0, len(rows) - win + 1)
        idx = rows[t:t + win]
        fo[b] = feats[idx]; ao[b] = acts[idx]
    return fo, ao


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--cache-out", required=True, help="where to dump the DINOv2 feature LatentCache")
    p.add_argument("--backbone", default="dinov2_small")
    p.add_argument("--action-dim", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--act-emb-dim", type=int, default=64)
    p.add_argument("--n-preds", type=int, default=3)
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = swm.data.load_dataset(args.dataset)
    wm = DinoWM(backbone_name=args.backbone, action_dim=args.action_dim,
                hidden_dim=args.hidden_dim, act_emb_dim=args.act_emb_dim).to(device)
    wm.eval()

    logging.info("precomputing frozen DINOv2 features ...")
    feats, acts, ep, st = precompute_features(dataset, wm, device)
    logging.info(f"features {tuple(feats.shape)} dino_dim={wm.dino_dim} latent_dim={wm.latent_dim}")

    # dump the shared latent cache for the metric learners
    LatentCache(z=feats.clone(), episode_idx=ep, step_idx=st,
                meta={"wm": args.run_name, "dataset": args.dataset}).save(args.cache_out)

    # train predictor (+ action encoder + proj) on cached feature sequences
    episodes = episode_index(ep, st)
    feats_d = feats.to(device)
    params = list(wm.predictor.parameters()) + list(wm.action_encoder.parameters()) + list(wm.proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    win = args.history + args.n_preds
    h = args.history
    wm.train()
    pbar = tqdm(range(args.steps), desc="dinowm-predictor")
    for step in pbar:
        f, a = sample_windows(episodes, feats, acts, win, args.batch_size, rng)
        f, a = f.to(device), a.to(device)
        z = wm.proj(f)                              # (B, win, latent)
        act_emb = wm.action_encoder(a)              # (B, win, A)
        pred = z[:, h - 1]
        loss = 0.0
        for k in range(args.n_preds):
            pred = wm.predictor(pred.unsqueeze(1), act_emb[:, h - 1 + k:h + k])[:, 0]
            loss = loss + F.mse_loss(pred, z[:, h + k].detach())
        loss = loss / args.n_preds
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            pbar.set_postfix(pred_loss=loss.item())

    wm.eval()
    config = {
        "_target_": "stable_worldmodel.wm.dinowm.DinoWM",
        "backbone_name": args.backbone, "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim, "act_emb_dim": args.act_emb_dim,
        "latent_dim": wm.latent_dim, "obs_key": "pixels", "goal_key": "goal",
    }
    save_pretrained(wm.cpu(), run_name=args.run_name, config=config)
    logging.success(f"DinoWM saved as '{args.run_name}'; feature cache -> {args.cache_out}")


if __name__ == "__main__":
    main()
