"""Train DINO-WM on pixels: frozen DINOv2 encoder + learned latent predictor.

Because DINOv2 is frozen, we encode every frame ONCE (GPU) into a feature cache,
then train only the small action-conditioned predictor on the cached feature
sequences. The same feature cache is dumped for the TRM metric learners, so the
expensive encoder runs a single time for the whole pipeline.

Example::

    pixi run train model=dino dataset=tworoom_pixels.lance \
        output.model_name=dinowm_tworoom core.world_model.action_dim=2 steps=6000 \
        cache_out=caches/dinowm_tworoom.pt device=cuda
"""

from io import BytesIO
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.transforms import v2 as T

from rlp.config import dispatch, run_hydra
from rlp.core.world_model import save_pretrained
from rlp.core.world_model.dinowm import DinoWM
from rlp.core.world_model.runtime import pick_device
from rlp.data import LatentCache
from rlp.data.protocols import Dataset
from rlp.logging import logger

from .utils import sample_windows


def _decode_pixel(p: object) -> np.ndarray[Any, Any]:
    """Decode a stored pixel frame: JPEG/PNG bytes -> HWC uint8 array, else passthrough."""
    if isinstance(p, (bytes, bytearray, np.bytes_)):
        return np.array(Image.open(BytesIO(bytes(p))))
    return np.asarray(p)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def precompute_features(
    dataset: Dataset, wm: DinoWM, device: str, img_size: int = 224, batch_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every frame with frozen DINOv2 -> (N, D) features + actions + idx."""
    tf = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.Resize(size=img_size),
        ]
    )
    ep = np.asarray(dataset.get_col_data("episode_idx")).reshape(-1).astype(np.int64)
    st = np.asarray(dataset.get_col_data("step_idx")).reshape(-1).astype(np.int64)
    n = len(ep)
    feature_chunks: list[torch.Tensor] = []
    action_chunks: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, n, batch_size)):
        idx = list(range(start, min(start + batch_size, n)))
        rows = dataset.get_row_data(idx)
        imgs = torch.stack([tf(_decode_pixel(p)) for p in rows["pixels"]]).to(device)
        with torch.no_grad():
            f = wm._encode_pixels(imgs).float().cpu()
        feature_chunks.append(f)
        action_chunks.append(np.nan_to_num(np.asarray(rows["action"]).reshape(len(idx), -1).astype(np.float32)))
        if batch_index % 100 == 0:
            logger.info(f"DINO encoding rows {start}/{n}")
    feats = torch.cat(feature_chunks, 0)
    acts = torch.from_numpy(np.concatenate(action_chunks, 0))
    return feats, acts, torch.from_numpy(ep), torch.from_numpy(st)


def episode_index(ep: torch.Tensor, st: torch.Tensor) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    ep_array = ep.numpy()
    st_array = st.numpy()
    for e in np.unique(ep_array):
        rows = np.nonzero(ep_array == e)[0]
        out[int(e)] = rows[np.argsort(st_array[rows])]
    return out


def _run(cfg: DictConfig) -> None:
    args = OmegaConf.merge(cfg, cfg.core.world_model)
    args.act_emb_dim = args.action_embedding_dim
    args.n_preds = args.num_predictions

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dataset = swm.data.load_dataset(args.dataset)
    wm = DinoWM(
        backbone_name=args.backbone,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        act_emb_dim=args.act_emb_dim,
    ).to(device)
    wm.eval()

    logger.info("Precomputing frozen DINOv2 features")
    feats, acts, ep, st = precompute_features(dataset, wm, device)
    logger.info(f"Features {tuple(feats.shape)} dino_dim={wm.dino_dim} latent_dim={wm.latent_dim}")

    # dump the shared latent cache for the metric learners
    if args.cache_out:
        LatentCache(
            z=feats.clone(),
            episode_idx=ep,
            step_idx=st,
            meta={"wm": args.output.model_name, "dataset": args.dataset},
        ).save(args.cache_out)

    # train predictor (+ action encoder + proj) on cached feature sequences
    episodes = episode_index(ep, st)
    params = list(wm.predictor.parameters()) + list(wm.action_encoder.parameters()) + list(wm.proj.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    win = args.history + args.n_preds
    h = args.history
    wm.train()
    for step in range(args.steps):
        f, a = sample_windows(episodes, feats, acts, win, args.batch_size, rng)
        f, a = f.to(device), a.to(device)
        z = torch.as_tensor(wm.proj(f))
        act_emb = torch.as_tensor(wm.action_encoder(a))
        pred = z[:, h - 1]
        loss = torch.zeros((), device=device)
        for k in range(args.n_preds):
            pred = wm.predictor(pred.unsqueeze(1), act_emb[:, h - 1 + k : h + k])[:, 0]
            loss = loss + F.mse_loss(pred, z[:, h + k].detach())
        loss = loss / args.n_preds
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            logger.info(f"DINO-WM step {step}/{args.steps}: prediction_loss={loss.item():.6f}")

    wm.eval()
    config = {
        "_target_": "rlp.core.world_model.dinowm.DinoWM",
        "backbone_name": args.backbone,
        "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim,
        "act_emb_dim": args.act_emb_dim,
        "latent_dim": wm.latent_dim,
        "obs_key": "pixels",
        "goal_key": "goal",
    }
    save_pretrained(wm.cpu(), run_name=args.output.model_name, config=config, cache_dir=args.run.directory)
    message = f"DinoWM saved to {args.run.checkpoints}/{args.output.model_name}"
    if args.cache_out:
        message += f"; feature cache={args.cache_out}"
    logger.success(message)


def main() -> object:
    return run_hydra(dispatch, config_name="train/dino")


if __name__ == "__main__":
    main()
