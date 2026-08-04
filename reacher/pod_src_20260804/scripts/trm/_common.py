"""Shared helpers for the TRM pipeline scripts."""

from __future__ import annotations

import numpy as np
import torch

import stable_worldmodel as swm
from stable_worldmodel.wm.utils import load_pretrained


def pick_device(name: str = "auto") -> str:
    if name and name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_statewm(wm) -> bool:
    return type(wm).__name__ == "StateWM"


def build_featurizer(wm, device: str = "cpu", img_size: int = 224, train_res: int | None = None):
    """Return ``featurizer(rows) -> (B, D)`` latents for caching.

    Dispatches on world-model type: low-dim ``StateWM`` encodes ``obs_key``
    directly; pixel WMs (LeWM) decode + ImageNet-normalise images first.

    ``train_res``: bottleneck images through the checkpoint's native training
    resolution before the final resize (e.g. 64 for OGBench play retrains,
    trained on 64px frames upscaled to 224). None = no-op. Must match the
    ``eval.train_res`` used at plan time so cache and deploy share a domain.
    """
    wm = wm.to(device).eval()

    if is_statewm(wm):
        obs_key = wm.obs_key

        @torch.no_grad()
        def featurize(rows):
            s = torch.as_tensor(np.asarray(rows[obs_key]).astype(np.float32))
            s = s.reshape(s.shape[0], -1).to(device)
            return wm.encode({obs_key: s.unsqueeze(1)})["emb"][:, 0]

        return featurize

    # pixel world model (LeWM / DINO-WM): decode images and ImageNet-normalise
    from io import BytesIO
    from PIL import Image
    from torchvision.transforms import v2 as T

    _steps = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train_res and int(train_res) != int(img_size):
        _steps.append(T.Resize(size=int(train_res)))
    _steps.append(T.Resize(size=img_size))
    tf = T.Compose(_steps)

    def _decode(p):
        if isinstance(p, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(p))))
        return np.asarray(p)

    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    wants_proprio = getattr(wm, "wants_proprio", False)

    @torch.no_grad()
    def featurize(rows):
        px = rows["pixels"]
        if isinstance(px, np.ndarray) and px.dtype == object:
            # lance returns a ragged object array; decode elements (raw arrays
            # pass through) and re-stack so the vectorized path below applies
            px = np.stack([_decode(p) for p in px])
        if (isinstance(px, np.ndarray) and px.dtype == np.uint8 and px.ndim == 4
                and px.shape[1] == img_size and px.shape[2] == img_size
                and px.shape[3] == 3):
            # raw already-sized frames: vectorized on-device normalize (the
            # per-image Compose below is identical math but ~100x slower)
            x = torch.from_numpy(px).to(device).permute(0, 3, 1, 2).float().div_(255)
            if train_res and int(train_res) != int(img_size):
                x = torch.nn.functional.interpolate(
                    x, size=int(train_res), mode="bilinear", antialias=True
                )
                x = torch.nn.functional.interpolate(
                    x, size=int(img_size), mode="bilinear"
                )
            imgs = (x - _mean) / _std
        else:
            imgs = torch.stack([tf(_decode(p)) for p in px]).to(device)  # (B,C,H,W)
        enc_in = {"pixels": imgs.unsqueeze(1)}  # (B,1,C,H,W)
        if wants_proprio:
            pro = np.asarray(rows["proprio"], dtype=np.float32).reshape(imgs.shape[0], -1)
            enc_in["proprio"] = torch.from_numpy(pro).unsqueeze(1).to(device)  # (B,1,P)
        out = wm.encode(enc_in)
        return out["emb"][:, 0]

    return featurize


def load_wm(name: str, cache_dir: str | None = None, device: str = "cpu"):
    wm = load_pretrained(name, cache_dir=cache_dir)
    wm = wm.to(device).eval()
    wm.requires_grad_(False)
    return wm
