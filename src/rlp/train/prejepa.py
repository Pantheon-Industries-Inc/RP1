"""DINO-WM (PreJEPA) on the canonical OGBench cube h5 — reproduction of the
`ogbench_cube_single_dino` handoff run (weights_step_100000.pt, CEM 84.0).

Identical to prejepa.py except:
  1. `data.merge_proprio_from=[qpos,qvel]` — the registered OGB Cube Lance dataset
     stores qpos(21)/qvel(20) separately; the handoff h5 had them pre-merged
     as a 41-d `proprio` column. Merged here (float32, cached in RAM).
  2. SaveCkptCallback saves every `step_interval` steps (handoff cadence:
     10000) — the stock epoch-only cadence never fires before the
     max_steps=100000 cap (~3.5 epochs).

Launch (handoff-exact hyperparams):
  CUDA_VISIBLE_DEVICES=5 pixi run train model=prejepa \
    data.path=/data/cube_single_expert.h5 output.model_name=prejepa_cube \
    data.loader.batch_size=64 data.loader.num_workers=12 \
    train.trainer.strategy=auto train.trainer.devices=1 \
    train.trainer.precision=bf16-mixed train.trainer.max_steps=100000
"""

from collections.abc import Callable
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor

from rlp.config import dispatch, run_hydra
from rlp.logging import logger

from .callbacks import NonFiniteGradientGuard, PortableCheckpointCallback
from .tracking import make_logger
from .transforms import image_preprocessor, nested_clip, nested_resize

with suppress(ImportError):  # the canonical h5 uses hdf5plugin codecs
    import hdf5plugin  # noqa: F401

torch.set_float32_matmul_precision("high")
_NONFINITE = {"loss": 0}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_img_preprocessor(source: str, target: str, img_size: int = 224) -> Any:
    return image_preprocessor(source, target, img_size)


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor: Any, source: str = "image", target: str = "image") -> None:
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        nested_get = cast(Callable[[dict[str, Any], str], Any], self.nested_get)
        nested_set = cast(Callable[[dict[str, Any], Any, str], None], self.nested_set)
        frames = nested_get(x, self.source)
        nested_set(
            x,
            self.processor(frames, return_tensors="pt")["pixel_values_videos"].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor: torch.Tensor, action_range: list[int]) -> torch.Tensor:
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self: Any, batch: dict[str, torch.Tensor], stage: str, cfg: DictConfig) -> dict[str, torch.Tensor]:
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target="emb",
        is_video=cfg.core.world_model.backbone.is_video_encoder,
    )

    embedding = batch["emb"][:, : cfg.core.world_model.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch["emb"][:, cfg.core.world_model.num_predictions :, ...].detach()

    # Per-modality losses
    pixels_dim = batch["pixels_emb"].size(-1)
    batch["pixels_loss"] = F.mse_loss(pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim])

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f"{key}_emb"].size(-1)
        lo, hi = start, start + dim
        if key == "action":
            action_range = [lo, hi]
        else:
            batch[f"{key}_loss"] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch["actionless_emb"] = _strip_action_dims(batch["emb"], action_range)
    batch["actionless_prev_emb"] = _strip_action_dims(embedding, action_range)
    batch["actionless_pred_emb"] = _strip_action_dims(pred_embedding, action_range)
    batch["actionless_target_emb"] = _strip_action_dims(target_embedding, action_range)

    batch["loss"] = F.mse_loss(
        batch["actionless_pred_emb"],
        batch["actionless_target_emb"].detach(),
    )

    # REPLICATION_CUBE.md item 3: this dataset throws a non-finite batch
    # roughly once per tens of thousands of steps — skip it, don't die.
    if not batch["loss"].isfinite():
        _NONFINITE["loss"] += 1
        logger.warning(f"Non-finite loss; skipping batch {_NONFINITE['loss']}")
        if _NONFINITE["loss"] > 200:
            raise ValueError("too many non-finite loss batches — aborting")
        anchor = next(p for p in self.model.parameters() if p.requires_grad)
        for k in list(batch):
            if k.endswith("_loss") or k == "loss":
                batch[k] = anchor.sum() * 0.0

    self.log_dict(
        {f"{stage}/{k}": v.detach() for k, v in batch.items() if "_loss" in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _run(cfg: DictConfig) -> None:
    # --- Dataset ---
    encoding_keys = list(cfg.core.world_model.encoding.keys())
    merge_src = list(cfg.data.merge_proprio_from)
    do_merge = bool(merge_src) and "proprio" in encoding_keys

    direct_keys = [k for k in encoding_keys if not (do_merge and k == "proprio")]
    keys_to_load = ["pixels"] + direct_keys + (merge_src if do_merge else [])

    cache_dir = cfg.data.cache_dir
    location = f"local cache: {cache_dir}" if cache_dir else "default location"
    logger.info(f'Loading dataset "{cfg.data.path}" from {location}')
    dataset = swm.data.load_dataset(
        cfg.data.path,
        num_steps=cfg.data.num_steps,
        frameskip=cfg.data.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=keys_to_load,
        keys_to_cache=direct_keys,
    )

    if do_merge:
        merged = np.concatenate(
            [np.asarray(dataset.get_col_data(c), dtype=np.float32) for c in merge_src],
            axis=-1,
        )
        dataset._cache["proprio"] = merged
        dataset._keys = [k for k in dataset._keys if k not in merge_src]
        if "proprio" not in dataset._keys:
            dataset._keys.append("proprio")
        logger.info(f"Merged {merge_src} -> proprio {merged.shape} ({merged.dtype})")

    normalizers = [get_column_normalizer(dataset, col, col) for col in cfg.core.world_model.encoding]
    clip = float(cfg.data.proprio_clip)
    if clip > 0 and "proprio" in cfg.core.world_model.encoding:
        normalizers.append(nested_clip(clip, "proprio", "proprio"))
        logger.info(f"Clipping z-scored proprio to ±{clip}")

    if cfg.core.world_model.backbone.is_video_encoder:
        processor_factory = cast(Callable[..., Any], AutoVideoProcessor.from_pretrained)
        processor = processor_factory(cfg.core.world_model.backbone.name)
        compose = cast(Callable[..., Any], spt.data.transforms.Compose)
        transform = compose(
            VideoPipeline(processor, source="pixels", target="pixels"),
            nested_resize(cfg.data.image_size, source="pixels", target="pixels"),
            *normalizers,
        )
    else:
        compose = cast(Callable[..., Any], spt.data.transforms.Compose)
        transform = compose(
            get_img_preprocessor("pixels", "pixels", cfg.data.image_size),
            *normalizers,
        )
    dataset.transform = transform

    extra_dims = {}
    for key in cfg.core.world_model.encoding:
        if key not in dataset.column_names:
            raise ValueError(f"Encoding key '{key}' not found in dataset columns.")
        dim = dataset.get_dim(key)
        extra_dims[key] = dim if key != "action" else dim * cfg.data.frameskip

    rnd_gen = torch.Generator().manual_seed(cfg.runtime.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.data.train_split, 1 - cfg.data.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.core.world_model.architecture.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, "hidden_sizes")
    embed_dim = encoder.config.hidden_sizes[-1] if is_cnn else encoder.config.hidden_size
    num_patches = 1 if is_cnn else (cfg.data.image_size // cfg.core.world_model.backbone.patch_size) ** 2
    embed_dim += sum(cfg.core.world_model.encoding.values())

    if cfg.core.world_model.backbone.is_video_encoder:
        num_patches += num_patches * (cfg.data.num_steps // 4)

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.core.world_model.architecture, resolve=True))
    model_cfg.predictor.dim = embed_dim
    model_cfg.predictor.num_patches = num_patches
    model_cfg.extra_encoders = {
        "_target_": "torch.nn.ModuleDict",
        "modules": {
            key: {
                "_target_": "stable_worldmodel.wm.prejepa.module.Embedder",
                "in_chans": extra_dims[key],
                "emb_dim": int(cfg.core.world_model.encoding[key]),
            }
            for key in cfg.core.world_model.encoding
        },
    }

    world_model = hydra.utils.instantiate(model_cfg, encoder=encoder)

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={"model_opt": {"modules": "model", "optimizer": dict(cfg.train.optimizer)}},
    )

    # --- Training ---
    run_dir = Path(cfg.run.directory)
    logger.info(f"Output directory: {run_dir}")
    experiment_logger = make_logger(cfg)
    hyperparameters = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(hyperparameters, dict):
        raise TypeError("training configuration must resolve to a dictionary")
    experiment_logger.log_hyperparams({str(key): value for key, value in hyperparameters.items()})

    callbacks = [
        NonFiniteGradientGuard(max_skipped=200),
        PortableCheckpointCallback(
            run_name=cfg.output.model_name,
            config=cast(DictConfig, model_cfg),
            cache_dir=run_dir,
            epoch_interval=5,
            step_interval=cfg.output.save_every_steps,
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.pytorch.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.run.checkpoints) / "lightning",
            filename="{epoch}-{step}",
            save_last=True,
        ),
    ]
    if hasattr(spt.callbacks, "CPUOffloadCallback"):
        callbacks.insert(0, spt.callbacks.CPUOffloadCallback())

    trainer = pl.Trainer(
        **cfg.train.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=experiment_logger,
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=run_dir,
    )

    ckpt_path = Path(cfg.run.checkpoints) / "trainer.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=str(ckpt_path) if ckpt_path.exists() else None,  # type: ignore[arg-type]  # Manager accepts None to disable resume despite its narrow annotation.
    )
    manager()


def run() -> object:
    """Launch PreJEPA training with the repository-level Hydra config."""
    return run_hydra(dispatch, config_name="train/prejepa")


if __name__ == "__main__":
    run()
