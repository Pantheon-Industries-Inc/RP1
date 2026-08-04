"""DINO-WM (PreJEPA) on the canonical OGBench cube h5 — reproduction of the
`ogbench_cube_single_dino` handoff run (weights_step_100000.pt, CEM 84.0).

Identical to prejepa.py except:
  1. `+merge_proprio_from=[qpos,qvel]` — our eval h5 (cube_single_expert.h5)
     stores qpos(21)/qvel(20) separately; the handoff h5 had them pre-merged
     as a 41-d `proprio` column. Merged here (float32, cached in RAM).
  2. SaveCkptCallback saves every `step_interval` steps (handoff cadence:
     10000) — the stock epoch-only cadence never fires before the
     max_steps=100000 cap (~3.5 epochs).

Launch (handoff-exact hyperparams):
  PYTHONPATH=/workspace/code/stable-worldmodel STABLEWM_HOME=/workspace/swm_home \
  CUDA_VISIBLE_DEVICES=5 python scripts/train/prejepa_cube.py \
    dataset_name=/workspace/datasets/lewm_cube_full/cube_single_expert.h5 \
    output_model_name=ogbench_cube_single_dino_rep \
    subdir=ogbench_cube_single_dino_rep_20260719 \
    batch_size=64 num_workers=12 \
    trainer.strategy=auto trainer.devices=1 trainer.precision=bf16-mixed \
    +trainer.max_steps=100000 '+merge_proprio_from=[qpos,qvel]'
"""

import os
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.utils import save_pretrained
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor

try:  # the canonical h5 uses hdf5plugin codecs
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class NestedResize(spt.data.transforms.Transform):
    """Resize that works across stable_pretraining versions (handoff-ported)."""

    def __init__(self, size, source='image', target='image'):
        super().__init__()
        from torchvision.transforms import v2

        self.resize = v2.Resize(size)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.resize(self.nested_get(x, self.source)), self.target
        )
        return x


class NestedClip(spt.data.transforms.Transform):
    """Clamp a column after z-scoring. Our expert h5 has near-constant qpos
    dims (std ~2e-3) whose rare excursions z-score to |z|~266 and overflow the
    bf16 backward once predictor weights grow (deterministic abort ~step 15k).
    Clipping at ±10σ touches ~0.001% of values."""

    def __init__(self, bound, source, target):
        super().__init__()
        self.bound = float(bound)
        self.source = source
        self.target = target

    def __call__(self, x):
        v = self.nested_get(x, self.source)
        v = torch.as_tensor(v).clamp(-self.bound, self.bound)
        self.nested_set(x, v, self.target)
        return x


def get_img_preprocessor(source, target, img_size=224):
    stats = spt.data.dataset_stats.ImageNet
    return spt.data.transforms.Compose(
        spt.data.transforms.ToImage(**stats, source=source, target=target),
        NestedResize(img_size, source=source, target=target),
    )


class VideoPipeline(spt.data.transforms.Transform):
    def __init__(self, processor, source='image', target='image'):
        super().__init__()
        self.processor, self.source, self.target = processor, source, target

    def __call__(self, x):
        frames = self.nested_get(x, self.source)
        self.nested_set(
            x,
            self.processor(frames, return_tensors='pt')[
                'pixel_values_videos'
            ].squeeze(0),
            self.target,
        )
        return x


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

_NONFINITE = {'loss': 0, 'grad': 0}


class SkipNonFiniteGrads(Callback):
    """REPLICATION_CUBE.md item 3: rare non-finite grads on this dataset —
    clip_grad_norm_ would scale ALL grads by NaN and permanently poison Adam.
    Zero the grads before clipping so the step becomes a no-op instead."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        for p in pl_module.parameters():
            if p.grad is not None and not p.grad.isfinite().all():
                for q in pl_module.parameters():
                    if q.grad is not None:
                        q.grad.zero_()
                _NONFINITE['grad'] += 1
                logging.warning(
                    f'non-finite grads zeroed (#{_NONFINITE["grad"]}) '
                    f'at step {trainer.global_step}'
                )
                if _NONFINITE['grad'] > 200:
                    raise ValueError(
                        'too many non-finite grad steps — aborting'
                    )
                return


class SaveCkptCallback(Callback):
    """Save portable model weights on a step or epoch cadence."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        step_interval: int = 0,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self._last_saved_step = -1

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )
        if not trainer.is_global_zero or self.step_interval <= 0:
            return
        step = trainer.global_step
        if step <= 0 or step == self._last_saved_step:
            return
        if step % self.step_interval == 0:
            self._save(pl_module.model, f'step_{step}')
            self._last_saved_step = step

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if self.epoch_interval > 0 and epoch % self.epoch_interval == 0:
            self._save(pl_module.model, f'epoch_{epoch}')
        if epoch == trainer.max_epochs:
            self._save(pl_module.model, f'epoch_{epoch}')

    def _save(self, model, suffix):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_{suffix}.pt',
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _strip_action_dims(tensor, action_range):
    """Remove the action dimensions from the last axis."""
    return torch.cat(
        [tensor[..., : action_range[0]], tensor[..., action_range[1] :]],
        dim=-1,
    )


def dinowm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute losses."""
    for key in self.model.extra_encoders:
        batch[key] = torch.nan_to_num(batch[key], 0.0).squeeze()

    batch = self.model.encode(
        batch,
        target='emb',
        is_video=cfg.backbone.get('is_video_encoder', False),
    )

    embedding = batch['emb'][:, : cfg.wm.history_size, ...]
    pred_embedding = self.model.predict(embedding)
    target_embedding = batch['emb'][:, cfg.wm.num_preds :, ...].detach()

    # Per-modality losses
    pixels_dim = batch['pixels_emb'].size(-1)
    batch['pixels_loss'] = F.mse_loss(
        pred_embedding[..., :pixels_dim], target_embedding[..., :pixels_dim]
    )

    start, action_range = pixels_dim, [0, 0]
    for key in self.model.extra_encoders:
        dim = batch[f'{key}_emb'].size(-1)
        lo, hi = start, start + dim
        if key == 'action':
            action_range = [lo, hi]
        else:
            batch[f'{key}_loss'] = F.mse_loss(
                pred_embedding[..., lo:hi],
                target_embedding[..., lo:hi].detach(),
            )
        start = hi

    # Actionless embeddings (for probes and total loss)
    batch['actionless_emb'] = _strip_action_dims(batch['emb'], action_range)
    batch['actionless_prev_emb'] = _strip_action_dims(embedding, action_range)
    batch['actionless_pred_emb'] = _strip_action_dims(
        pred_embedding, action_range
    )
    batch['actionless_target_emb'] = _strip_action_dims(
        target_embedding, action_range
    )

    batch['loss'] = F.mse_loss(
        batch['actionless_pred_emb'],
        batch['actionless_target_emb'].detach(),
    )

    # REPLICATION_CUBE.md item 3: this dataset throws a non-finite batch
    # roughly once per tens of thousands of steps — skip it, don't die.
    if not batch['loss'].isfinite():
        _NONFINITE['loss'] += 1
        logging.warning(
            f'non-finite loss — skipping batch (#{_NONFINITE["loss"]})'
        )
        if _NONFINITE['loss'] > 200:
            raise ValueError('too many non-finite loss batches — aborting')
        anchor = next(p for p in self.model.parameters() if p.requires_grad)
        for k in list(batch):
            if k.endswith('_loss') or k == 'loss':
                batch[k] = anchor.sum() * 0.0

    self.log_dict(
        {f'{stage}/{k}': v.detach() for k, v in batch.items() if '_loss' in k},
        on_step=True,
        sync_dist=True,
    )
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path='./config', config_name='prejepa')
def run(cfg):
    # --- Dataset ---
    encoding_keys = list(cfg.wm.get('encoding', {}).keys())
    merge_src = list(cfg.get('merge_proprio_from') or [])
    do_merge = bool(merge_src) and 'proprio' in encoding_keys

    direct_keys = [k for k in encoding_keys if not (do_merge and k == 'proprio')]
    keys_to_load = ['pixels'] + direct_keys + (merge_src if do_merge else [])

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{cfg.dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cache_dir,
        keys_to_load=keys_to_load,
        keys_to_cache=direct_keys,
    )

    if do_merge:
        merged = np.concatenate(
            [
                np.asarray(dataset.get_col_data(c), dtype=np.float32)
                for c in merge_src
            ],
            axis=-1,
        )
        dataset._cache['proprio'] = merged
        dataset._keys = [k for k in dataset._keys if k not in merge_src]
        if 'proprio' not in dataset._keys:
            dataset._keys.append('proprio')
        logging.info(
            f'Merged {merge_src} -> proprio {merged.shape} ({merged.dtype})'
        )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get('encoding', {})
    ]
    clip = float(cfg.get('proprio_clip', 0) or 0)
    if clip > 0 and 'proprio' in cfg.wm.get('encoding', {}):
        normalizers.append(NestedClip(clip, 'proprio', 'proprio'))
        logging.info(f'clipping z-scored proprio to ±{clip}')

    if cfg.backbone.get('is_video_encoder', False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source='pixels', target='pixels'),
            NestedResize(cfg.image_size, source='pixels', target='pixels'),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor('pixels', 'pixels', cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get('encoding', {}):
            if key not in dataset.column_names:
                raise ValueError(
                    f"Encoding key '{key}' not found in dataset columns."
                )
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = (
                dim if key != 'action' else dim * cfg.frameskip
            )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    encoder.eval()
    encoder.requires_grad_(False)

    is_cnn = hasattr(encoder.config, 'hidden_sizes')
    embed_dim = (
        encoder.config.hidden_sizes[-1]
        if is_cnn
        else encoder.config.hidden_size
    )
    num_patches = 1 if is_cnn else (cfg.image_size // cfg.patch_size) ** 2
    embed_dim += sum(cfg.wm.get('encoding', {}).values())

    if cfg.backbone.get('is_video_encoder', False):
        num_patches += num_patches * (cfg.n_steps // 4)

    with open_dict(cfg):
        cfg.model.predictor.dim = embed_dim
        cfg.model.predictor.num_patches = num_patches
        cfg.model.extra_encoders = {
            '_target_': 'torch.nn.ModuleDict',
            'modules': {
                key: {
                    '_target_': 'stable_worldmodel.wm.prejepa.module.Embedder',
                    'in_chans': cfg.extra_dims[key],
                    'emb_dim': int(cfg.wm.encoding[key]),
                }
                for key in cfg.wm.get('encoding', {})
            },
        }

    world_model = hydra.utils.instantiate(cfg.model, encoder=encoder)

    world_model = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={
            'model_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer)}
        },
    )

    # --- Training ---
    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f'Run ID: {run_id}')

    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        logger = pl.pytorch.loggers.CSVLogger(
            save_dir=str(run_dir), name='csv_logs'
        )

    callbacks = [
        SkipNonFiniteGrads(),
        SaveCkptCallback(
            run_name=cfg.output_model_name,
            cfg=cfg.model,
            epoch_interval=5,
            step_interval=int(cfg.get('save_every_steps', 10000)),
        ),
        pl.pytorch.callbacks.LearningRateMonitor(logging_interval='step'),
    ]
    if hasattr(spt.callbacks, 'CPUOffloadCallback'):
        callbacks.insert(0, spt.callbacks.CPUOffloadCallback())

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == '__main__':
    run()
