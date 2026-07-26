"""PLDM fine-tune trainer = pldm.py + Dyna patches (parallel to lewm_expert.py):
  1. NestedResize (nested-dict-aware image resize)
  2. GradGuard (skip non-finite optimizer steps)
  3. INIT_WEIGHTS env hook -> world_model.load_state_dict(strict) into the raw PLDM
  4. action_stats_pin (fixed action z-score; 'expert' or a JSON path)
  5. SaveCkptCallback constructed with cfg.model (architecture-only config;
     avoids the training-config-shadowing bug that broke load_pretrained on lewm)
IDMHead is NOT ported: PLDM already has a native idm module + PLDMLoss.
"""
import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from functools import partial

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained

from torchvision.transforms import v2 as _tv2


class NestedResize(dt.transforms.Transform):
    def __init__(self, size, source='image', target='image'):
        super().__init__()
        self.resize = _tv2.Resize(size)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(
            x, self.resize(self.nested_get(x, self.source)), self.target
        )
        return x


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = NestedResize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class GradGuard(Callback):
    """Skip optimizer steps with non-finite gradients (NaN-poisoning guard)."""

    def __init__(self):
        self.skipped = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                sq = sq + p.grad.detach().float().pow(2).sum()
        if not torch.isfinite(torch.as_tensor(sq)):
            optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            print(f'[grad-guard] non-finite grads, step skipped '
                  f'(total {self.skipped})', flush=True)


class SaveCkptCallback(Callback):
    """Save checkpoint each epoch via save_pretrained. cfg = ARCHITECTURE cfg."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def pldm_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    inpt_emb = emb[:, : cfg.wm.history_size]  # (B, T-1, D)
    inpt_act = act_emb[:, : cfg.wm.history_size]
    tgt_emb = emb[:, cfg.wm.num_preds :]  # (B, T-1, patches, dim)
    pred_emb = self.model.predict(inpt_emb, inpt_act)

    output['idm_emb'] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output['act_label'] = batch['action'][:, :-1].detach()
    output['act_pred'] = self.idm(output['idm_emb'])
    output['pred_loss'] = (pred_emb - tgt_emb).square().mean()
    output['temp_straight_loss'] = self.path_straight(emb)
    output.update(self.pldm(emb, output['act_pred'], output['act_label']))

    output['loss'] = output['pred_loss']
    for k, v in cfg.loss.items():
        loss_key = f'{k}_loss'
        if not v.enabled or (loss_key not in output):
            continue
        output['loss'] = output['loss'] + v.weight * output[loss_key]

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='pldm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(f'Loading dataset "{dataset_name}"')
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    img_processor = get_img_preprocessor('pixels', 'pixels', cfg.img_size)

    extra_transforms = []
    _pin = cfg.get('action_stats_pin', None)
    _EXPERT = ([0.010831, -0.003126, 0.002633, 0.000422, 0.15846],
               [0.2887, 0.392736, 0.641535, 0.391823, 0.249935])

    def _make_norm(col):
        if col == 'action' and _pin:
            import json as _json
            import numpy as _np
            from stable_worldmodel.data.normalization import ZScoreScaler
            from stable_pretraining.data.transforms import WrapTorchTransform
            if str(_pin) == 'expert':
                _m, _s = _EXPERT
            else:
                _d = _json.load(open(_pin))
                _m, _s = _d['mean'], _d['std']
            _scaler = ZScoreScaler(
                mean=_np.asarray(_m, dtype=_np.float32).reshape(1, -1),
                std=_np.asarray(_s, dtype=_np.float32).reshape(1, -1),
            )
            print(f'[action-pin] fixed action stats mean={_m} std={_s}',
                  flush=True)
            return WrapTorchTransform(_scaler, source=col, target=col)
        return get_column_normalizer(dataset, col, col)

    for col in cfg.data.dataset.keys_to_load:
        if col in ['pixels']:
            continue
        extra_transforms.append(_make_norm(col))

    if hasattr(cfg.data.dataset, 'keys_to_merge'):
        for col in cfg.data.dataset.keys_to_merge:
            extra_transforms.append(_make_norm(col))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col in ['pixels']:
                continue
            setattr(cfg.wm, f'{col}_dim', dataset.get_dim(col))

        effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
        cfg.model.action_encoder.input_dim = effective_act_dim
        cfg.idm.input_dim = 2 * cfg.wm.embed_dim
        cfg.idm.output_dim = effective_act_dim

    transform = spt.data.transforms.Compose(img_processor, *extra_transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )
    train = DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################
    world_model = hydra.utils.instantiate(cfg.model)
    idm = hydra.utils.instantiate(cfg.idm)

    # ---- INIT_WEIGHTS: fine-tune from a pretrained PLDM checkpoint ----
    _init_w = os.environ.get('INIT_WEIGHTS')
    if _init_w:
        _sd = torch.load(_init_w, map_location='cpu', weights_only=True)
        world_model.load_state_dict(_sd, strict=True)  # raw PLDM, not spt.Module
        print(f'[init-weights] PLDM fine-tune init from {_init_w}', flush=True)

    models = {'model': world_model, 'idm': idm}
    losses = {'pldm': PLDMLoss(), 'path_straight': TemporalStraighteningLoss()}

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {}
    for model_name in models.keys():
        optimizers[f'{model_name}_opt'] = {
            'modules': str(model_name),
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        **models, **losses, forward=partial(pldm_forward, cfg=cfg), optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################
    run_id = cfg.get('subdir') or ''
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    logging.info(f'Run ID: {run_id}')

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    # cfg.model = resolved ARCHITECTURE config (has _target_ at top) -> the saved
    # config.json is load_pretrained-compatible (no training-config shadowing).
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, GradGuard()],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()
    return


if __name__ == '__main__':
    run()
