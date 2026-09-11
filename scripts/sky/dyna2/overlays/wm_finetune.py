"""Anchored world-model fine-tune for Dyna v2 (cube, LeWM or PLDM class).

Derived from stable-worldmodel/scripts/train/lewm_expert.py (the trainer that
produced every shipped Dyna WM) with three additions the shipped WMs lacked:

  * native PLDM objective when launched with ``--config-name pldm`` (PLDMLoss
    std/cov/temp_align + 1-step prediction), so the PLDM base is no longer
    fine-tuned under LeWM's SIGReg;
  * ANCHOR_WEIGHT>0: latent anchor ``||enc_ft(x) - enc_base(x)||^2`` against a
    frozen copy of the INIT_WEIGHTS model, on every frame of the mixture, so the
    representation the critic/actor are rebuilt on cannot drift away from the
    base where the on-policy data is silent (long-horizon segments);
  * FREEZE_ENCODER=1: train predictor + action encoder only (hard anchor).

Kept verbatim: INIT_WEIGHTS strict load, action_stats_pin, GradGuard,
per-epoch save with the ARCHITECTURE config, LR probe.
Env: INIT_WEIGHTS (required), ANCHOR_WEIGHT (default 0), FREEZE_ENCODER (0/1),
     EXPORT_CONFIG (optional path: config.json to write next to the weights
     instead of cfg.model -- used to export PLDM fine-tunes with the
     LeWM-target config the eval stack loads pldm_cube with).
"""
import copy
import os
import re
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import PLDMLoss, SIGReg, TemporalStraighteningLoss
from stable_worldmodel.wm.utils import save_pretrained
from torchvision.transforms import v2 as _tv2


class NestedResize(dt.transforms.Transform):
    def __init__(self, size, source='image', target='image'):
        super().__init__()
        self.resize = _tv2.Resize(size)
        self.source = source
        self.target = target

    def __call__(self, x):
        self.nested_set(x, self.resize(self.nested_get(x, self.source)), self.target)
        return x


def get_img_preprocessor(source, target, img_size=224):
    stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**stats, source=source, target=target)
    return dt.transforms.Compose(to_image, NestedResize(img_size, source=source, target=target))


class GradGuard(Callback):
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
            print(f'[grad-guard] non-finite grads, step skipped (total {self.skipped})', flush=True)


class LRProbe(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        for i, opt in enumerate(trainer.optimizers):
            print(f'[lr-probe] epoch {trainer.current_epoch} opt{i} lr={opt.param_groups[0]["lr"]:.6g}', flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        for i, opt in enumerate(trainer.optimizers):
            print(f'[lr-probe] end of epoch {trainer.current_epoch} opt{i} lr={opt.param_groups[0]["lr"]:.6g}', flush=True)


class LossProbe(Callback):
    """Print running means of every logged loss at epoch end (stdout survives)."""

    def on_train_epoch_end(self, trainer, pl_module):
        m = {k: float(v) for k, v in trainer.callback_metrics.items() if 'loss' in k}
        print(f'[loss-probe] epoch {trainer.current_epoch} ' + ' '.join(f'{k}={v:.5g}' for k, v in sorted(m.items())), flush=True)


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval=1):
        super().__init__()
        self.run_name, self.cfg, self.epoch_interval = run_name, cfg, epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero and ((trainer.current_epoch + 1) % self.epoch_interval == 0
                                       or (trainer.current_epoch + 1) == trainer.max_epochs):
            save_pretrained(pl_module.model, run_name=self.run_name, config=self.cfg,
                            filename=f'weights_epoch_{trainer.current_epoch + 1}.pt')


def _remap_old_vit(k):
    m = re.match(r'encoder\.encoder\.layer\.(\d+)\.(.*)', k)
    if not m:
        return k
    n, rest = m.groups()
    rest = (rest.replace('attention.attention.query', 'attention.q_proj')
                .replace('attention.attention.key', 'attention.k_proj')
                .replace('attention.attention.value', 'attention.v_proj')
                .replace('attention.output.dense', 'attention.o_proj')
                .replace('intermediate.dense', 'mlp.fc1')
                .replace('output.dense', 'mlp.fc2'))
    return f'encoder.layers.{n}.{rest}'


def _anchor_emb(anchor, batch):
    with torch.no_grad():
        return anchor.encode({'pixels': batch['pixels']})['emb']


def lewm_forward(self, batch, stage, cfg):
    ctx_len, n_preds = cfg.wm.history_size, cfg.wm.num_preds
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output['emb'], output['act_emb']
    pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    output['pred_loss'] = (pred_emb - emb[:, n_preds:]).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + cfg.loss.sigreg.weight * output['sigreg_loss']
    if ANCHOR_W > 0:
        output['anchor_loss'] = (emb - _anchor_emb(self.anchor, batch)).pow(2).mean()
        output['loss'] = output['loss'] + ANCHOR_W * output['anchor_loss']
    self.log_dict({f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k}, on_step=True, sync_dist=True)
    return output


def pldm_forward(self, batch, stage, cfg):
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output['emb'], output['act_emb']
    H = cfg.wm.history_size
    pred_emb = self.model.predict(emb[:, :H], act_emb[:, :H])
    output['idm_emb'] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output['act_label'] = batch['action'][:, :-1].detach()
    output['act_pred'] = self.idm(output['idm_emb'])
    output['pred_loss'] = (pred_emb - emb[:, cfg.wm.num_preds:]).square().mean()
    output['temp_straight_loss'] = self.path_straight(emb)
    output.update(self.pldm(emb, output['act_pred'], output['act_label']))
    output['loss'] = output['pred_loss']
    for k, v in cfg.loss.items():
        lk = f'{k}_loss'
        if v.enabled and lk in output:
            output['loss'] = output['loss'] + v.weight * output[lk]
    if ANCHOR_W > 0:
        output['anchor_loss'] = (emb - _anchor_emb(self.anchor, batch)).pow(2).mean()
        output['loss'] = output['loss'] + ANCHOR_W * output['anchor_loss']
    self.log_dict({f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k}, on_step=True, sync_dist=True)
    return output


ANCHOR_W = float(os.environ.get('ANCHOR_WEIGHT', '0'))
_EXPERT = ([0.010831, -0.003126, 0.002633, 0.000422, 0.15846],
           [0.2887, 0.392736, 0.641535, 0.391823, 0.249935])


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    is_pldm = 'pldm' in str(cfg.model._target_).lower()
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset = swm.data.load_dataset(dataset_name, transform=None,
                                    cache_dir=os.environ.get('LOCAL_DATASET_DIR'), **dataset_cfg)
    transforms = [get_img_preprocessor('pixels', 'pixels', cfg.img_size)]
    _pin = cfg.get('action_stats_pin', None)

    def _make_norm(col):
        if col == 'action' and _pin:
            import json as _json
            import numpy as _np
            from stable_pretraining.data.transforms import WrapTorchTransform
            from stable_worldmodel.data.normalization import ZScoreScaler
            if str(_pin) == 'expert':
                _m, _s = _EXPERT
            else:
                _d = _json.load(open(_pin)); _m, _s = _d['mean'], _d['std']
            scaler = ZScoreScaler(mean=_np.asarray(_m, dtype=_np.float32).reshape(1, -1),
                                  std=_np.asarray(_s, dtype=_np.float32).reshape(1, -1))
            print(f'[action-pin] fixed action stats mean={_m} std={_s}', flush=True)
            return WrapTorchTransform(scaler, source=col, target=col)
        return get_column_normalizer(dataset, col, col)

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if not col.startswith('pixels'):
                transforms.append(_make_norm(col))
        if cfg.data.dataset.get('keys_to_merge'):
            for col in cfg.data.dataset.keys_to_merge:
                transforms.append(_make_norm(col))
        eff_act = cfg.data.dataset.frameskip * dataset.get_dim('action')
        cfg.model.action_encoder.input_dim = eff_act
        if is_pldm:
            cfg.wm.action_dim = dataset.get_dim('action')
            cfg.idm.input_dim = 2 * cfg.wm.embed_dim
            cfg.idm.output_dim = eff_act
    dataset.transform = spt.data.transforms.Compose(*transforms)

    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=gen)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, generator=gen)
    val_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    world_model = hydra.utils.instantiate(cfg.model)
    init_w = os.environ.get('INIT_WEIGHTS')
    assert init_w, 'INIT_WEIGHTS is required for a fine-tune'
    sd = torch.load(init_w, map_location='cpu', weights_only=True)
    try:
        world_model.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd = {_remap_old_vit(k): v for k, v in sd.items()}
        world_model.load_state_dict(sd, strict=True)
        print('[init-weights] loaded after old->new ViT key remap', flush=True)
    print(f'[init-weights] {"PLDM" if is_pldm else "LeWM"} fine-tune init from {init_w} '
          f'({len(sd)} tensors)', flush=True)

    modules = {'model': world_model}
    if ANCHOR_W > 0:
        anchor = copy.deepcopy(world_model).eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
        modules['anchor'] = anchor
        print(f'[anchor] latent anchor to the init weights, weight {ANCHOR_W}', flush=True)
    if os.environ.get('FREEZE_ENCODER', '0') == '1':
        n = 0
        for p in world_model.encoder.parameters():
            p.requires_grad_(False); n += 1
        print(f'[freeze] encoder frozen ({n} tensors)', flush=True)

    total_steps = cfg.trainer.max_epochs * len(train)
    def opt(name):
        return {'modules': name, 'optimizer': dict(cfg.optimizer),
                'scheduler': {'type': 'LinearWarmupCosineAnnealingLR',
                              'warmup_steps': max(1, int(0.01 * total_steps)), 'max_steps': total_steps},
                'interval': 'epoch'}
    optimizers = {'model_opt': opt('model')}
    if is_pldm:
        modules['idm'] = hydra.utils.instantiate(cfg.idm)
        modules['pldm'] = PLDMLoss()
        modules['path_straight'] = TemporalStraighteningLoss()
        optimizers['idm_opt'] = opt('idm')
        fwd = partial(pldm_forward, cfg=cfg)
    else:
        modules['sigreg'] = SIGReg(**cfg.loss.sigreg.kwargs)
        fwd = partial(lewm_forward, cfg=cfg)
    module = spt.Module(**modules, forward=fwd, optim=optimizers)

    run_id = cfg.get('subdir') or ''
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)
    export_cfg = cfg.model
    if os.environ.get('EXPORT_CONFIG'):
        export_cfg = OmegaConf.load(os.environ['EXPORT_CONFIG'])
        print(f'[export] weights will be saved with {os.environ["EXPORT_CONFIG"]}', flush=True)
    logger = WandbLogger(**cfg.wandb.config) if cfg.wandb.enabled else None
    trainer = pl.Trainer(**cfg.trainer, callbacks=[SaveCkptCallback(cfg.output_model_name, export_cfg, 1),
                                                    GradGuard(), LRProbe(), LossProbe()],
                         num_sanity_val_steps=1, logger=logger, enable_checkpointing=True)
    ckpt = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    spt.Manager(trainer=trainer, module=module, data=spt.data.DataModule(train=train, val=val),
                ckpt_path=ckpt if ckpt.exists() else None)()


if __name__ == '__main__':
    run()
