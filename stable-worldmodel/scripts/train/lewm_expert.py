"""LeWM fine-tune trainer = lewm.py + Dyna patches (parallel to pldm_expert.py):
  1. NestedResize (nested-dict-aware image resize)
  2. GradGuard (skip non-finite optimizer steps)
  3. INIT_WEIGHTS env hook -> world_model.load_state_dict(strict) into the raw LeWM
  4. action_stats_pin (fixed action z-score; 'expert' or a JSON path)
  5. SaveCkptCallback constructed with cfg.model (architecture-only config)

RECONSTRUCTED 2026-07-31. The original lived only on the AP-JP-1 pod volume and
was never committed; the volume became unattachable. Every patch here is ported
verbatim from pldm_expert.py, whose own docstring names this file as its
sibling and enumerates the same five patches -- so this is a port, not a guess.
It reproduces the trainer that produced the whole Dyna campaign (v2WM ->
dyna_*_5050, LeWM +3.6 at 6 seeds).

One deliberate deviation: the original passed the FULL cfg to SaveCkptCallback,
so the config.json beside its weights was the TRAINING config while
load_pretrained wants the ARCHITECTURE config -- the ops gotcha that killed
cache_latents 11 s in and forced every driver to copy the base's config.json by
hand (HANDOFF_20260728 sec 6). pldm_expert.py had already fixed it by passing
cfg.model; this reconstruction adopts the fix. Drivers that still copy the base
config afterwards remain correct -- the two agree.
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
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2 as _tv2

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


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


class LRProbe(Callback):
    """Print the ACTUAL optimizer lr at each epoch boundary.

    Added 2026-07-31 because the scheduler is parameterised in optimizer steps
    (warmup_steps ~345, max_steps ~34.5k) but Lightning is told
    ``'interval': 'epoch'``, so with max_epochs=2 it advances only 2 of its 345
    warmup steps. Simulating that config standalone gives lr 0.0 during epoch 1
    and 2.9e-08 during epoch 2 -- i.e. the nominal 1e-5 would never be reached,
    and epoch 1 (the checkpoint every Dyna driver ships) would be a no-op.
    That contradicts the observed campaign behaviour (fine-tuned WMs really did
    diverge from their bases and produced a replicated +3.6), so it is a
    suspicion, not a finding. This prints the ground truth instead of inferring
    it: if the lr really is ~0, the whole Dyna null trilogy (mixture, volume,
    duplication, coverage all flat) has a much simpler explanation than
    'the recipe is saturated'.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        for i, opt in enumerate(trainer.optimizers):
            for j, pg in enumerate(opt.param_groups):
                print(f'[lr-probe] epoch {trainer.current_epoch} '
                      f'opt{i} group{j} lr={pg["lr"]:.6g}', flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        for i, opt in enumerate(trainer.optimizers):
            print(f'[lr-probe] end of epoch {trainer.current_epoch} '
                  f'opt{i} lr={opt.param_groups[0]["lr"]:.6g}', flush=True)


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


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    K = cfg.wm.get('rollout_len', 0)
    if K > 0:
        pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
        output['pred_loss'] = (pred_emb - emb[:, 1:ctx_len + 1]).pow(2).mean()
        embs = [emb[:, i] for i in range(ctx_len)]
        roll = 0.0
        for k in range(K):
            win_e = torch.stack(embs[-ctx_len:], dim=1)
            nxt = self.model.predict(win_e, act_emb[:, k:k + ctx_len])[:, -1]
            roll = roll + (nxt - emb[:, ctx_len + k].detach()).pow(2).mean()
            embs.append(nxt)
        output['rollout_loss'] = roll / K
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # label
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
        output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()

    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
    if K > 0:
        output['loss'] = output['loss'] + cfg.wm.get(
            'rollout_weight', 1.0
        ) * output['rollout_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    # ---- action_stats_pin: fine-tuning MUST reuse the base model's action
    # z-score. Recomputing it from a Dyna mixture would silently change the
    # action space the pretrained predictor was trained in.
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

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            transforms.append(_make_norm(col))

        if cfg.data.dataset.get('keys_to_merge'):
            for col in cfg.data.dataset.keys_to_merge:
                transforms.append(_make_norm(col))

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    # ---- INIT_WEIGHTS: fine-tune from a pretrained checkpoint ----
    # strict=True on the RAW model (not the spt.Module wrapper): a silent
    # partial load here would train a half-random world model.
    _init_w = os.environ.get('INIT_WEIGHTS')
    if _init_w:
        _sd = torch.load(_init_w, map_location='cpu', weights_only=True)
        world_model.load_state_dict(_sd, strict=True)
        print(f'[init-weights] LeWM fine-tune init from {_init_w}', flush=True)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    # cfg.model = resolved ARCHITECTURE config -> the saved config.json is the
    # one load_pretrained wants (see module docstring).
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, GradGuard(), LRProbe()],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
