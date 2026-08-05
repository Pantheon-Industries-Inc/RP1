import os
import time
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2

from functools import partial
from einops import rearrange
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


torch.set_float32_matmul_precision('high')


class NestedResize(dt.transforms.Transform):
    def __init__(self, size, source='image', target='image'):
        super().__init__()
        self.resize = v2.Resize(size)
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
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero and self.epoch_interval > 0:
            epoch = trainer.current_epoch + 1
            if epoch % self.epoch_interval == 0:
                self._save(pl_module.model, f'epoch_{epoch}')

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)

        if trainer.is_global_zero:
            self._save(pl_module.model, f'final_step_{trainer.global_step}')

    def _save(self, model, suffix):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_{suffix}.pt',
        )


class TimedWandbEvalCallback(Callback):
    """Run a wall-clock validation pass and log decoder clips to W&B."""

    def __init__(self, val_loader, cfg):
        super().__init__()
        self.val_loader = val_loader
        self.cfg = cfg
        self._fixed_batch = None
        self._next_log_time = 0.0
        self._last_log_step = -1

    def _eval_cfg(self):
        return self.cfg.get('timed_eval', self.cfg.get('hourly_eval', {}))

    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

        eval_cfg = self._eval_cfg()
        if not eval_cfg.get('enabled', False):
            return
        if not eval_cfg.get('run_at_start', True):
            return
        if not trainer.is_global_zero or trainer.logger is None:
            return

        self._run_eval(trainer, pl_module, trainer.global_step)
        self._last_log_step = trainer.global_step
        self._next_log_time = time.monotonic() + float(
            eval_cfg.get('interval_seconds', 3600)
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        super().on_train_batch_end(
            trainer, pl_module, outputs, batch, batch_idx
        )

        eval_cfg = self._eval_cfg()
        if not eval_cfg.get('enabled', False):
            return
        if not trainer.is_global_zero or trainer.logger is None:
            return

        step = trainer.global_step
        first_step = int(eval_cfg.get('first_step', 0) or 0)
        if step < first_step or step == self._last_log_step:
            return

        now = time.monotonic()
        if now < self._next_log_time:
            return

        self._run_eval(trainer, pl_module, step)
        self._last_log_step = step
        self._next_log_time = now + float(
            eval_cfg.get('interval_seconds', 3600)
        )

    def _get_fixed_batch(self):
        if self._fixed_batch is None:
            self._fixed_batch = next(iter(self.val_loader))
        return self._fixed_batch

    def _move_batch(self, batch, device):
        return {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _run_eval(self, trainer, pl_module, step):
        try:
            import wandb
        except ImportError:
            wandb = None

        model = pl_module.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            val_metrics = self._validate(pl_module)
            video = self._make_visual_video(pl_module)

        if was_training:
            model.train()

        logs = {f'timed_eval/{k}': v for k, v in val_metrics.items()}
        if wandb is not None and video is not None:
            video_key = 'timed_eval/gt_future_encoder_recon'
            if self._eval_cfg().get('show_pred', True):
                video_key += '_wm_pred'
            logs[video_key] = wandb.Video(
                video,
                fps=int(self._eval_cfg().get('fps', 4)),
                format='mp4',
            )
        if wandb is not None and wandb.run is not None:
            wandb.log(logs, step=step)
            return

        experiment = getattr(trainer.logger, 'experiment', None)
        if experiment is not None and hasattr(experiment, 'log'):
            experiment.log(logs, step=step)
            return

        scalar_logs = {
            k: v for k, v in logs.items() if isinstance(v, (int, float))
        }
        if scalar_logs:
            trainer.logger.log_metrics(scalar_logs, step=step)

    def _validate(self, pl_module):
        max_batches = self._eval_cfg().get('max_val_batches', None)
        if max_batches is not None:
            max_batches = int(max_batches)

        totals = {}
        count = 0
        for batch_idx, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = self._move_batch(batch, pl_module.device)
            output = compute_lewm_output(
                pl_module, batch, stage='validate', cfg=self.cfg
            )
            for name, value in scalar_outputs(output).items():
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().cpu()
                )
            count += 1

        if count == 0:
            return {}
        return {name: total / count for name, total in totals.items()}

    def _make_visual_video(self, pl_module):
        if getattr(pl_module.model, 'decoder', None) is None:
            return None

        model = pl_module.model
        batch = self._move_batch(self._get_fixed_batch(), pl_module.device)
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)
        output = model.encode(batch)
        emb = output['emb']
        act_emb = output['act_emb']
        ctx_len = self.cfg.wm.history_size
        n_preds = self.cfg.wm.num_preds
        pred_emb = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])

        pred_len = pred_emb.size(1)
        target_pixels = batch['pixels'][:, n_preds : n_preds + pred_len]
        recon_pixels = model.decode_embeddings(
            emb[:, n_preds : n_preds + pred_len]
        )
        pred_pixels = None
        if self._eval_cfg().get('show_pred', True):
            pred_pixels = model.decode_embeddings(pred_emb)

        video = make_panel_video(
            target_pixels,
            recon_pixels,
            pred_pixels,
            num_clips=int(self._eval_cfg().get('num_clips', 2)),
        )
        return video


def denormalize_pixels(x):
    """Invert the ImageNet normalization used by get_img_preprocessor."""
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def make_panel_video(target, recon, pred=None, num_clips=2):
    """Return a labeled uint8 TCHW video panel.

    Each YAM 2x2 mosaic is unwrapped into a 3-camera strip, so each row is one
    validation clip and each column group is gt future/encoder recon/optional
    wm pred.
    """
    videos = [
        camera_strip_from_yam_mosaic(denormalize_pixels(target.detach())),
        camera_strip_from_yam_mosaic(denormalize_pixels(recon.detach())),
    ]
    group_names = ['gt future', 'encoder recon']
    if pred is not None:
        videos.append(
            camera_strip_from_yam_mosaic(denormalize_pixels(pred.detach()))
        )
        group_names.append('wm pred')
    t = min(v.size(1) for v in videos)
    n = min(num_clips, *(v.size(0) for v in videos))
    frames = []
    for ti in range(t):
        rows = []
        for bi in range(n):
            rows.append(torch.cat([v[bi, ti] for v in videos], dim=-1))
        frames.append(torch.cat(rows, dim=-2))
    video = torch.stack(frames, dim=0)
    video = video.mul(255).byte().cpu().numpy()
    return add_panel_labels(
        video, cell_h=video.shape[2] // n, group_names=tuple(group_names)
    )


def camera_strip_from_yam_mosaic(x):
    """Unwrap the 3 populated tiles of a YAM 2x2 mosaic into one row."""
    h, w = x.shape[-2:]
    if h % 2 != 0 or w % 2 != 0:
        return x

    tile_h, tile_w = h // 2, w // 2
    cameras = (
        x[..., :tile_h, :tile_w],
        x[..., :tile_h, tile_w:],
        x[..., tile_h:, :tile_w],
    )
    return torch.cat(cameras, dim=-1)


def add_panel_labels(video, cell_h, group_names, header_h=40):
    """Add lightweight labels/grid lines to a TCHW uint8 W&B video."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return video

    camera_names = ('cam 1', 'cam 2', 'cam 3')
    t, c, h, w = video.shape
    if c != 3 or len(group_names) == 0:
        return video

    group_w = w // len(group_names)
    camera_w = group_w // len(camera_names)
    labeled = []
    for frame in video:
        img = np.moveaxis(frame, 0, -1)
        canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        canvas[header_h:] = img
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        for group_idx, group_name in enumerate(group_names):
            group_x = group_idx * group_w
            draw.text((group_x + 6, 4), group_name, fill=(255, 255, 255))
            if group_idx > 0:
                draw.line(
                    (group_x, 0, group_x, h + header_h),
                    fill=(200, 200, 200),
                    width=3,
                )
            for cam_idx, cam_name in enumerate(camera_names):
                cam_x = group_x + cam_idx * camera_w
                draw.text((cam_x + 6, 22), cam_name, fill=(210, 210, 210))
                if cam_idx > 0:
                    draw.line(
                        (cam_x, header_h, cam_x, h + header_h),
                        fill=(80, 80, 80),
                        width=1,
                    )
        for y in range(header_h + cell_h, h + header_h, cell_h):
            draw.line((0, y, w, y), fill=(110, 110, 110), width=2)
        labeled.append(np.moveaxis(np.asarray(pil), -1, 0))
    return np.stack(labeled, axis=0)


def scalar_from_cfg(cfg, key, default=0.0):
    node = cfg
    for part in key.split('.'):
        if node is None or part not in node:
            return default
        node = node[part]
    return float(node)


def latent_diagnostics(z, eps=1e-4):
    """Cheap VICReg-style summaries for collapse monitoring."""
    z = z.detach().float().reshape(-1, z.size(-1))
    if z.size(0) < 2:
        zero = z.new_tensor(0.0)
        return {
            'latent_std_mean': zero,
            'latent_var_loss': zero,
            'latent_cov_loss': zero,
        }

    std = torch.sqrt(z.var(dim=0) + eps)
    centered = z - z.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (z.size(0) - 1)
    cov = cov - torch.diag(torch.diag(cov))
    return {
        'latent_std_mean': std.mean(),
        'latent_var_loss': F.relu(1 - std).mean(),
        'latent_cov_loss': cov.pow(2).sum() / z.size(1),
    }


def compute_lewm_output(self, batch, stage, cfg):
    """Encode observations, predict next states, and compute LeWM losses."""
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
    output['pred_emb'] = pred_emb

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    recon_weight = scalar_from_cfg(cfg, 'loss.reconstruction.weight', 0.0)
    if getattr(self.model, 'decoder', None) is not None:
        pred_len = pred_emb.size(1)
        recon_pixels = self.model.decode_embeddings(
            emb[:, n_preds : n_preds + pred_len].detach()
        )
        target_pixels = batch['pixels'][:, n_preds : n_preds + pred_len]
        output['recon_loss'] = F.mse_loss(
            recon_pixels.float(), target_pixels.float()
        )
        if recon_weight > 0:
            output['loss'] = output['loss'] + recon_weight * output['recon_loss']

    proprio_key = getattr(self.model, 'proprio_key', 'observation')
    proprio_weight = scalar_from_cfg(cfg, 'loss.proprio.weight', 0.0)
    if (
        getattr(self.model, 'proprio_head', None) is not None
        and proprio_key in batch
    ):
        pred_len = pred_emb.size(1)
        proprio_pred = self.model.proprio_head(
            rearrange(pred_emb.detach(), 'b t d -> (b t) d')
        )
        proprio_pred = rearrange(
            proprio_pred, '(b t) d -> b t d', b=pred_emb.size(0)
        )
        proprio_target = batch[proprio_key][
            :, n_preds : n_preds + pred_len
        ].float()
        output['proprio_pred_mse'] = F.mse_loss(
            proprio_pred.float(), proprio_target
        )
        if proprio_weight > 0:
            output['loss'] = (
                output['loss'] + proprio_weight * output['proprio_pred_mse']
            )

    diag_interval = int(cfg.get('diagnostics', {}).get('every_n_steps', 100))
    should_log_diag = stage != 'train' or (
        diag_interval > 0
        and getattr(self, 'global_step', 0) % diag_interval == 0
    )
    if should_log_diag:
        output.update(latent_diagnostics(emb))
        output['temporal_align_mse'] = F.mse_loss(
            emb[:, 1:].detach().float(), emb[:, :-1].detach().float()
        )
    return output


def scalar_outputs(output):
    return {
        k: v
        for k, v in output.items()
        if torch.is_tensor(v)
        and v.ndim == 0
        and (
            'loss' in k
            or 'mse' in k
            or k.startswith('latent_')
            or k.startswith('temporal_')
        )
    }


def lejepa_forward(self, batch, stage, cfg):
    """Lightning forward hook for train/validation steps."""
    output = compute_lewm_output(self, batch, stage, cfg)
    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in scalar_outputs(output).items()
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
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        proprio_key = cfg.model.get('proprio_key', None)
        if (
            proprio_key
            and cfg.model.get('proprio_encoder', None) is not None
        ):
            proprio_dim = dataset.get_dim(proprio_key)
            cfg.model.proprio_encoder.input_dim = proprio_dim
            if cfg.model.get('proprio_head', None) is not None:
                cfg.model.proprio_head.output_dim = proprio_dim

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

    trainer_max_steps = cfg.trainer.get('max_steps', None)
    if trainer_max_steps is not None and int(trainer_max_steps) > 0:
        total_steps = int(trainer_max_steps)
    else:
        total_steps = cfg.trainer.max_epochs * len(train)
    optimizer_cfg = OmegaConf.to_container(
        cfg.optimizer, resolve=True
    )
    scheduler_interval = optimizer_cfg.pop('scheduler_interval', 'step')
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': optimizer_cfg,
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': scheduler_interval,
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

    checkpoint_cfg = cfg.get('checkpoint', {})
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=int(checkpoint_cfg.get('epoch_interval', 1) or 0),
        step_interval=int(
            checkpoint_cfg.get('every_n_train_steps', 0) or 0
        ),
    )
    callbacks = [object_dump_callback]
    timed_eval_cfg = cfg.get('timed_eval', cfg.get('hourly_eval', {}))
    if timed_eval_cfg.get('enabled', False):
        callbacks.append(TimedWandbEvalCallback(val, cfg))

    sanity_steps = 0 if timed_eval_cfg.get('enabled', False) else 1
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=sanity_steps,
        logger=logger,
        enable_checkpointing=False,
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
