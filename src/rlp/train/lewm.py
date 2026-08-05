from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg

from rlp.config import dispatch, run_hydra
from rlp.logging import logger

from .callbacks import NonFiniteGradientGuard, PortableCheckpointCallback
from .tracking import make_logger
from .transforms import image_preprocessor


def get_img_preprocessor(source: str, target: str, img_size: int = 224) -> Any:
    return image_preprocessor(source, target, img_size)


def lejepa_forward(self: Any, batch: dict[str, torch.Tensor], stage: str, cfg: DictConfig) -> dict[str, torch.Tensor]:
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.core.world_model.history_size
    n_preds = cfg.core.world_model.num_predictions
    lambd = cfg.train.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    raw_output = self.model.encode(batch)
    if not isinstance(raw_output, dict) or not all(torch.is_tensor(value) for value in raw_output.values()):
        raise TypeError("LeWM encode must return a tensor mapping")
    output = cast(dict[str, torch.Tensor], raw_output)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    # --- prediction loss: 1-step base (+ optional free-running rollout loss) ---
    # rollout_len K>0 adds a multi-step term that mirrors the solver's
    # rollout_traj: roll the WM's OWN predictions forward K steps under the real
    # actions and match the encoder targets (stop-grad). This trains the WM on
    # its free-running rollout distribution — the exact regime the planner uses —
    # so it can't fabricate a plausible near-goal terminal over a multi-step plan.
    K = cfg.core.world_model.rollout_length
    if K > 0:
        pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
        output["pred_loss"] = (pred_emb - emb[:, 1 : ctx_len + 1]).pow(2).mean()
        embs = [emb[:, i] for i in range(ctx_len)]
        roll = torch.zeros((), device=emb.device)
        for k in range(K):
            win_e = torch.stack(embs[-ctx_len:], dim=1)
            nxt = self.model.predict(win_e, act_emb[:, k : k + ctx_len])[:, -1]
            roll = roll + (nxt - emb[:, ctx_len + k].detach()).pow(2).mean()
            embs.append(nxt)
        output["rollout_loss"] = roll / K
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # label
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
        output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
    if K > 0:
        output["loss"] = output["loss"] + cfg.core.world_model.rollout_weight * output["rollout_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def _run(cfg: DictConfig) -> None:
    #########################
    ##       dataset       ##
    #########################

    raw_dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    if not isinstance(raw_dataset_cfg, Mapping):
        raise TypeError("data.dataset must resolve to a mapping")
    dataset_cfg: dict[str, Any] = {str(key): value for key, value in raw_dataset_cfg.items()}
    dataset_name = str(dataset_cfg.pop("name"))
    cache_dir = cfg.data.cache_dir
    location = f"local cache: {cache_dir}" if cache_dir else "default location"
    logger.info(f'Loading dataset "{dataset_name}" from {location}')
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.data.image_size)]

    action_stats_pin = cfg.data.action_stats
    expert_action_stats = (
        [0.010831, -0.003126, 0.002633, 0.000422, 0.15846],
        [0.2887, 0.392736, 0.641535, 0.391823, 0.249935],
    )

    def make_normalizer(column: str) -> Any:
        if column != "action" or not action_stats_pin:
            return get_column_normalizer(dataset, column, column)

        import json

        import numpy as np
        from stable_pretraining.data.transforms import WrapTorchTransform
        from stable_worldmodel.data.normalization import ZScoreScaler

        if str(action_stats_pin) == "expert":
            mean, std = expert_action_stats
        else:
            with Path(action_stats_pin).open() as file:
                statistics = json.load(file)
            mean, std = statistics["mean"], statistics["std"]
        scaler = ZScoreScaler(
            mean=np.asarray(mean, dtype=np.float32).reshape(1, -1),
            std=np.asarray(std, dtype=np.float32).reshape(1, -1),
        )
        logger.info(f"Fixed action statistics mean={mean} std={std}")
        return WrapTorchTransform(scaler, source=column, target=column)

    for col in cfg.data.dataset.keys_to_load:
        if col.startswith("pixels"):
            continue
        transforms.append(make_normalizer(col))

    for col in cfg.data.dataset.keys_to_merge:
        transforms.append(make_normalizer(col))

    compose = cast(Callable[..., Any], spt.data.transforms.Compose)
    transform = compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.runtime.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.data.train_split, 1 - cfg.data.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.data.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.data.loader}
    val_cfg["shuffle"] = False
    val_cfg["drop_last"] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.core.world_model.architecture, resolve=True))
    model_cfg.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    world_model = hydra.utils.instantiate(model_cfg)

    init_weights = cfg.train.initial_weights
    if init_weights:
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        world_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Initialized LeWM weights from {init_weights}")

    total_steps = cfg.train.trainer.max_epochs * len(train)
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.train.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.train.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_dir = Path(cfg.run.directory)
    experiment_logger = make_logger(cfg)
    hyperparameters = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(hyperparameters, dict):
        raise TypeError("training configuration must resolve to a dictionary")
    experiment_logger.log_hyperparams({str(key): value for key, value in hyperparameters.items()})

    object_dump_callback = PortableCheckpointCallback(
        run_name=cfg.output.model_name,
        config=cast(DictConfig, model_cfg),
        cache_dir=run_dir,
        epoch_interval=1,
    )
    lightning_checkpoint = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=Path(cfg.run.checkpoints) / "lightning",
        filename="{epoch}-{step}",
        save_last=True,
    )

    trainer = pl.Trainer(
        **cfg.train.trainer,
        callbacks=[object_dump_callback, NonFiniteGradientGuard(), lightning_checkpoint],
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
        data=data_module,
        ckpt_path=str(ckpt_path) if ckpt_path.exists() else None,  # type: ignore[arg-type]  # Manager accepts None to disable resume despite its narrow annotation.
    )

    manager()
    return


def run() -> object:
    """Launch LeWM training with the repository-level Hydra config."""
    return run_hydra(dispatch, config_name="train/lewm")


if __name__ == "__main__":
    run()
