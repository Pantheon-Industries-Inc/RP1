"""Train a PLDM world model (the LeWM paper's PLDM baseline) on a logged dataset.

Same ViT-tiny encoder / transformer predictor as LeWM -- the two classes share
all 303 parameter names, so the checkpoint is saved with the LeWM-target
architecture config and loads through every LeWM code path (world-model
eval, latent caching, RLP) with no conversion. Only the objective differs,
reproduced from the authors' recipe (``stable-worldmodel/scripts/train/pldm.py``
+ ``config/pldm.yaml`` on Value_Metric_LeWM@eval-sweep):

    loss = pred_loss (1-step latent prediction)
         + std * std_loss + std_t * std_t_loss        (VCReg variance terms)
         + cov * cov_loss + cov_t * cov_t_loss        (VCReg covariance terms)
         + temp_align * ||z_t - z_{t+1}||^2           (temporal alignment)
         + idm * ||IDM([z_t, z_{t+1}]) - a_t||^2      (inverse dynamics head)
         + temp_straight * (-cos(v_t, v_{t+1}))       (optional straightening)

with the authors' default weights (std 18, std_t 0.7, cov 12, cov_t 0,
temp_align 0.2, idm 0, temp_straight off). The IDM head is a separate module
with its own optimizer, as upstream.

Data: any dataset ``swm.data.load_dataset`` reads (lance, HDF5 ...).
``train_episodes`` restricts training to the first N episodes (the held-out
protocol: PushT evaluates on episodes >= 16000).

    pixi run train model=pldm dataset=$RLP_DATA_HOME/datasets/pusht/pusht_expert_train.h5 \
        train_episodes=16000 train.trainer.max_epochs=20
"""

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
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss

from rlp.config import dispatch, run_hydra
from rlp.logging import logger

from .callbacks import NonFiniteGradientGuard, PortableCheckpointCallback
from .tracking import make_logger
from .transforms import image_preprocessor


def pldm_forward(self: Any, batch: dict[str, torch.Tensor], stage: str, cfg: DictConfig) -> dict[str, torch.Tensor]:
    """Encode observations, predict the next latent, apply the PLDM losses."""
    ctx_len = cfg.core.world_model.history_size
    n_preds = cfg.core.world_model.num_predictions

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    raw_output = self.model.encode(batch)
    if not isinstance(raw_output, dict) or not all(torch.is_tensor(raw_output.get(k)) for k in ("emb", "act_emb")):
        raise TypeError("world-model encode must return a mapping with tensor 'emb' and 'act_emb'")
    output = cast(dict[str, torch.Tensor], {k: v for k, v in raw_output.items() if torch.is_tensor(v)})

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    pred_emb = self.model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    output["pred_loss"] = (pred_emb - emb[:, n_preds:]).pow(2).mean()

    # inverse dynamics head on consecutive latent pairs (upstream: cat[z_{t+1}, z_t])
    output["idm_emb"] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output["act_label"] = batch["action"][:, :-1].detach()
    output["act_pred"] = self.idm(output["idm_emb"])
    output["temp_straight_loss"] = self.path_straight(emb)
    output.update(self.pldm(emb, output["act_pred"], output["act_label"]))

    loss = output["pred_loss"]
    for name, term in cfg.train.loss.items():
        key = f"{name}_loss"
        if not term.enabled or key not in output:
            continue
        loss = loss + float(term.weight) * output[key]
    output["loss"] = loss

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
    logger.info(f'Loading dataset "{dataset_name}"')
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)

    transforms: list[Any] = [image_preprocessor("pixels", "pixels", cfg.data.image_size)]
    for col in cfg.data.dataset.keys_to_load:
        if str(col).startswith("pixels"):
            continue
        transforms.append(get_column_normalizer(dataset, col, col))
    for col in cfg.data.dataset.keys_to_merge:
        transforms.append(get_column_normalizer(dataset, col, col))
    compose = cast(Callable[..., Any], spt.data.transforms.Compose)
    dataset.transform = compose(*transforms)

    # Held-out protocol: train on episodes [0, train_episodes) only. Clips are
    # (episode, start) pairs, so filter by episode id before the random split.
    n_train_eps = cfg.get("train_episodes")
    if n_train_eps is not None:
        keep = [i for i, (ep, _start) in enumerate(dataset.clip_indices) if ep < int(n_train_eps)]
        if not keep:
            raise ValueError(f"no clips in episodes < {n_train_eps}")
        logger.info(f"Episode cap {n_train_eps}: {len(keep)}/{len(dataset)} clips kept")
        pool: Any = torch.utils.data.Subset(dataset, keep)
    else:
        pool = dataset

    rnd_gen = torch.Generator().manual_seed(cfg.runtime.seed)
    train_set, val_set = spt.data.random_split(
        pool, lengths=[cfg.data.train_split, 1 - cfg.data.train_split], generator=rnd_gen
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.data.loader, generator=rnd_gen)
    val_cfg = {**cfg.data.loader}
    val_cfg["shuffle"] = False
    val_cfg["drop_last"] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################
    # Saved with the LeWM-target architecture (identical parameters), so the
    # exported checkpoint is a drop-in world model for the whole eval stack.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.core.world_model.architecture, resolve=True))
    act_dim = int(dataset.get_dim("action"))
    eff_act_dim = int(cfg.data.dataset.frameskip) * act_dim
    model_cfg.action_encoder.input_dim = eff_act_dim
    world_model = hydra.utils.instantiate(model_cfg)

    embed_dim = int(model_cfg.predictor.output_dim)
    idm = hydra.utils.instantiate(
        {
            "_target_": "stable_worldmodel.wm.lewm.module.MLP",
            "input_dim": 2 * embed_dim,
            "hidden_dim": int(cfg.train.idm_hidden_dim),
            "output_dim": eff_act_dim,
        }
    )

    init_weights = cfg.train.initial_weights
    if init_weights:
        state_dict = torch.load(init_weights, map_location="cpu", weights_only=True)
        world_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Initialized PLDM weights from {init_weights}")

    total_steps = cfg.train.trainer.max_epochs * len(train)

    def opt(module_name: str) -> dict[str, Any]:
        return {
            "modules": module_name,
            "optimizer": dict(cfg.train.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        model=world_model,
        idm=idm,
        pldm=PLDMLoss(),
        path_straight=TemporalStraighteningLoss(),
        forward=partial(pldm_forward, cfg=cfg),
        optim={"model_opt": opt("model"), "idm_opt": opt("idm")},
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
        dirpath=Path(cfg.run.checkpoints) / "lightning", filename="{epoch}-{step}", save_last=True
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
        module=module,
        data=data_module,
        ckpt_path=str(ckpt_path) if ckpt_path.exists() else None,  # type: ignore[arg-type]
    )
    manager()


def run() -> object:
    """Launch PLDM training with the repository-level Hydra config."""
    return run_hydra(dispatch, config_name="train/pldm")


if __name__ == "__main__":
    run()
