"""Compile-time fixtures for RLP's public typing contracts."""

from __future__ import annotations

from typing import assert_type

import torch
from omegaconf import DictConfig
from torch import nn

from rlp.core.rollout import rollout_terminal
from rlp.core.solver.lip import EncoderWorldModel
from rlp.core.value import MetricCost
from rlp.core.value.protocols import TensorInfo
from rlp.core.value.samplers import PairBatch, TransitionBatch
from rlp.core.world_model import load_pretrained
from rlp.utils.config import dispatch


def _metric_cost_contract(cost: MetricCost, info: TensorInfo, actions: torch.Tensor) -> None:
    assert_type(cost.get_cost(info, actions), torch.Tensor)


def _world_model_contract(
    model: EncoderWorldModel,
    info: dict[str, torch.Tensor],
    history: torch.Tensor,
    action_history: torch.Tensor,
    plan: torch.Tensor,
) -> None:
    assert_type(model.encode(info), dict[str, torch.Tensor])
    assert_type(rollout_terminal(model, history, action_history, plan), torch.Tensor)


def _sampler_batch_contract(pair: PairBatch, transition: TransitionBatch) -> None:
    assert_type(pair["z_i"], torch.Tensor)
    assert_type(pair["label"], torch.Tensor)
    assert_type(transition["z_tp1"], torch.Tensor)
    assert_type(transition["done"], torch.Tensor)


def _hydra_dispatch_contract(config: DictConfig) -> None:
    assert_type(dispatch(config), object)


def _checkpoint_contract(path: str) -> None:
    assert_type(load_pretrained(path), nn.Module)
