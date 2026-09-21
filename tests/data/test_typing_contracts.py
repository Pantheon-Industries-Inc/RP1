"""Compile-time fixtures for rp1's public typing contracts."""

from __future__ import annotations

from typing import assert_type

import torch
from omegaconf import DictConfig
from torch import nn

from rp1.core.agent.solver.base import EncoderWorldModel
from rp1.core.agent.value import MetricCost
from rp1.core.agent.value.base import TensorInfo
from rp1.core.world_model.rollout import rollout_terminal
from rp1.training.harness.checkpointing import load_pretrained
from rp1.training.phases.agent.samplers import PairBatch, TransitionBatch
from rp1.utils.config import dispatch


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
