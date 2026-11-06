"""
fine_tunable.py
---------------
Shared shape of the fine-tunable models: a pretrained base model (optionally wrapped
by a peft LoRA adapter) plus a correction head.

    pred = base_model(context) + head(context_features)

Subclasses only implement `_base_predict`; training mode, trainable parameters and
saving are identical for every base model.
"""

from abc import abstractmethod
from typing import List

import torch
import torch.nn as nn

from ..data import Batch
from .base import Forecaster


class FineTunableForecaster(Forecaster):
    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        context_len: int,
        horizon_len: int,
        device: str,
        trainable: bool,
    ):
        # `model` is what gets switched to train/eval mode and saved: the peft model
        # when an adapter is attached, otherwise the plain base model.
        self.model = model
        self.head = head
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.device = device
        self.is_trainable = trainable

    @abstractmethod
    def _base_predict(self, batch: Batch) -> torch.Tensor:
        """Base model forecast (batch, horizon_len), float32, on `self.device`."""

    def predict(self, batch: Batch) -> torch.Tensor:
        delta = self.head(batch.context_features.to(self.device).float())
        return self._base_predict(batch) + delta

    def set_training(self, mode: bool) -> None:
        self.model.train(mode)
        self.head.train(mode)

    def trainable_parameters(self) -> List[nn.Parameter]:
        if not self.is_trainable:
            return super().trainable_parameters()
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.head.parameters())

    def save(self, output_dir: str) -> None:
        if not self.is_trainable:
            return super().save(output_dir)
        self.model.save_pretrained(output_dir)
        self.head.save(output_dir)
