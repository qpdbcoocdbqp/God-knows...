"""Tiny stand-ins used by the tests (no pretrained weights needed)."""

import os
from typing import List

import torch
import torch.nn as nn

from jsforecast.data import Batch
from jsforecast.models import Forecaster


class LastValueForecaster(Forecaster):
    """Naive inference-only baseline: repeats the last context value."""

    def __init__(self, context_len: int, horizon_len: int = 1):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.device = "cpu"

    def predict(self, batch: Batch) -> torch.Tensor:
        return batch.context[:, -1:].repeat(1, self.horizon_len)


class LinearForecaster(Forecaster):
    """Trainable: a linear map from the context window to the horizon."""

    is_trainable = True

    def __init__(self, context_len: int, horizon_len: int = 1):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.device = "cpu"
        self.net = nn.Linear(context_len, horizon_len)
        self.modes: List[bool] = []

    def predict(self, batch: Batch) -> torch.Tensor:
        return self.net(batch.context)

    def set_training(self, mode: bool) -> None:
        self.modes.append(mode)
        self.net.train(mode)

    def trainable_parameters(self):
        return list(self.net.parameters())

    def save(self, output_dir: str) -> None:
        torch.save(self.net.state_dict(), os.path.join(output_dir, "linear.pt"))
