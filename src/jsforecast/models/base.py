"""
base.py
-------
The interface every model implements. Training, prediction and backtesting only
talk to a `Forecaster`, so adding a model means adding one subclass.
"""

from abc import ABC, abstractmethod
from typing import List

import torch

from ..data import Batch


class Forecaster(ABC):
    """Maps a `Batch` to a point forecast of shape (batch, horizon_len)."""

    context_len: int
    horizon_len: int
    device: str

    # Whether `trainable_parameters` / `save` are meaningful.
    is_trainable: bool = False

    @abstractmethod
    def predict(self, batch: Batch) -> torch.Tensor:
        """Point forecast (batch, horizon_len). Differentiable when the model is trainable."""

    def set_training(self, mode: bool) -> None:
        """Switches train/eval mode (dropout etc.). No-op for inference-only models."""

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        raise TypeError(f"{type(self).__name__} is inference-only")

    def save(self, output_dir: str) -> None:
        raise TypeError(f"{type(self).__name__} is inference-only")
