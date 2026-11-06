"""Forecasting models and the factory that builds them from a `ModelConfig`."""

from typing import Optional

from ..config import DataConfig, LoraConfigOpts, ModelConfig
from .base import Forecaster
from .feature_head import FeatureCorrectionHead, NullHead, load_head
from .losses import JaneStreetWeightedMSELoss

__all__ = [
    "Forecaster", "FeatureCorrectionHead", "NullHead", "JaneStreetWeightedMSELoss",
    "load_head", "build_forecaster",
]


def _forecaster_class(model_type: str):
    # imported lazily: each pulls in its own heavy dependency
    if model_type == "timesfm3":
        from .timesfm3 import TimesFM3Forecaster

        return TimesFM3Forecaster
    from .timesfm25 import TimesFM25Forecaster

    return TimesFM25Forecaster


def build_forecaster(
    model_cfg: ModelConfig,
    *,
    lora_cfg: Optional[LoraConfigOpts] = None,
    data_cfg: Optional[DataConfig] = None,
    adapter_dir: Optional[str] = None,
    trainable: bool = False,
) -> Forecaster:
    """
    One entry point for every way of getting a forecaster:

      trainable=True          a fresh model to fine-tune (needs `lora_cfg`, `data_cfg`)
      adapter_dir=<dir>       a model saved by `Forecaster.save`
      neither                 the zero-shot pretrained model
    """
    if trainable and adapter_dir:
        raise ValueError("Resuming training from an adapter is not supported")

    cls = _forecaster_class(model_cfg.model_type)
    if trainable:
        if lora_cfg is None or data_cfg is None:
            raise ValueError("trainable=True needs lora_cfg and data_cfg")
        return cls.for_training(model_cfg, lora_cfg, num_features=len(data_cfg.feature_cols))
    if adapter_dir:
        return cls.from_adapter(model_cfg, adapter_dir)
    return cls.zero_shot(model_cfg)
