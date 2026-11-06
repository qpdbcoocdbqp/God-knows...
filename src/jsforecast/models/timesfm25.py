"""
timesfm25.py
------------
TimesFM 2.5 via HuggingFace `transformers` (`TimesFm2_5ModelForPrediction`), with an
optional LoRA adapter (`peft`) and an optional `FeatureCorrectionHead`, following
google-research/timesfm: timesfm-forecasting/examples/finetuning/finetune_lora.py.

The model normalizes its own input (RevIN); do not normalize externally. Its built-in
loss is unweighted, so training uses `JaneStreetWeightedMSELoss` on `predict()` instead.
"""

import os

import torch

from ..config import LoraConfigOpts, ModelConfig
from ..data import Batch
from ..utils import get_device, resolve_dtype
from .feature_head import FeatureCorrectionHead, NullHead, load_head
from .fine_tunable import FineTunableForecaster

ADAPTER_CONFIG_FILENAME = "adapter_config.json"


def _load_base_model(model_cfg: ModelConfig, device: str):
    from transformers import TimesFm2_5ModelForPrediction

    return TimesFm2_5ModelForPrediction.from_pretrained(
        model_cfg.resolved_model_id,
        dtype=resolve_dtype(model_cfg.torch_dtype),
        device_map=device,
    )


class TimesFM25Forecaster(FineTunableForecaster):
    # ---- constructors -------------------------------------------------------------

    @classmethod
    def zero_shot(cls, model_cfg: ModelConfig) -> "TimesFM25Forecaster":
        """Plain pretrained checkpoint, no adapter, no head."""
        device = get_device(model_cfg.device)
        base = _load_base_model(model_cfg, device)
        return cls(
            base, NullHead(model_cfg.horizon_len).to(device),
            context_len=min(model_cfg.context_len, base.config.context_length),
            horizon_len=model_cfg.horizon_len, device=device, trainable=False,
        )

    @classmethod
    def for_training(cls, model_cfg: ModelConfig, lora_cfg: LoraConfigOpts, num_features: int) -> "TimesFM25Forecaster":
        """Pretrained checkpoint + trainable LoRA adapter (or all weights) + optional head."""
        device = get_device(model_cfg.device)
        model = _load_base_model(model_cfg, device)
        context_len = min(model_cfg.context_len, model.config.context_length)

        if model_cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                target_modules=lora_cfg.target_modules,
                lora_dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
            ))
            model.print_trainable_parameters()
        else:
            for p in model.parameters():
                p.requires_grad_(True)

        if model_cfg.use_xreg:
            head = FeatureCorrectionHead(
                num_features=num_features,
                context_len=context_len,
                horizon_len=model_cfg.horizon_len,
                hidden_dim=model_cfg.feature_head_hidden_dim,
                dropout=model_cfg.feature_head_dropout,
            ).to(device)
        else:
            head = NullHead(model_cfg.horizon_len).to(device)

        return cls(model, head, context_len, model_cfg.horizon_len, device, trainable=True)

    @classmethod
    def from_adapter(cls, model_cfg: ModelConfig, adapter_dir: str) -> "TimesFM25Forecaster":
        """
        Loads what `save()` wrote: a LoRA adapter (`adapter_config.json`) or a fully
        fine-tuned model, plus `feature_head.pt` if the directory has one.
        """
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")

        device = get_device(model_cfg.device)
        if os.path.isfile(os.path.join(adapter_dir, ADAPTER_CONFIG_FILENAME)):
            from peft import PeftModel

            base = _load_base_model(model_cfg, device)
            context_len = min(model_cfg.context_len, base.config.context_length)
            model = PeftModel.from_pretrained(base, adapter_dir)
        else:
            from transformers import TimesFm2_5ModelForPrediction

            model = TimesFm2_5ModelForPrediction.from_pretrained(
                adapter_dir, dtype=resolve_dtype(model_cfg.torch_dtype), device_map=device)
            context_len = min(model_cfg.context_len, model.config.context_length)

        head = load_head(adapter_dir, device, model_cfg.horizon_len)
        return cls(model, head, context_len, model_cfg.horizon_len, device, trainable=False)

    # ---- FineTunableForecaster ----------------------------------------------------

    def _base_predict(self, batch: Batch) -> torch.Tensor:
        return self.model(
            past_values=batch.context.to(self.device), forecast_context_len=self.context_len,
        ).mean_predictions[:, :self.horizon_len].float()
