"""
model.py
--------
Loads TimesFM 2.5 via HuggingFace `transformers` and applies LoRA via
`peft`, following Google's official fine-tuning reference:

    google-research/timesfm: timesfm-forecasting/examples/finetuning/finetune_lora.py
    (uses transformers.TimesFm2_5ModelForPrediction + peft.LoraConfig)

This REPLACES the earlier approach of poking at the standalone `timesfm`
package's internals -- the `transformers` checkpoint natively supports a
standard PyTorch training loop:

    outputs = model(past_values=context, future_values=target,
                     forecast_context_len=context_len)
    outputs.loss               # native (unweighted) MSE + quantile loss
    outputs.mean_predictions   # (batch, horizon) point forecast

Verified against the installed `transformers` source
(`transformers/models/timesfm2_5/modeling_timesfm2_5.py`):
  - `forward(past_values, future_values=None, forecast_context_len=None, ...)`
  - `past_values` is a `Sequence[torch.Tensor]`; a stacked 2D tensor works
    fine since it's iterated row-by-row internally.
  - `outputs.loss` is computed in *normalized* space and is *unweighted*,
    so for Jane Street we ignore it and instead compute our own
    trading-weighted loss directly on `outputs.mean_predictions` (see
    `losses.JaneStreetWeightedMSELoss` + `train.py`).
"""

import torch

from config import DataConfig, LoraConfigOpts, ModelConfig
from feature_head import FeatureCorrectionHead, load_feature_head, save_feature_head  # noqa: F401


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[name]


def load_base_model(model_cfg: ModelConfig, device: str):
    """Loads the plain (no adapter) TimesFM 2.5 transformers checkpoint."""
    from transformers import TimesFm2_5ModelForPrediction

    model = TimesFm2_5ModelForPrediction.from_pretrained(
        model_cfg.model_id,
        torch_dtype=_resolve_dtype(model_cfg.torch_dtype),
        device_map=device,
    )
    return model


def apply_lora(model, lora_cfg: LoraConfigOpts):
    """Wraps a loaded model with a LoRA adapter (trainable, base frozen)."""
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.dropout,
        bias=lora_cfg.bias,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_adapter(model_cfg: ModelConfig, adapter_dir: str, device: str):
    """Loads a base model + a previously trained LoRA adapter (for eval/backtest)."""
    from peft import PeftModel

    base_model = load_base_model(model_cfg, device)
    return PeftModel.from_pretrained(base_model, adapter_dir)


def effective_context_len(model, requested_context_len: int) -> int:
    """TimesFM caps context at model.config.context_length."""
    return min(requested_context_len, model.config.context_length)


def build_feature_head(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    context_len: int,
    horizon_len: int,
) -> FeatureCorrectionHead:
    """Build a FeatureCorrectionHead from config objects."""
    return FeatureCorrectionHead(
        num_features=len(data_cfg.feature_cols),
        context_len=context_len,
        horizon_len=horizon_len,
        hidden_dim=model_cfg.feature_head_hidden_dim,
        dropout=model_cfg.feature_head_dropout,
    )
