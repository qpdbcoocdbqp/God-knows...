"""
timesfm3.py
-----------
TimesFM 3.0 (`google/timesfm-3.0-pytorch`) from the standalone `timesfm` package
(module `timesfm3`). Upstream ships no fine-tuning entry point and no `transformers`
class, so the LoRA path here is ours:

* The LoRA layers are injected in place into the engine's `TimesFM3Torch` module, so the
  upstream inference engine (`predict_batch`) automatically uses the adapted weights.
* Training calls the same `decode` computation without its `@torch.no_grad()` wrapper
  (`decode.__wrapped__`), then applies the same quantile sorting and symmetric averaging
  as inference, so training optimizes exactly the function that is used to predict.

Notes
-----
* Responders can be negative, so `make_positive` is always False (the official
  `TimesFM3Evaluator` defaults it to True).
* `symmetric` averages forecast(x) and -forecast(-x); it doubles the compute (and the
  activation memory when training: lower the batch size, e.g. `--batch-size 32` on 8 GB).
* `use_covariates` feeds the feature columns as past-only covariates. The model handles
  at most 32 variates per forward pass, so `TimesFM3Evaluator` randomly keeps 31 of them.
  Covariates are zero-shot only; fine-tuning uses the target series (plus the optional
  feature head, as for TimesFM 2.5).
"""

import math
import os

import numpy as np
import torch

from ..config import LoraConfigOpts, ModelConfig
from ..data import Batch
from ..utils import get_device
from .feature_head import FeatureCorrectionHead, NullHead, load_head
from .fine_tunable import FineTunableForecaster

ADAPTER_CONFIG_FILENAME = "adapter_config.json"

# peft's "all-linear" only supports `transformers` models, so list the Linear layers of
# TimesFM3Torch explicitly (attention q/k/v/out in both attention blocks, feed-forward,
# the input residual block and the output head).
TIMESFM3_LINEAR_MODULES = [
    "query_proj", "key_proj", "value_proj", "out_proj", "ff0", "ff1",
    "hidden_layer", "output_layer", "residual_layer", "output_head",
]


def _build_engine(model_cfg: ModelConfig, device: str):
    try:
        import timesfm3
    except ImportError as e:
        raise ImportError(
            "TimesFM 3.0 needs the `timesfm` package: pip install 'timesfm[torch]==3.0.2'"
        ) from e

    tfm_cfg = timesfm3.ModelConfig(
        checkpoint_path=model_cfg.resolved_model_id,
        per_core_batch_size=model_cfg.per_core_batch_size,
        device=device,
    )
    # The evaluator subclass adds automatic variate chunking/subsampling, which is
    # needed once covariates are used; the plain forecaster is enough otherwise.
    engine_cls = timesfm3.TimesFM3Evaluator if model_cfg.use_covariates else timesfm3.TimesFM3Forecaster
    return engine_cls(tfm_cfg)


def _require_target_only(model_cfg: ModelConfig, action: str) -> None:
    if model_cfg.use_covariates:
        raise ValueError(f"TimesFM 3.0 covariates are zero-shot only; cannot {action} with use_covariates")


class TimesFM3Forecaster(FineTunableForecaster):
    def __init__(self, engine, model, head, model_cfg: ModelConfig, device: str, trainable: bool):
        super().__init__(model, head, model_cfg.context_len, model_cfg.horizon_len, device, trainable)
        self.engine = engine
        self.cfg = model_cfg
        self._training = False

    # ---- constructors -------------------------------------------------------------

    @classmethod
    def zero_shot(cls, model_cfg: ModelConfig) -> "TimesFM3Forecaster":
        device = get_device(model_cfg.device)
        engine = _build_engine(model_cfg, device)
        return cls(engine, engine.model, NullHead(model_cfg.horizon_len).to(device),
                   model_cfg, device, trainable=False)

    @classmethod
    def for_training(cls, model_cfg: ModelConfig, lora_cfg: LoraConfigOpts, num_features: int) -> "TimesFM3Forecaster":
        """Pretrained model + trainable LoRA adapter + optional feature head."""
        _require_target_only(model_cfg, "train")
        if not model_cfg.use_lora:
            raise ValueError("TimesFM 3.0 supports LoRA fine-tuning only (use_lora=True)")
        from peft import LoraConfig, get_peft_model

        device = get_device(model_cfg.device)
        engine = _build_engine(model_cfg, device)
        targets = TIMESFM3_LINEAR_MODULES if lora_cfg.target_modules == "all-linear" else lora_cfg.target_modules
        model = get_peft_model(engine.model, LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=targets,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
        ))
        model.print_trainable_parameters()

        if model_cfg.use_xreg:
            head = FeatureCorrectionHead(
                num_features=num_features,
                context_len=model_cfg.context_len,
                horizon_len=model_cfg.horizon_len,
                hidden_dim=model_cfg.feature_head_hidden_dim,
                dropout=model_cfg.feature_head_dropout,
            ).to(device)
        else:
            head = NullHead(model_cfg.horizon_len).to(device)
        return cls(engine, model, head, model_cfg, device, trainable=True)

    @classmethod
    def from_adapter(cls, model_cfg: ModelConfig, adapter_dir: str) -> "TimesFM3Forecaster":
        """Loads a LoRA adapter saved by `save()` (and `feature_head.pt` if present)."""
        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"adapter directory not found: {adapter_dir}")
        if not os.path.isfile(os.path.join(adapter_dir, ADAPTER_CONFIG_FILENAME)):
            raise ValueError(f"{adapter_dir} has no {ADAPTER_CONFIG_FILENAME}: "
                             "TimesFM 3.0 supports LoRA adapters only")
        _require_target_only(model_cfg, "load an adapter")
        from peft import PeftModel

        device = get_device(model_cfg.device)
        engine = _build_engine(model_cfg, device)
        model = PeftModel.from_pretrained(engine.model, adapter_dir)
        head = load_head(adapter_dir, device, model_cfg.horizon_len)
        return cls(engine, model, head, model_cfg, device, trainable=False)

    # ---- FineTunableForecaster ----------------------------------------------------

    def set_training(self, mode: bool) -> None:
        super().set_training(mode)
        self._training = mode

    def _base_predict(self, batch: Batch) -> torch.Tensor:
        if self._training:
            return self._differentiable_predict(batch.context.to(self.device).float())
        return self._engine_predict(batch)

    @torch.no_grad()
    def _engine_predict(self, batch: Batch) -> torch.Tensor:
        """Inference through the upstream engine (numpy in/out, chunked, optional covariates)."""
        kwargs = dict(
            horizon=self.horizon_len,
            return_quantiles=False,
            use_symmetric_averaging=self.cfg.symmetric,
            make_positive=False,
        )
        if self.cfg.use_covariates:
            # (B, context_len, F) -> B arrays of (F, context_len)
            kwargs["past_only_covariates"] = list(batch.context_features.numpy().transpose(0, 2, 1))

        outputs = list(self.engine.predict_batch(list(batch.context.numpy()), **kwargs))
        pred = np.stack([np.asarray(o.forecast)[:self.horizon_len] for o in outputs])
        return torch.from_numpy(pred).float().to(self.device)

    def _differentiable_predict(self, context: torch.Tensor) -> torch.Tensor:
        """Same computation as `_engine_predict` for one target series, with gradients."""
        inner = self.engine.model
        decode = type(inner).decode.__wrapped__          # `decode` minus its @torch.no_grad()
        horizon = math.ceil(self.horizon_len / inner.output_patch_len) * inner.output_patch_len

        x = context.unsqueeze(1)                          # (B, 1 variate, context_len)
        if self.cfg.symmetric:
            x = torch.cat([x, -x])
        quantiles = decode(inner, target=x, horizon=horizon)          # (B', 1, horizon, num_quantiles)
        quantiles = torch.sort(quantiles, dim=-1).values              # engine default: sort_quantiles
        if self.cfg.symmetric:
            pos, neg = quantiles.chunk(2)
            quantiles = (pos - neg.flip(-1)) / 2
        return quantiles[:, 0, :self.horizon_len, self.engine.config.median_quantile_index]
