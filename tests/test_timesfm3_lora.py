"""
TimesFM 3.0 LoRA fine-tuning checks on the real pretrained model (random inputs, no Jane
Street data needed). Needs the `timesfm` package and the cached weights, so it is
deselected by default:

    pytest -m integration -s tests/test_timesfm3_lora.py
"""

import numpy as np
import pytest
import torch

from jsforecast.config import DataConfig, LoraConfigOpts, ModelConfig
from jsforecast.data import Batch
from jsforecast.models import build_forecaster

pytestmark = pytest.mark.integration

CONTEXT_LEN = 512
N_FEATURES = 4


@pytest.fixture(scope="module")
def batch() -> Batch:
    g = torch.Generator().manual_seed(0)
    b = 6
    return Batch(
        context=torch.randn(b, CONTEXT_LEN, generator=g) * 0.5,
        target=torch.randn(b, 1, generator=g) * 0.5,
        weight=torch.ones(b, 1),
        context_features=torch.randn(b, CONTEXT_LEN, N_FEATURES, generator=g),
        target_features=torch.randn(b, 1, N_FEATURES, generator=g),
        symbol_id=torch.zeros(b, dtype=torch.long),
        date_id=torch.zeros(b, dtype=torch.long),
        time_id=torch.zeros(b, dtype=torch.long),
    )


def _data_cfg() -> DataConfig:
    return DataConfig(feature_cols=[f"feature_{i:02d}" for i in range(N_FEATURES)])


def _trainable(use_xreg: bool = False, symmetric: bool = True):
    cfg = ModelConfig(model_type="timesfm3", use_xreg=use_xreg, symmetric=symmetric,
                      feature_head_hidden_dim=8)
    return cfg, build_forecaster(cfg, lora_cfg=LoraConfigOpts(), data_cfg=_data_cfg(), trainable=True)


@pytest.mark.parametrize("symmetric", [True, False])
def test_training_path_matches_the_inference_engine(batch, symmetric):
    cfg, f = _trainable(symmetric=symmetric)        # LoRA B starts at zero: same function as zero-shot
    with torch.no_grad():
        train_path = f._differentiable_predict(batch.context.to(f.device)).cpu()
        engine_path = f._engine_predict(batch).cpu()
    assert train_path.shape == engine_path.shape == (len(batch), 1)
    assert torch.allclose(train_path, engine_path, atol=1e-4), (train_path - engine_path).abs().max()


def test_only_the_lora_adapter_and_head_are_trainable_and_get_gradients(batch):
    _, f = _trainable(use_xreg=True)
    params = f.trainable_parameters()
    assert 0 < sum(p.numel() for p in params) < 0.02 * sum(p.numel() for p in f.engine.model.parameters())
    named = {n for n, p in f.model.named_parameters() if p.requires_grad}
    assert named and all("lora_" in n for n in named)

    f.set_training(True)
    loss = ((f.predict(batch) - batch.target.to(f.device)) ** 2).mean()
    loss.backward()
    assert all(p.grad is not None for p in params)
    grads = {n: p.grad for n, p in f.model.named_parameters() if p.requires_grad}
    assert any(g.abs().sum() > 0 for n, g in grads.items() if "lora_B" in n)
    assert all(p.grad is None for p in f.engine.model.parameters() if not p.requires_grad)


def test_a_training_step_changes_the_prediction(batch):
    _, f = _trainable()
    f.set_training(False)
    with torch.no_grad():
        before = f.predict(batch).cpu()
    f.set_training(True)
    opt = torch.optim.AdamW(f.trainable_parameters(), lr=1e-2)
    for _ in range(3):
        loss = ((f.predict(batch) - batch.target.to(f.device)) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    f.set_training(False)
    with torch.no_grad():
        after = f.predict(batch).cpu()
    assert not torch.allclose(before, after, atol=1e-6)


def test_save_and_load_reproduce_the_predictions(batch, tmp_path):
    cfg, f = _trainable(use_xreg=True)
    f.set_training(True)
    opt = torch.optim.AdamW(f.trainable_parameters(), lr=1e-2)
    for _ in range(2):
        loss = ((f.predict(batch) - batch.target.to(f.device)) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    f.set_training(False)
    with torch.no_grad():
        expected = f.predict(batch).cpu()

    f.save(str(tmp_path))
    assert (tmp_path / "adapter_config.json").is_file() and (tmp_path / "feature_head.pt").is_file()

    loaded = build_forecaster(cfg, adapter_dir=str(tmp_path))
    assert loaded.is_trainable is False
    with torch.no_grad():
        got = loaded.predict(batch).cpu()
    assert np.allclose(got.numpy(), expected.numpy(), atol=1e-5), (got - expected).abs().max()
