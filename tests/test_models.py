import pytest
import torch

from fakes import LastValueForecaster
from jsforecast.config import MODEL_IDS, DataConfig, LoraConfigOpts, ModelConfig
from jsforecast.models import FeatureCorrectionHead, NullHead, build_forecaster, load_head
from jsforecast.models.feature_head import FEATURE_HEAD_FILENAME
from jsforecast.models.losses import JaneStreetWeightedMSELoss


def test_model_config_resolves_default_model_id_per_type():
    assert ModelConfig(model_type="timesfm3").resolved_model_id == MODEL_IDS["timesfm3"]
    assert ModelConfig(model_type="timesfm2_5").resolved_model_id == MODEL_IDS["timesfm2_5"]
    assert ModelConfig(model_id="my/repo").resolved_model_id == "my/repo"


def test_model_config_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="Unknown model_type"):
        ModelConfig(model_type="timesfm9")


def test_null_head_returns_zero_delta_of_the_right_shape():
    delta = NullHead(horizon_len=3)(torch.randn(4, 5, 7))
    assert delta.shape == (4, 3) and torch.count_nonzero(delta) == 0


def test_null_head_leaves_the_base_prediction_unchanged():
    base = torch.randn(4, 3)
    assert torch.equal(base + NullHead(3)(torch.randn(4, 5, 7)), base)


def test_feature_head_maps_features_to_horizon():
    head = FeatureCorrectionHead(num_features=7, context_len=5, horizon_len=2)
    assert head(torch.randn(4, 5, 7)).shape == (4, 2)


def test_feature_head_save_load_roundtrip(tmp_path):
    head = FeatureCorrectionHead(num_features=7, context_len=5, horizon_len=2, hidden_dim=16).eval()
    head.save(str(tmp_path))
    loaded = load_head(str(tmp_path), "cpu", horizon_len=2)
    x = torch.randn(3, 5, 7)
    assert isinstance(loaded, FeatureCorrectionHead)
    assert not loaded.training                      # dropout must be off right after loading
    assert torch.equal(head(x), loaded(x))


def test_load_head_without_file_gives_null_head(tmp_path):
    assert not (tmp_path / FEATURE_HEAD_FILENAME).exists()
    assert isinstance(load_head(str(tmp_path), "cpu", horizon_len=1), NullHead)


def test_null_head_save_writes_nothing(tmp_path):
    NullHead(1).save(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_weighted_mse_loss_value():
    loss = JaneStreetWeightedMSELoss()(
        torch.tensor([[1.0], [2.0]]), torch.tensor([[0.0], [0.0]]), torch.tensor([[1.0], [0.5]]))
    assert loss.item() == pytest.approx((1.0 * 1 + 0.5 * 4) / 2)


def test_inference_only_forecaster_refuses_training_hooks(tmp_path):
    f = LastValueForecaster(context_len=5)
    assert f.is_trainable is False
    with pytest.raises(TypeError, match="inference-only"):
        f.trainable_parameters()
    with pytest.raises(TypeError, match="inference-only"):
        f.save(str(tmp_path))


def test_factory_timesfm3_training_needs_lora_and_data_config():
    with pytest.raises(ValueError, match="lora_cfg and data_cfg"):
        build_forecaster(ModelConfig(model_type="timesfm3"), trainable=True)


def test_timesfm3_covariates_are_zero_shot_only(tmp_path):
    cfg = ModelConfig(model_type="timesfm3", use_covariates=True)
    with pytest.raises(ValueError, match="zero-shot only"):
        build_forecaster(cfg, trainable=True, lora_cfg=LoraConfigOpts(), data_cfg=DataConfig())
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="zero-shot only"):
        build_forecaster(cfg, adapter_dir=str(tmp_path))


def test_timesfm3_fine_tuning_is_lora_only():
    cfg = ModelConfig(model_type="timesfm3", use_lora=False)
    with pytest.raises(ValueError, match="LoRA fine-tuning only"):
        build_forecaster(cfg, trainable=True, lora_cfg=LoraConfigOpts(), data_cfg=DataConfig())


def test_timesfm3_adapter_dir_must_exist_and_hold_a_lora_adapter(tmp_path):
    cfg = ModelConfig(model_type="timesfm3")
    with pytest.raises(FileNotFoundError, match="adapter directory not found"):
        build_forecaster(cfg, adapter_dir=str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="LoRA adapters only"):
        build_forecaster(cfg, adapter_dir=str(tmp_path))


def test_factory_rejects_resuming_training_from_adapter():
    with pytest.raises(ValueError, match="Resuming"):
        build_forecaster(ModelConfig(), trainable=True, adapter_dir="somewhere")


def test_factory_trainable_needs_lora_and_data_config():
    with pytest.raises(ValueError, match="lora_cfg and data_cfg"):
        build_forecaster(ModelConfig(), trainable=True)
    with pytest.raises(ValueError, match="lora_cfg and data_cfg"):
        build_forecaster(ModelConfig(), trainable=True, lora_cfg=LoraConfigOpts(), data_cfg=None)


def test_factory_missing_adapter_dir_is_an_error_not_a_silent_skip(tmp_path):
    with pytest.raises(FileNotFoundError, match="adapter directory not found"):
        build_forecaster(ModelConfig(), adapter_dir=str(tmp_path / "missing"))
