import pytest
import torch

from conftest import CONTEXT_LEN
from fakes import LastValueForecaster, LinearForecaster
from jsforecast.config import TrainConfig
from jsforecast.data import JaneStreetDataModule
from jsforecast.trainer import EarlyStopping, Trainer


def test_early_stopping_tracks_best_and_stops_after_patience():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(0.5) is True and stopper.best == 0.5
    assert stopper.update(0.4) is False and not stopper.should_stop
    assert stopper.update(0.4) is False and stopper.should_stop


def test_early_stopping_resets_the_counter_on_improvement():
    stopper = EarlyStopping(patience=2)
    stopper.update(0.5)
    stopper.update(0.1)
    stopper.update(0.9)
    assert stopper.bad_epochs == 0 and not stopper.should_stop


def test_trainer_rejects_an_inference_only_forecaster(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    with pytest.raises(ValueError, match="not trainable"):
        Trainer(LastValueForecaster(CONTEXT_LEN), dm, TrainConfig(), data_cfg)


def test_trainer_fits_validates_and_saves_the_best_checkpoint(data_cfg, tmp_path):
    torch.manual_seed(0)
    forecaster = LinearForecaster(CONTEXT_LEN)
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    cfg = TrainConfig(epochs=2, batch_size=8, eval_batch_size=16, lr=1e-3, output_dir=str(tmp_path / "ckpt"))

    result = Trainer(forecaster, dm, cfg, data_cfg).fit()

    assert result.epochs_run == 2
    assert result.best_val_r2 > float("-inf")
    assert (tmp_path / "ckpt" / "linear.pt").is_file()
    # train mode before each epoch, eval mode inside every validation pass
    assert forecaster.modes == [True, False, True, False]


def test_trainer_updates_the_weights(data_cfg, tmp_path):
    torch.manual_seed(0)
    forecaster = LinearForecaster(CONTEXT_LEN)
    before = [p.detach().clone() for p in forecaster.trainable_parameters()]
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    cfg = TrainConfig(epochs=1, batch_size=8, lr=1e-2, output_dir=str(tmp_path))
    Trainer(forecaster, dm, cfg, data_cfg).fit()
    after = forecaster.trainable_parameters()
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_trainer_respects_the_batch_caps(data_cfg, tmp_path):
    forecaster = LinearForecaster(CONTEXT_LEN)
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    cfg = TrainConfig(epochs=1, batch_size=8, max_train_batches=2, max_val_batches=1, output_dir=str(tmp_path))
    steps = []
    original = forecaster.predict

    def counting_predict(batch):
        steps.append(forecaster.net.training)
        return original(batch)

    forecaster.predict = counting_predict
    Trainer(forecaster, dm, cfg, data_cfg).fit()
    assert steps == [True, True, False]      # 2 train batches + 1 validation batch
