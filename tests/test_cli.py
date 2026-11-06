import pytest

from conftest import CONTEXT_LEN
from fakes import LastValueForecaster
from jsforecast import cli


def _config(*argv):
    args = cli.build_parser().parse_args(argv)
    return args, cli.build_config(args)


def test_defaults_come_from_the_config_dataclasses():
    _, cfg = _config("train")
    assert cfg.train.batch_size == 64 and cfg.train.eval_batch_size == 128
    assert cfg.model.model_type == "timesfm2_5"


def test_train_overrides_land_in_the_right_sections():
    _, cfg = _config("train", "--batch-size", "16", "--eval-batch-size", "32", "--epochs", "5",
                     "--output-dir", "ckpt", "--no-xreg", "--no-lora", "--symbol-sample-rate", "0.03",
                     "--train-partitions", "0", "1")
    assert (cfg.train.batch_size, cfg.train.eval_batch_size, cfg.train.epochs) == (16, 32, 5)
    assert cfg.train.output_dir == "ckpt"
    assert cfg.model.use_xreg is False and cfg.model.use_lora is False
    assert cfg.data.symbol_sample_rate == 0.03 and cfg.data.train_partitions == [0, 1]
    assert cfg.backtest.batch_size == 128          # untouched


def test_backtest_batch_size_and_output_dir_do_not_leak_into_training_config():
    _, cfg = _config("backtest", "--batch-size", "32", "--output-dir", "res", "--max-batches", "120")
    assert cfg.backtest.batch_size == 32 and cfg.backtest.output_dir == "res"
    assert cfg.backtest.max_batches == 120
    assert cfg.train.batch_size == 64 and cfg.train.output_dir != "res"


def test_timesfm3_options():
    _, cfg = _config("backtest", "--model", "timesfm3", "--covariates", "--no-symmetric",
                     "--test-partitions", "8")
    assert cfg.model.model_type == "timesfm3" and cfg.model.use_covariates is True
    assert cfg.model.symmetric is False and cfg.data.test_partitions == [8]
    assert cfg.model.resolved_model_id == "google/timesfm-3.0-pytorch"


def test_unknown_model_and_missing_command_are_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["backtest", "--model", "timesfm9"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def _run_backtest(monkeypatch, tmp_path, data_cfg, argv, adapter_dir):
    built = []

    def fake_build(model_cfg, *, adapter_dir=None, **kwargs):
        built.append(adapter_dir)
        return LastValueForecaster(CONTEXT_LEN)

    monkeypatch.setattr(cli, "build_forecaster", fake_build)
    args, cfg = _config("backtest", "--output-dir", str(tmp_path), *argv)
    cfg.data = data_cfg
    cli.run_backtest(cfg, "test", adapter_dir=adapter_dir)
    return built, sorted(p.name for p in tmp_path.glob("predictions_*.parquet"))


def test_backtest_without_adapter_scores_only_the_zero_shot_model(monkeypatch, tmp_path, data_cfg):
    built, files = _run_backtest(monkeypatch, tmp_path, data_cfg, ["--no-zero-shot"], adapter_dir=None)
    assert built == [None]
    assert files == ["predictions_timesfm2_5_zero_shot.parquet"]


def test_backtest_with_adapter_scores_finetuned_and_zero_shot(monkeypatch, tmp_path, data_cfg):
    built, files = _run_backtest(monkeypatch, tmp_path, data_cfg, [], adapter_dir="ckpt")
    assert built == ["ckpt", None]
    assert files == ["predictions_timesfm2_5_finetuned.parquet", "predictions_timesfm2_5_zero_shot.parquet"]


def test_backtest_with_adapter_and_no_zero_shot_scores_only_the_adapter(monkeypatch, tmp_path, data_cfg):
    built, files = _run_backtest(monkeypatch, tmp_path, data_cfg, ["--no-zero-shot"], adapter_dir="ckpt")
    assert built == ["ckpt"]
    assert files == ["predictions_timesfm2_5_finetuned.parquet"]


def test_covariate_runs_get_an_xreg_label(monkeypatch, tmp_path, data_cfg):
    _, files = _run_backtest(monkeypatch, tmp_path, data_cfg, ["--model", "timesfm3", "--covariates"], None)
    assert files == ["predictions_timesfm3_zero_shot_xreg.parquet"]
