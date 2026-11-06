import numpy as np
import pyarrow.parquet as pq
import pytest

from conftest import CONTEXT_LEN, N_SYMBOLS, ROWS_PER_SYMBOL
from fakes import LastValueForecaster
from jsforecast.config import BacktestConfig, DataConfig
from jsforecast.data import JaneStreetDataModule
from jsforecast.evaluate import Backtester, Evaluator, Predictor
from jsforecast.utils import build_prediction_table, jane_street_r2

WINDOWS = N_SYMBOLS * (ROWS_PER_SYMBOL - (CONTEXT_LEN + 1) + 1)


def _table(dates, weight, true, pred):
    n = len(dates)
    return build_prediction_table(
        DataConfig(),
        symbol_id=np.zeros(n), date_id=np.array(dates), time_id=np.arange(n),
        weight=np.array(weight).reshape(n, 1), target=np.array(true).reshape(n, 1),
        pred=np.array(pred).reshape(n, 1),
    )


def test_predictor_returns_one_row_per_window_with_the_forecast(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    table = Predictor(data_cfg).run(LastValueForecaster(CONTEXT_LEN), dm.loader("val", 16))
    assert table.num_rows == WINDOWS
    # LastValueForecaster predicts the previous target, which is 1 less than the true next one
    diff = np.array(table["responder_6_true"].to_pylist()) - np.array(table["responder_6_pred"].to_pylist())
    assert np.allclose(diff, 1.0)


def test_predictor_max_batches_limits_the_rows(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    table = Predictor(data_cfg).run(LastValueForecaster(CONTEXT_LEN), dm.loader("val", 10), max_batches=2)
    assert table.num_rows == 20


def test_predictor_puts_the_forecaster_in_eval_mode(data_cfg):
    from fakes import LinearForecaster

    forecaster = LinearForecaster(CONTEXT_LEN)
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    Predictor(data_cfg).run(forecaster, dm.loader("val", 16), max_batches=1)
    assert forecaster.modes == [False]


def test_predictor_on_an_empty_loader_returns_an_empty_table(data_cfg):
    table = Predictor(data_cfg).run(LastValueForecaster(CONTEXT_LEN), [])
    assert table.num_rows == 0 and "responder_6_pred" in table.column_names


def test_evaluator_scores_match_hand_computation():
    # day 1: pos=+1 -> pnl = 2*1*0.5 = 1.0 ; day 2: pos=-1, true=-1 -> pnl = 1*(-1)*(-1) = 1.0
    table = _table(dates=[1, 2], weight=[2.0, 1.0], true=[0.5, -1.0], pred=[0.3, -0.2])
    m = Evaluator(DataConfig(), annualization_days=4).score(table)
    assert m.n_days == 2 and m.n_rows == 2
    assert m.total_pnl == pytest.approx(2.0)
    assert m.sharpe == pytest.approx(2.0 / np.sqrt(2.0) * np.sqrt(4 / 2))
    assert m.r2 == pytest.approx(jane_street_r2(
        np.array([0.5, -1.0]), np.array([0.3, -0.2]), np.array([2.0, 1.0])))


def test_daily_pnl_is_sorted_and_cumulative():
    table = _table(dates=[3, 1, 1, 2], weight=[1.0] * 4, true=[1.0, 1.0, 2.0, -1.0], pred=[1.0, 1.0, 1.0, 1.0])
    daily = Evaluator(DataConfig()).daily_pnl(table)
    assert daily["date_id"].to_pylist() == [1, 2, 3]
    assert daily["pnl"].to_pylist() == [3.0, -1.0, 1.0]
    assert daily["equity"].to_pylist() == [3.0, 2.0, 3.0]


def test_zero_prediction_takes_no_position():
    table = _table(dates=[1], weight=[1.0], true=[5.0], pred=[0.0])
    assert Evaluator(DataConfig()).score(table).total_pnl == 0.0


def test_evaluator_rejects_an_empty_table(data_cfg):
    empty = Predictor(data_cfg).run(LastValueForecaster(CONTEXT_LEN), [])
    with pytest.raises(ValueError, match="No windows to score"):
        Evaluator(data_cfg).score(empty)


def test_backtester_prints_saves_and_returns_metrics(data_cfg, tmp_path, capsys):
    bt_cfg = BacktestConfig(output_dir=str(tmp_path / "out"))
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    metrics = Backtester(data_cfg, bt_cfg).run(
        LastValueForecaster(CONTEXT_LEN), dm.loader("test", 32), label="naive")

    assert metrics.n_rows == WINDOWS
    assert "[naive] Backtest results" in capsys.readouterr().out
    saved = pq.read_table(tmp_path / "out" / "predictions_naive.parquet")
    assert saved.num_rows == WINDOWS
    assert pq.read_table(tmp_path / "out" / "daily_pnl_naive.parquet").num_rows == metrics.n_days
