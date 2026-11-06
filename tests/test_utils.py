import numpy as np
import pytest
import torch

from jsforecast.config import DataConfig
from jsforecast.utils import (
    build_prediction_table, ensure_dir, get_device, jane_street_r2, seed_everything,
)


def test_r2_is_one_for_perfect_predictions():
    y = np.array([0.1, -0.2, 0.3])
    assert jane_street_r2(y, y, np.ones(3)) == pytest.approx(1.0)


def test_r2_is_zero_for_zero_predictions():
    y = np.array([0.1, -0.2, 0.3])
    assert jane_street_r2(y, np.zeros(3), np.ones(3)) == pytest.approx(0.0, abs=1e-9)


def test_r2_uses_sample_weights_and_no_demeaning():
    y, p, w = np.array([1.0, 2.0]), np.array([0.0, 2.0]), np.array([3.0, 1.0])
    # num = 3*(1-0)^2 + 1*0 = 3 ; den = 3*1 + 1*4 = 7
    assert jane_street_r2(y, p, w) == pytest.approx(1 - 3 / 7)


def test_get_device_resolves_auto_and_falls_back_to_cpu():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("auto") == expected
    assert get_device("cuda") == expected
    assert get_device("cpu") == "cpu"


def test_seed_everything_makes_torch_reproducible():
    seed_everything(7)
    a = torch.rand(3)
    seed_everything(7)
    assert torch.equal(a, torch.rand(3))


def test_ensure_dir_creates_nested_directory(tmp_path):
    target = tmp_path / "a" / "b"
    assert ensure_dir(str(target)) == str(target)
    assert target.is_dir()


def test_build_prediction_table_is_horizon_step_major():
    cfg = DataConfig()
    table = build_prediction_table(
        cfg,
        symbol_id=np.array([1, 2]), date_id=np.array([10, 10]), time_id=np.array([5, 5]),
        weight=np.array([[1.0, 1.5], [2.0, 2.5]]),
        target=np.array([[0.1, 0.2], [0.3, 0.4]]),
        pred=np.array([[0.5, 0.6], [0.7, 0.8]]),
    )
    assert table.column_names == [
        "symbol_id", "date_id", "time_id", "weight", "responder_6_true", "responder_6_pred"]
    assert table.column("symbol_id").to_pylist() == [1, 2, 1, 2]
    assert table.column("responder_6_pred").to_pylist() == [0.5, 0.7, 0.6, 0.8]
    assert table.column("weight").to_pylist() == [1.0, 2.0, 1.5, 2.5]
