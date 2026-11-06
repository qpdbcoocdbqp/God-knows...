import pytest
import torch

from conftest import CONTEXT_LEN, N_DATES, N_FEATURES, N_SYMBOLS, N_TIMES, ROWS_PER_SYMBOL, _target_value
from jsforecast.data import Batch, JaneStreetDataModule, JaneStreetStreamDataset, collate_batch


def _samples(dataset):
    return list(dataset)


def test_window_count_is_rows_minus_window_plus_one(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    windows_per_symbol = ROWS_PER_SYMBOL - (CONTEXT_LEN + 1) + 1
    assert len(_samples(dm.dataset("val"))) == N_SYMBOLS * windows_per_symbol


def test_eval_stride_defaults_to_horizon(data_cfg):
    horizon = 2
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=horizon)
    window = CONTEXT_LEN + horizon
    per_symbol = (ROWS_PER_SYMBOL - window) // horizon + 1
    assert len(_samples(dm.dataset("val"))) == N_SYMBOLS * per_symbol


def test_window_holds_past_targets_as_context_and_next_as_target(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    first = [s for s in _samples(dm.dataset("val")) if s["symbol_id"] == 0][0]
    assert first["context"].tolist() == [_target_value(0, 1, r) for r in range(CONTEXT_LEN)]
    assert first["target"].tolist() == [_target_value(0, 1, CONTEXT_LEN)]
    assert first["context_features"].shape == (CONTEXT_LEN, N_FEATURES)
    assert first["target_features"].shape == (1, N_FEATURES)
    # the first predicted row sits on date index CONTEXT_LEN // N_TIMES of partition 1
    assert first["date_id"] == 1 * N_DATES + CONTEXT_LEN // N_TIMES


def test_windows_never_mix_symbols(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    for s in _samples(dm.dataset("val")):
        values = torch.cat([s["context"], s["target"]])
        assert (values // 1000 == s["symbol_id"]).all()


def test_collate_batch_shapes(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=2)
    batch = next(iter(dm.loader("val", batch_size=4)))
    assert isinstance(batch, Batch) and len(batch) == 4
    assert batch.context.shape == (4, CONTEXT_LEN)
    assert batch.target.shape == batch.weight.shape == (4, 2)
    assert batch.context_features.shape == (4, CONTEXT_LEN, N_FEATURES)
    assert batch.target_features.shape == (4, 2, N_FEATURES)
    assert batch.symbol_id.shape == batch.date_id.shape == batch.time_id.shape == (4,)


def test_collate_batch_roundtrip():
    sample = {
        "context": torch.zeros(3), "target": torch.zeros(1), "weight": torch.ones(1),
        "context_features": torch.zeros(3, 2), "target_features": torch.zeros(1, 2),
        "symbol_id": 4, "date_id": 5, "time_id": 6,
    }
    batch = collate_batch([sample, sample])
    assert batch.symbol_id.tolist() == [4, 4] and batch.date_id.tolist() == [5, 5]


def test_train_shuffle_buffer_changes_order_but_not_content(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    shuffled = _samples(dm.dataset("train"))
    data_cfg.shuffle_buffer_size = 0
    ordered = _samples(JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1).dataset("train"))

    def key(s):
        return (s["symbol_id"], s["date_id"], s["time_id"])

    assert [key(s) for s in shuffled] != [key(s) for s in ordered]
    assert sorted(key(s) for s in shuffled) == sorted(key(s) for s in ordered)


def test_symbol_sampling_keeps_a_deterministic_subset(data_cfg):
    data_cfg.symbol_sample_rate = 0.34            # floor(3 * 0.34) = 1 symbol
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    first = {s["symbol_id"] for s in _samples(dm.dataset("val"))}
    second = {s["symbol_id"] for s in _samples(dm.dataset("val"))}
    assert len(first) == 1 and first == second


def test_symbol_sampling_always_keeps_at_least_one_symbol(data_cfg):
    data_cfg.symbol_sample_rate = 0.001
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    assert len({s["symbol_id"] for s in _samples(dm.dataset("val"))}) == 1


def test_max_rows_per_symbol_caps_the_stream(data_cfg):
    data_cfg.max_rows_per_symbol = 10
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, horizon_len=1)
    assert len(_samples(dm.dataset("val"))) == N_SYMBOLS * (10 - (CONTEXT_LEN + 1) + 1)


def test_too_short_split_yields_no_windows(data_cfg):
    dm = JaneStreetDataModule(data_cfg, context_len=ROWS_PER_SYMBOL, horizon_len=1)
    assert _samples(dm.dataset("val")) == []


def test_dataset_without_partitions_is_empty(data_cfg):
    ds = JaneStreetStreamDataset(data_cfg, [], CONTEXT_LEN, 1)
    assert _samples(ds) == []


def test_unknown_split_is_rejected(data_cfg):
    with pytest.raises(ValueError, match="Unknown split"):
        JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1).dataset("holdout")


def test_only_train_loader_drops_last_batch(data_cfg):
    dm = JaneStreetDataModule(data_cfg, CONTEXT_LEN, 1)
    assert dm.loader("train", 4).drop_last is True
    assert dm.loader("val", 4).drop_last is False
