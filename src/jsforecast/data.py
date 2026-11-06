"""
data.py
-------
Lazy, batched Jane Street data loading via `pyarrow.dataset` -- no pandas, no
full-split materialization.

  1. The parquet directory is opened as a `pyarrow.dataset.Dataset` (only the
     schema/file list is read).
  2. It is scanned with a column projection and a `partition_id` filter, so only the
     requested partitions are opened.
  3. `RecordBatch`es are streamed one at a time (`scanner.to_batches()`).
  4. A bounded per-symbol sliding window (`deque(maxlen=context_len+horizon_len)`)
     is updated as rows stream past; once full, a sample is emitted. Memory is
     O(num_symbols * window_len), not O(rows in split).

Assumption: rows within a partition are ordered by (date_id, time_id, symbol_id),
which holds for the official Jane Street parquet files.
"""

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, IterableDataset

from .config import DataConfig

SPLITS = ("train", "val", "test")


@dataclass
class Batch:
    """One collated mini-batch. All tensors are on the CPU."""

    context: torch.Tensor           # (B, context_len)               past target values
    target: torch.Tensor            # (B, horizon_len)               values to predict
    weight: torch.Tensor            # (B, horizon_len)               trading weights
    context_features: torch.Tensor  # (B, context_len, num_features)
    target_features: torch.Tensor   # (B, horizon_len, num_features)
    symbol_id: torch.Tensor         # (B,)
    date_id: torch.Tensor           # (B,) date of the first target step
    time_id: torch.Tensor           # (B,) time of the first target step

    def __len__(self) -> int:
        return self.context.shape[0]


def collate_batch(samples: List[dict]) -> Batch:
    def stack(key: str) -> torch.Tensor:
        return torch.stack([s[key] for s in samples])

    return Batch(
        context=stack("context"),
        target=stack("target"),
        weight=stack("weight"),
        context_features=stack("context_features"),
        target_features=stack("target_features"),
        symbol_id=torch.tensor([s["symbol_id"] for s in samples]),
        date_id=torch.tensor([s["date_id"] for s in samples]),
        time_id=torch.tensor([s["time_id"] for s in samples]),
    )


def _make_scanner(data_cfg: DataConfig, partitions: List[int], sampled_symbols: Optional[List[int]] = None):
    dataset = ds.dataset(data_cfg.data_dir, format="parquet", partitioning="hive")
    needed = [data_cfg.date_col, data_cfg.time_col, data_cfg.symbol_col,
              data_cfg.weight_col, data_cfg.target_col] + list(data_cfg.feature_cols)
    filt = ds.field("partition_id").isin(partitions)
    if sampled_symbols is not None:
        filt = filt & ds.field(data_cfg.symbol_col).isin(sampled_symbols)
    return dataset.scanner(columns=needed, filter=filt, batch_size=data_cfg.arrow_batch_size)


class JaneStreetStreamDataset(IterableDataset):
    """
    Streams Jane Street rows from parquet and yields per-symbol windows as dicts of
    torch tensors (see `collate_batch`).

    `shuffle_buffer_size > 0` enables a streaming shuffle buffer (reservoir-style local
    randomization) for training; 0 gives deterministic, in-stream-order iteration.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        partitions: List[int],
        context_len: int,
        horizon_len: int,
        stride: int = 1,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
        max_rows_per_symbol: Optional[int] = None,
    ):
        self.data_cfg = data_cfg
        self.partitions = partitions
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.window_len = context_len + horizon_len
        self.stride = max(1, stride)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.max_rows_per_symbol = max_rows_per_symbol

    def _sample_symbols(self, rng: random.Random) -> Optional[List[int]]:
        """Scans the partitions once for unique symbol_ids and keeps a seeded subsample.

        Returns None when every symbol is kept (`symbol_sample_rate >= 1.0`).
        """
        if self.data_cfg.symbol_sample_rate >= 1.0:
            return None
        dataset = ds.dataset(self.data_cfg.data_dir, format="parquet", partitioning="hive")
        scanner = dataset.scanner(
            columns=[self.data_cfg.symbol_col],
            filter=ds.field("partition_id").isin(self.partitions),
            batch_size=self.data_cfg.arrow_batch_size,
        )
        symbols: set = set()
        for batch in scanner.to_batches():
            symbols.update(batch.column(self.data_cfg.symbol_col).to_pylist())
        symbols_list = sorted(symbols)
        rng.shuffle(symbols_list)
        k = max(math.floor(len(symbols_list) * self.data_cfg.symbol_sample_rate), 1)
        return sorted(symbols_list[:k])

    def _partitions_for_worker(self) -> List[int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.partitions
        # split partition files across DataLoader workers so they don't re-stream the same data
        return self.partitions[worker_info.id::worker_info.num_workers]

    def _emit(self, symbol_id: int, buf: deque) -> dict:
        items = list(buf)
        ctx_items, tgt_items = items[:self.context_len], items[self.context_len:]
        return {
            "context": torch.from_numpy(np.array([it["target"] for it in ctx_items], dtype=np.float32)),
            "target": torch.from_numpy(np.array([it["target"] for it in tgt_items], dtype=np.float32)),
            "weight": torch.from_numpy(np.array([it["weight"] for it in tgt_items], dtype=np.float32)),
            "context_features": torch.from_numpy(np.stack([it["features"] for it in ctx_items], axis=0)),
            "target_features": torch.from_numpy(np.stack([it["features"] for it in tgt_items], axis=0)),
            "symbol_id": symbol_id,
            "date_id": tgt_items[0]["date"],
            "time_id": tgt_items[0]["time"],
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (worker_info.id if worker_info is not None else 0))

        sampled_symbols = self._sample_symbols(rng)
        partitions = self._partitions_for_worker()
        if not partitions:
            return
        scanner = _make_scanner(self.data_cfg, partitions, sampled_symbols=sampled_symbols)

        buffers: Dict[int, deque] = {}
        since_last_emit: Dict[int, int] = {}
        row_count: Dict[int, int] = {}
        shuffle_pool: List[dict] = []

        def push(sample: dict) -> Optional[dict]:
            """Returns the sample to yield now, or None while the shuffle buffer is filling."""
            if self.shuffle_buffer_size <= 0:
                return sample
            shuffle_pool.append(sample)
            if len(shuffle_pool) < self.shuffle_buffer_size:
                return None
            idx = rng.randrange(len(shuffle_pool))
            shuffle_pool[idx], shuffle_pool[-1] = shuffle_pool[-1], shuffle_pool[idx]
            return shuffle_pool.pop()

        cfg = self.data_cfg
        for batch in scanner.to_batches():
            date_arr = batch.column(cfg.date_col).to_numpy(zero_copy_only=False)
            time_arr = batch.column(cfg.time_col).to_numpy(zero_copy_only=False)
            symbol_arr = batch.column(cfg.symbol_col).to_numpy(zero_copy_only=False)
            weight_arr = batch.column(cfg.weight_col).to_numpy(zero_copy_only=False).astype(np.float32)
            target_arr = batch.column(cfg.target_col).to_numpy(zero_copy_only=False).astype(np.float32)

            # (batch_rows, num_features); missing feature values become 0
            feature_matrix = np.stack([
                batch.column(col).to_numpy(zero_copy_only=False).astype(np.float32)
                for col in cfg.feature_cols
            ], axis=1)
            np.nan_to_num(feature_matrix, copy=False, nan=0.0)

            for i in np.nonzero(~np.isnan(target_arr))[0]:
                sid = int(symbol_arr[i])

                if self.max_rows_per_symbol is not None:
                    row_count[sid] = row_count.get(sid, 0) + 1
                    if row_count[sid] > self.max_rows_per_symbol:
                        continue

                buf = buffers.setdefault(sid, deque(maxlen=self.window_len))
                buf.append({
                    "target": float(target_arr[i]),
                    "weight": float(weight_arr[i]),
                    "date": int(date_arr[i]),
                    "time": int(time_arr[i]),
                    "features": feature_matrix[i],
                })
                since_last_emit[sid] = since_last_emit.get(sid, self.stride) + 1

                if len(buf) == self.window_len and since_last_emit[sid] >= self.stride:
                    since_last_emit[sid] = 0
                    ready = push(self._emit(sid, buf))
                    if ready is not None:
                        yield ready

        # flush remaining shuffle-buffered samples in random order at the end of the pass
        rng.shuffle(shuffle_pool)
        yield from shuffle_pool


class JaneStreetDataModule:
    """Builds the train / val / test datasets and loaders from one `DataConfig`."""

    def __init__(self, data_cfg: DataConfig, context_len: int, horizon_len: int):
        self.data_cfg = data_cfg
        self.context_len = context_len
        self.horizon_len = horizon_len

    def dataset(self, split: str) -> JaneStreetStreamDataset:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
        cfg = self.data_cfg
        if split == "train":
            partitions, stride, shuffle, seed = cfg.train_partitions, cfg.train_stride, cfg.shuffle_buffer_size, cfg.random_seed
        else:
            partitions = cfg.val_partitions if split == "val" else cfg.test_partitions
            stride, shuffle, seed = cfg.eval_stride or self.horizon_len, 0, cfg.eval_seed
        return JaneStreetStreamDataset(
            cfg, partitions, self.context_len, self.horizon_len,
            stride=stride, shuffle_buffer_size=shuffle, seed=seed,
            max_rows_per_symbol=cfg.max_rows_per_symbol,
        )

    def loader(self, split: str, batch_size: int) -> DataLoader:
        # IterableDataset: no shuffle=True; train randomization lives in the shuffle buffer.
        return DataLoader(
            self.dataset(split), batch_size=batch_size,
            drop_last=(split == "train"), collate_fn=collate_batch,
        )
