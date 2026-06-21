"""
data.py
-------
Lazy, batched Jane Street data loading via `pyarrow.dataset` -- NO pandas,
no full-split materialization.

Why: the previous pandas-based loader (`pd.read_parquet` + `pd.concat` +
per-symbol numpy arrays for the *entire* split) could use many GB of RAM
and get silently killed by the OS with no Python traceback. This version:

  1. Opens the parquet directory as a `pyarrow.dataset.Dataset` (no data
     read yet -- just schema/file discovery).
  2. Scans it with `columns=[...]` (only the 5 columns we need) and
     `filter=ds.field("partition_id").isin([...])` so pyarrow only opens
     the requested partition files (partition pruning) and never
     materializes the other ~85 feature columns at all.
  3. Streams `RecordBatch`es of size `DataConfig.arrow_batch_size`
     (e.g. 50,000 rows) one at a time via `scanner.to_batches()` -- a
     generator, not a bulk read.
  4. Maintains a small, BOUNDED per-symbol sliding window (a
     `collections.deque(maxlen=context_len+horizon_len)`) as rows stream
     past. Once a symbol's window is full, a training/eval sample is
     emitted directly as torch tensors. Memory use is O(num_symbols *
     window_len), not O(rows in split).

Assumption (standard for this dataset): within each partition file, rows
are already ordered by (date_id, time_id, symbol_id) -- i.e. roughly time-
ordered as you scan forward. If your copy of the data isn't sorted this
way, set `DataConfig.assume_sorted = False` to force an in-batch sort
(still streaming -- only sorts one Arrow batch at a time, not the whole
split).
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset

from config import DataConfig


def _make_scanner(data_cfg: DataConfig, partitions: List[int], sampled_symbols: Optional[List[int]] = None):
    dataset = ds.dataset(data_cfg.data_dir, format="parquet", partitioning="hive")
    needed = [data_cfg.date_col, data_cfg.time_col, data_cfg.symbol_col,
              data_cfg.weight_col, data_cfg.target_col] + list(data_cfg.feature_cols)
    partition_filt = ds.field("partition_id").isin(partitions)

    if sampled_symbols is not None:
        symbol_filt = ds.field(data_cfg.symbol_col).isin(sampled_symbols)
        filt = partition_filt & symbol_filt
    else:
        filt = partition_filt

    return dataset.scanner(columns=needed, filter=filt, batch_size=data_cfg.arrow_batch_size)


class JaneStreetStreamDataset(IterableDataset):
    """
    Streams Jane Street rows from parquet via pyarrow.dataset and yields
    per-symbol (context, target, weight, context_features, target_features)
    windows as torch tensors, without ever materializing a full split in memory.

    `shuffle_buffer_size > 0` enables a streaming shuffle buffer (reservoir-
    style local randomization) for training; set to 0 for deterministic,
    in-stream-order iteration (used for val/test).

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
        # _sampled_symbols is populated lazily on first __iter__ when
        # symbol_sample_rate < 1.0; None means "accept all".
        self._sampled_symbols: Optional[set] = None
  
    def _symbol_included(self, sid: int) -> bool:
        """Returns True when the symbol should be included in this dataset."""
        if self._sampled_symbols is None:
            return True
        return sid in self._sampled_symbols

    def _build_sampled_symbols(self, rng: random.Random) -> None:
        """Scan parquet once to collect unique symbol_ids, then subsample."""
        if self.data_cfg.symbol_sample_rate >= 1.0:
            self._sampled_symbols = None
            return
        dataset = ds.dataset(self.data_cfg.data_dir, format="parquet", partitioning="hive")
        partition_filt = ds.field("partition_id").isin(self.partitions)
        symbol_scanner = dataset.scanner(
            columns=[self.data_cfg.symbol_col],
            filter=partition_filt,
            batch_size=self.data_cfg.arrow_batch_size,
        )
        symbols: set = set()
        for batch in symbol_scanner.to_batches():
            symbols.update(batch.column(self.data_cfg.symbol_col).to_pylist())
        symbols_list = sorted(symbols)
        rng.shuffle(symbols_list)
        k = max(math.floor(len(symbols_list) * self.data_cfg.symbol_sample_rate), 1)
        self._sampled_symbols = set(symbols_list[:k])
    def _partitions_for_worker(self) -> List[int]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self.partitions
        # split partition files across DataLoader workers so they don't
        # each re-stream the same data
        return self.partitions[worker_info.id::worker_info.num_workers]

    def _emit(self, symbol_id: int, buf: deque):
        items = list(buf)
        ctx = np.array([it["target"] for it in items[:self.context_len]], dtype=np.float32)
        tgt = np.array([it["target"] for it in items[self.context_len:]], dtype=np.float32)
        wgt = np.array([it["weight"] for it in items[self.context_len:]], dtype=np.float32)
        date_id = items[self.context_len]["date"]
        time_id = items[self.context_len]["time"]
        ctx_feats = np.stack([it["features"] for it in items[:self.context_len]], axis=0)
        tgt_feats = np.stack([it["features"] for it in items[self.context_len:]], axis=0)
        
        return {
            "context": torch.from_numpy(ctx),
            "target": torch.from_numpy(tgt),
            "weight": torch.from_numpy(wgt),
            "context_features": torch.from_numpy(ctx_feats),
            "target_features": torch.from_numpy(tgt_feats),
            "symbol_id": symbol_id,
            "date_id": date_id,
            "time_id": time_id,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng = random.Random(seed)

        # Build the sampled symbol set once per iterator (lazy, seeded).
        self._build_sampled_symbols(rng)

        partitions = self._partitions_for_worker()
        if not partitions:
            return iter([])

        # Pass sampled_symbols list to scanner so pyarrow filters at read time.
        sampled_list = sorted(self._sampled_symbols) if self._sampled_symbols is not None else None
        scanner = _make_scanner(self.data_cfg, partitions, sampled_symbols=sampled_list)

        buffers: Dict[int, deque] = {}
        since_last_emit: Dict[int, int] = {}
        row_count: Dict[int, int] = {}
        shuffle_pool: List[dict] = []

        def _maybe_yield_now(sample):
            if self.shuffle_buffer_size <= 0:
                return sample, None
            shuffle_pool.append(sample)
            if len(shuffle_pool) >= self.shuffle_buffer_size:
                idx = rng.randrange(len(shuffle_pool))
                shuffle_pool[idx], shuffle_pool[-1] = shuffle_pool[-1], shuffle_pool[idx]
                return shuffle_pool.pop(), None
            return None, None

        for batch in scanner.to_batches():
            date_arr = batch.column(self.data_cfg.date_col).to_numpy(zero_copy_only=False)
            time_arr = batch.column(self.data_cfg.time_col).to_numpy(zero_copy_only=False)
            symbol_arr = batch.column(self.data_cfg.symbol_col).to_numpy(zero_copy_only=False)
            weight_arr = batch.column(self.data_cfg.weight_col).to_numpy(zero_copy_only=False).astype(np.float32)
            target_arr = batch.column(self.data_cfg.target_col).to_numpy(zero_copy_only=False).astype(np.float32)

            # Stack feature columns into shape (batch_size, num_features)
            feature_matrix = np.stack([
                batch.column(col).to_numpy(zero_copy_only=False).astype(np.float32)
                for col in self.data_cfg.feature_cols
            ], axis=1)
            np.nan_to_num(feature_matrix, copy=False, nan=0.0)

            valid = ~np.isnan(target_arr)
            for i in np.nonzero(valid)[0]:
                sid = int(symbol_arr[i])

                # Symbol-level sampling is already handled by the pyarrow filter
                # in _make_scanner, so no need to call _symbol_included here.

                if self.max_rows_per_symbol is not None:
                    row_count[sid] = row_count.get(sid, 0) + 1
                    if row_count[sid] > self.max_rows_per_symbol:
                        continue

                buf = buffers.setdefault(sid, deque(maxlen=self.window_len))
                # buf.append((float(target_arr[i]), float(weight_arr[i]), int(date_arr[i]), int(time_arr[i]), feature_matrix[i]))
                buf.append({
                    "target": float(target_arr[i]),
                    "weight": float(weight_arr[i]),
                    "date": int(date_arr[i]),
                    "time": int(time_arr[i]),
                    "features": feature_matrix[i]
                })
                cnt = since_last_emit.get(sid, self.stride)
                since_last_emit[sid] = cnt + 1

                if len(buf) == self.window_len and since_last_emit[sid] >= self.stride:
                    since_last_emit[sid] = 0
                    sample = self._emit(sid, buf)
                    ready, _ = _maybe_yield_now(sample)
                    if ready is not None:
                        yield ready

        # flush remaining shuffle-buffered samples in random order at end of epoch
        rng.shuffle(shuffle_pool)
        for sample in shuffle_pool:
            yield sample


def build_datasets(data_cfg: DataConfig, context_len: int, horizon_len: int, load_test: bool = False):
    train_ds = JaneStreetStreamDataset(
        data_cfg, data_cfg.train_partitions, context_len, horizon_len,
        stride=data_cfg.train_stride,
        shuffle_buffer_size=data_cfg.shuffle_buffer_size,
        seed=data_cfg.random_seed,
        max_rows_per_symbol=data_cfg.max_rows_per_symbol,
    )
    val_ds = JaneStreetStreamDataset(
        data_cfg, data_cfg.val_partitions, context_len, horizon_len,
        stride=data_cfg.eval_stride or horizon_len,
        shuffle_buffer_size=0,
        max_rows_per_symbol=data_cfg.max_rows_per_symbol,
    )
    if load_test:
        test_ds = JaneStreetStreamDataset(
            data_cfg, data_cfg.test_partitions, context_len, horizon_len,
            stride=data_cfg.eval_stride or horizon_len,
            shuffle_buffer_size=0,
            max_rows_per_symbol=data_cfg.max_rows_per_symbol,
        )
    else:
        test_ds = None

    return train_ds, val_ds, test_ds
