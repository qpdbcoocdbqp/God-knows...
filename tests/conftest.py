import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jsforecast.config import DataConfig

N_SYMBOLS = 3
N_DATES = 4
N_TIMES = 10                      # rows per symbol per partition = N_DATES * N_TIMES = 40
ROWS_PER_SYMBOL = N_DATES * N_TIMES
N_FEATURES = 3
CONTEXT_LEN = 5
PARTITIONS = (0, 1, 2)


def _target_value(symbol: int, partition: int, row: int) -> float:
    """Unique, order-revealing target so tests can check window contents."""
    return symbol * 1000.0 + partition * 100.0 + row


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> str:
    """Hive-partitioned parquet shaped like the Jane Street train set (rows sorted by date, time, symbol)."""
    root = tmp_path_factory.mktemp("js") / "train.parquet"
    for part in PARTITIONS:
        rows = []
        for d in range(N_DATES):
            for t in range(N_TIMES):
                for s in range(N_SYMBOLS):
                    row = d * N_TIMES + t
                    rows.append((part * N_DATES + d, t, s, 1.0 + 0.1 * s, _target_value(s, part, row), row))
        date, time, sym, weight, target, row_idx = (np.array(c) for c in zip(*rows))
        columns = {
            "date_id": date.astype(np.int32),
            "time_id": time.astype(np.int32),
            "symbol_id": sym.astype(np.int32),
            "weight": weight.astype(np.float32),
            "responder_6": target.astype(np.float32),
        }
        for i in range(N_FEATURES):
            columns[f"feature_{i:02d}"] = (row_idx * (i + 1)).astype(np.float32)
        out = root / f"partition_id={part}"
        out.mkdir(parents=True)
        pq.write_table(pa.table(columns), out / "part-0.parquet")
    return str(root)


@pytest.fixture
def data_cfg(synthetic_data_dir) -> DataConfig:
    return DataConfig(
        data_dir=synthetic_data_dir,
        train_partitions=[0],
        val_partitions=[1],
        test_partitions=[2],
        feature_cols=[f"feature_{i:02d}" for i in range(N_FEATURES)],
        arrow_batch_size=50,          # smaller than a partition, so streaming spans several batches
        shuffle_buffer_size=8,
    )
