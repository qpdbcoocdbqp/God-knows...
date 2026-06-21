"""
Quick sampled backtest script -- loads ~5% of unique symbol_ids
from the test partition so we can verify the backtest pipeline
without running the full dataset.
"""
import sys
sys.path.insert(0, "src")

import pyarrow.dataset  # noqa: F401  (must come before torch on Windows)

from config import BacktestConfig, DataConfig, ModelConfig
from backtest import run_backtest

data_cfg = DataConfig(
    symbol_sample_rate=0.03,   # ~5% of unique symbols
    test_partitions=[8]
)
model_cfg = ModelConfig()
bt_cfg = BacktestConfig(compare_zero_shot=False)  # skip zero-shot for speed

pred_df, daily_df = run_backtest(
    data_cfg,
    model_cfg,
    bt_cfg,
    adapter_dir="./checkpoints/jane_street_lora_adapter",
)
