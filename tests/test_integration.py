"""
Regression check against predictions saved by the pre-refactor pipeline.

Needs the real Jane Street data, the downloaded model weights and the saved
checkpoints / prediction files, so it is deselected by default:

    pytest -m integration -s
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from jsforecast.config import DataConfig, ModelConfig
from jsforecast.data import JaneStreetDataModule
from jsforecast.evaluate import Predictor
from jsforecast.models import build_forecaster

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "output" / "backtest_results"
CHECKPOINTS = ROOT / "checkpoints"
# The baselines were produced with batch size 1024. The LoRA path runs in bf16, whose
# result depends slightly on the batch size (max diff ~1e-2 at 128), so use the same size
# to compare exactly.
BATCH_SIZE = 1024
N_BATCHES = 2

# (baseline file, adapter dir, ModelConfig overrides)
CASES = {
    "2.5 zero-shot": ("predictions_zero_shot.parquet", None, {}),
    "2.5 lora": ("predictions_model_lora.parquet", "jane_street_lora_adapter", {}),
    "2.5 lora + xreg head": ("predictions_model_xreg.parquet", "jane_street_lora_adapter_and_xreg", {}),
    "3.0 zero-shot": ("predictions_timesfm3_zero_shot.parquet", None, {"model_type": "timesfm3"}),
}


@pytest.mark.parametrize("name", CASES)
def test_predictions_match_the_pre_refactor_baseline(name):
    baseline_file, adapter, overrides = CASES[name]
    baseline_path = RESULTS / baseline_file
    adapter_dir = CHECKPOINTS / adapter if adapter else None
    data_cfg = DataConfig(symbol_sample_rate=0.03, test_partitions=[8])
    if not baseline_path.exists() or not Path(data_cfg.data_dir).exists():
        pytest.skip("baseline predictions or data not available")
    if adapter_dir is not None and not adapter_dir.exists():
        pytest.skip(f"checkpoint {adapter_dir} not available")

    forecaster = build_forecaster(ModelConfig(**overrides), adapter_dir=str(adapter_dir) if adapter_dir else None)
    loader = JaneStreetDataModule(data_cfg, forecaster.context_len, 1).loader("test", BATCH_SIZE)
    new = Predictor(data_cfg).run(forecaster, loader, max_batches=N_BATCHES)
    old = pq.read_table(baseline_path).slice(0, new.num_rows)

    assert new.num_rows == old.num_rows == N_BATCHES * BATCH_SIZE
    for col in ("symbol_id", "date_id", "time_id", "weight", "responder_6_true"):
        assert np.array_equal(new[col].to_numpy(), old[col].to_numpy()), col

    new_pred, old_pred = new["responder_6_pred"].to_numpy(), old["responder_6_pred"].to_numpy()
    diff = np.abs(new_pred - old_pred)
    print(f"\n[{name}] max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} "
          f"pred std={old_pred.std():.3e}")
    assert diff.max() == 0.0
