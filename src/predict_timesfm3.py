"""
predict_timesfm3.py
-------------------
Zero-shot Jane Street forecasting with TimesFM 3.0 (`google/timesfm-3.0-pytorch`).

TimesFM 3 ships in the standalone `timesfm` package (>=3.0, module `timesfm3`);
there is no `transformers` class and no official fine-tuning entrypoint for it,
so unlike `train.py` (TimesFM 2.5 + LoRA) this is inference only.

It reuses the existing data pipeline (`data.build_datasets`) and metrics
(`backtest.jane_street_r2` / `backtest.simulate_pnl`), so the numbers are directly
comparable with the TimesFM 2.5 zero-shot / LoRA results in docs/fine-tune.md.

Notes
-----
* Responders can be negative, so `make_positive` is always False.
* `--symmetric` averages forecast(x) and -forecast(-x) (the TimesFM 3 analogue of
  `force_flip_invariance` in the 2.5 config). It doubles the compute.
* `--covariates` feeds the 79 anonymous `feature_XX` columns as past-only
  covariates (native TimesFM 3 feature). The model handles at most 32 variates per
  forward pass, so `TimesFM3Evaluator` randomly keeps 31 of them (seed 42).

Examples
--------
# Dry run on 3% of the symbols in partition 8 (same setup as docs/fine-tune.md)
python src/predict_timesfm3.py --symbol-sample-rate 0.03 --partition 8

# Quick smoke test: stop after 20 batches
python src/predict_timesfm3.py --symbol-sample-rate 0.03 --max-batches 20

# With past-only covariates
python src/predict_timesfm3.py --symbol-sample-rate 0.03 --covariates
"""

# On Windows, importing pyarrow.dataset after torch can trigger a native
# access violation in some environments. Preload it before torch.
import pyarrow.dataset  # noqa: F401
import argparse
import itertools

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backtest import _collate, simulate_pnl
from config import BacktestConfig, DataConfig, ModelConfig
from data import build_datasets

TIMESFM3_MODEL_ID = "google/timesfm-3.0-pytorch"


def load_forecaster(model_id: str, batch_size: int, device: str, covariates: bool):
    from timesfm3 import ModelConfig as TimesFM3Config
    from timesfm3 import TimesFM3Evaluator, TimesFM3Forecaster

    cfg = TimesFM3Config(checkpoint_path=model_id, per_core_batch_size=batch_size, device=device)
    # The evaluator subclass adds automatic variate chunking/subsampling, which is
    # needed once covariates are used; the plain forecaster is enough otherwise.
    return TimesFM3Evaluator(cfg) if covariates else TimesFM3Forecaster(cfg)


@torch.no_grad()
def generate_predictions(
    forecaster,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    context_len: int,
    batch_size: int,
    symmetric: bool,
    covariates: bool,
    max_batches: int | None = None,
) -> pa.Table:
    horizon = model_cfg.horizon_len
    _, _, test_ds = build_datasets(data_cfg, context_len, horizon, load_test=True)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    if max_batches is not None:
        loader = itertools.islice(loader, max_batches)

    predict_kwargs = dict(
        horizon=horizon,
        return_quantiles=False,
        use_symmetric_averaging=symmetric,
        make_positive=False,
    )

    tables = []
    for context, target, context_features, weight, symbol_id, date_id, time_id in tqdm(loader):
        contexts = list(context.numpy())
        if covariates:
            # (batch, ctx, 79) -> list of (79, ctx) past-only covariates
            predict_kwargs["past_only_covariates"] = list(context_features.numpy().transpose(0, 2, 1))
        outs = list(forecaster.predict_batch(contexts, **predict_kwargs))
        pred = np.stack([np.asarray(o.forecast)[:horizon] for o in outs])

        target, weight = target.numpy(), weight.numpy()
        for t in range(horizon):
            tables.append(pa.table({
                data_cfg.symbol_col: symbol_id.tolist(),
                data_cfg.date_col: date_id.tolist(),
                data_cfg.time_col: time_id.tolist(),
                data_cfg.weight_col: weight[:, t].tolist(),
                f"{data_cfg.target_col}_true": target[:, t].tolist(),
                f"{data_cfg.target_col}_pred": pred[:, t].tolist(),
            }))
    return pa.concat_tables(tables)


def parse_args():
    p = argparse.ArgumentParser(description="TimesFM 3 zero-shot x Jane Street")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--partition", type=int, nargs="+", default=[8], help="Test partition ids")
    p.add_argument("--symbol-sample-rate", type=float, default=1.0)
    p.add_argument("--context-len", type=int, default=512)
    p.add_argument("--horizon-len", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--model-id", type=str, default=TIMESFM3_MODEL_ID)
    p.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--covariates", action="store_true", help="Use feature_00..78 as past-only covariates")
    p.add_argument("--max-batches", type=int, default=None, help="Stop early (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_cfg = DataConfig(test_partitions=args.partition, symbol_sample_rate=args.symbol_sample_rate)
    if args.data_dir:
        data_cfg.data_dir = args.data_dir
    model_cfg = ModelConfig(context_len=args.context_len, horizon_len=args.horizon_len)
    bt_cfg = BacktestConfig()

    forecaster = load_forecaster(args.model_id, args.batch_size, device, args.covariates)
    preds = generate_predictions(
        forecaster, data_cfg, model_cfg, args.context_len, args.batch_size,
        args.symmetric, args.covariates, args.max_batches,
    )
    label = "timesfm3_xreg" if args.covariates else "timesfm3_zero_shot"
    simulate_pnl(preds, data_cfg, bt_cfg, label=label)


if __name__ == "__main__":
    main()
