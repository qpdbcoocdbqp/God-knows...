# God knows...

Research Jane Street data. Playing with [God knows...](https://www.youtube.com/watch?v=WWB01IuMvzA)

* **About God knows...**

> 涼宮ハルヒ（CV.平野 綾）
>
> 涼宮ハルヒの完奏〜コンプリートサウンドトラック〜

## Jane Street TimesFM Pipeline

TimesFM forecasting of `responder_6` on the Kaggle "Jane Street Real-Time Market Data Forecasting" data.

- TimesFM 2.5: zero-shot, LoRA fine-tune, optional feature head (`feature_00`-`feature_78`)
- TimesFM 3.0: zero-shot, optional past-only covariates; LoRA fine-tune (our own: upstream ships no fine-tuning entry point)
- One prediction -> metrics path (weighted R², daily PnL, Sharpe-like) shared by training validation and backtests

## Project Layout

- `src/jsforecast/`: the package (`config`, `utils`, `data`, `models/`, `trainer`, `evaluate`, `cli`)
- `tests/`: pytest suite (`test_integration.py` needs real data and weights)
- `scripts/`: data download and inspection helpers
- `docs/`: experiment notes ([docs/fine-tune.md](docs/fine-tune.md))
- `data/`, `checkpoints/`, `output/`: local only, ignored by Git
- `refactor-todo.md`: refactor plan and progress

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,timesfm3]"
```

- `timesfm3` extra: TimesFM 3.0 (`timesfm[torch]==3.0.2`)
- `scripts` extra: pandas and kaggle for `scripts/`

## Prepare Data

```powershell
python scripts/download_data.py
```

- Expected layout: `data/jane-street-real-time-market-data-forecasting/train.parquet/partition_id=N/part-0.parquet`
- Override with `--data-dir`

## Usage

```powershell
# fine-tune TimesFM 2.5 (LoRA + feature head), then backtest the saved checkpoint on the test split
python -m jsforecast train --backtest

# backtest a saved checkpoint next to the zero-shot base model
python -m jsforecast backtest --adapter-dir ./checkpoints/jane_street_lora_adapter

# TimesFM 3.0 zero-shot
python -m jsforecast backtest --model timesfm3

# TimesFM 3.0 LoRA fine-tune (fp32 + symmetric averaging: use a small batch on an 8 GB GPU)
python -m jsforecast train --model timesfm3 --batch-size 32 --output-dir ./checkpoints/timesfm3_lora
python -m jsforecast backtest --model timesfm3 --adapter-dir ./checkpoints/timesfm3_lora
```

- `train`: writes the best checkpoint to `--output-dir` (default `./checkpoints/jane_street_lora_adapter`, **overwritten on improvement**)
- `backtest`: writes `predictions_<label>.parquet` and `daily_pnl_<label>.parquet` to `--output-dir` (default `./output/backtest_results`)
- `--help` on each command lists all options

## Run Tests

```powershell
pytest                    # unit tests, no downloads
pytest -m integration -s  # compare against saved baseline predictions (data + weights + checkpoints)
```

## Backtest Results (TimesFM x Jane Street dry run)

Results from `output/backtest_results/` (ignored by Git). Fine-tune setup: [docs/fine-tune.md](docs/fine-tune.md).

### Parameters (all runs)

- Test data: `partition_id=8`, `date_id` 1360-1529 (159 days)
- `symbol_sample_rate`: 0.03 (= 1 symbol, `symbol_id=0`)
- Target: `responder_6`
- `context_len`: 512
- `horizon_len`: 1
- Stride: 1 (window = 513 steps, 153,400 windows in total)
- Weighted R²: `1 - sum(w*(y-p)^2) / sum(w*y^2)`
- Sharpe-like: annualized, position = `sign(pred)`, daily PnL = `sum(w * position * y)`

### TimesFM 3.0 parameters

- Model: `google/timesfm-3.0-pytorch` (non-commercial license)
- `make_positive`: False
- `use_symmetric_averaging`: True
- Covariates (zero-shot only): `feature_00`-`feature_78` as past-only covariates (31 of 79 kept per forward pass)
- Batch size: 128 (zero-shot), 32 (covariates)

### TimesFM 3.0 LoRA parameters

- LoRA: `r=4`, `alpha=8`, `dropout=0.05`, all 204 Linear layers (attention q/k/v/out, feed-forward, input block, output head)
- Trainable parameters: 2,077,440 of 332,788,416 (0.62%)
- Train: `partition_id=0`, val: `partition_id=1`, same 3% symbol sampling, no feature head (`--no-xreg`)
- 1 epoch capped at 300 steps, batch size 32 (about 9,600 windows), validation capped at 40 batches of 128
- Optimizer: AdamW, `lr=1e-4`, `weight_decay=0.01`, grad clip 1.0
- Loss: weight-weighted MSE on the same forecast used for inference (quantiles sorted, symmetric average)
- Backtest: batch size 1024, first 13 batches

### Full test partition (159 days, 153,400 windows)

| Model | weighted R² | Total PnL | Sharpe-like |
|---|---|---|---|
| TimesFM 2.5 zero-shot | 0.8076 | 255,798 | 14.19 |
| TimesFM 3.0 zero-shot | 0.8092 | 256,223 | 14.19 |
| TimesFM 2.5 + LoRA | **0.8264** | 258,789 | 14.19 |
| TimesFM 2.5 + LoRA + xreg (feature head, 79 features) | 0.8256 | 258,641 | 14.19 |

### Same windows, shorter ranges

First N windows in time order; every model scored on identical windows.

| Model | First 3,840 windows (5 days) R² | First 19,200 windows (21 days) R² |
|---|---|---|
| TimesFM 2.5 zero-shot | 0.7908 | 0.8105 |
| TimesFM 2.5 + LoRA | **0.8177** | **0.8294** |
| TimesFM 2.5 + LoRA + xreg | 0.8161 | 0.8283 |
| TimesFM 3.0 zero-shot | 0.7946 | 0.8112 |
| TimesFM 3.0 zero-shot + covariates (31 of 79 features) | 0.7944 | not run |

### TimesFM 3.0 LoRA vs the others (first 13,312 windows, 15 dates)

Same windows for every row (first 13 batches of 1024, `partition_id=8`).

| Model | Training | weighted R² | Total PnL | Sharpe-like |
|---|---|---|---|---|
| TimesFM 2.5 zero-shot | none | 0.8155 | 17,484 | 14.93 |
| TimesFM 3.0 zero-shot | none | 0.8161 | 17,527 | 14.92 |
| TimesFM 3.0 + LoRA | 300 steps (about 9,600 windows) | 0.8199 | 17,526 | 14.93 |
| TimesFM 2.5 + LoRA | 2 epochs on partition 0 | **0.8336** | 17,704 | 14.93 |

Validation weighted R² of the TimesFM 3.0 LoRA checkpoint (40 batches of `partition_id=1`): 0.8298.

### Runtime (RTX 5070 Laptop GPU, 8 GB)

| Run | Batch size | Time per batch | Total |
|---|---|---|---|
| TimesFM 3.0 zero-shot, 153,400 windows | 128 | 1.3 s | 25 min 48 s |
| TimesFM 3.0 zero-shot, 3,840 windows | 32 | - | 75 s |
| TimesFM 3.0 + covariates, 3,840 windows | 32 | 4.2 s | 537 s |
| TimesFM 3.0 LoRA train, 300 steps + 40 validation batches | 32 / 128 | about 1.1 s per step, 4.0 s per validation batch | 487 s |
| TimesFM 3.0 backtest, LoRA + zero-shot, 13,312 windows each | 1024 | - | 693 s |

### Reproduce

```powershell
# TimesFM 3.0 zero-shot, full partition 8, 3% symbols
python -m jsforecast backtest --model timesfm3 --symbol-sample-rate 0.03 --test-partitions 8 --batch-size 128

# TimesFM 3.0 with covariates, first 120 batches (about 9 minutes on an RTX 5070 Laptop GPU)
python -m jsforecast backtest --model timesfm3 --symbol-sample-rate 0.03 --test-partitions 8 --batch-size 32 --max-batches 120 --covariates

# TimesFM 3.0 LoRA: train (300 steps), then backtest next to zero-shot on the first 13 batches
python -m jsforecast train --model timesfm3 --no-xreg --symbol-sample-rate 0.03 --epochs 1 --batch-size 32 --eval-batch-size 128 --max-train-batches 300 --max-val-batches 40 --output-dir ./checkpoints/timesfm3_lora
python -m jsforecast backtest --model timesfm3 --adapter-dir ./checkpoints/timesfm3_lora --symbol-sample-rate 0.03 --test-partitions 8 --batch-size 1024 --max-batches 13
```
