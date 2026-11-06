# God knows...

Research Jane Street data. Playing with [God knows...](https://www.youtube.com/watch?v=WWB01IuMvzA)

* **About God knows...**

> 涼宮ハルヒ（CV.平野 綾）
>
> 涼宮ハルヒの完奏〜コンプリートサウンドトラック〜

## Jane Street Kaggle Analysis

This project contains a Python workflow for analyzing Kaggle Jane Street competition data. The current focus is exploratory data analysis that works with local CSV or Parquet files downloaded from Kaggle.

```bash
# search dataset
kaggle datasets list -s "jane street"
```

## Project Layout

- `src/jane_street_analysis/`: reusable Python analysis package.
- `tests/`: automated tests for analysis helpers.
- `data/`: local Kaggle data directory. This folder is ignored by Git.
- `reports/`: generated summaries and charts. This folder is ignored by Git.
- `READMD.md`: original task note.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

For report generation without charts, install the base project with `pip install -e .`.

## Prepare Data

Download a Jane Street competition dataset from Kaggle and place the main training file under `data/`, for example:

```text
data/train.csv
```

Large Kaggle datasets and generated reports should stay local and should not be committed.

## Run Analysis

```powershell
jane-street-analyze --input data/train.csv --output reports/jane_street_summary
```

To try the workflow before downloading Kaggle data, run it against the included sample:

```powershell
jane-street-analyze --input examples/sample_train.csv --output reports/sample_summary
```

The command creates:

- `summary.md`: dataset shape, missingness, target/action summaries, and numeric feature overview.
- `missing_values.csv`: missing value counts and rates.
- `numeric_profile.csv`: descriptive statistics for numeric columns.
- `target_distribution.png`: target distribution chart when a `resp` or `target` column exists.

## Run Tests

```powershell
pytest
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

- Model: `google/timesfm-3.0-pytorch` (non-commercial license), zero-shot only
- `make_positive`: False
- `use_symmetric_averaging`: True
- Covariates: `feature_00`-`feature_78` as past-only covariates (31 of 79 kept per forward pass)
- Batch size: 128 (zero-shot), 32 (covariates)

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

### Runtime (RTX 5070 Laptop GPU, 8 GB)

| Run | Batch size | Time per batch | Total |
|---|---|---|---|
| TimesFM 3.0 zero-shot, 153,400 windows | 128 | 1.3 s | 25 min 48 s |
| TimesFM 3.0 zero-shot, 3,840 windows | 32 | - | 75 s |
| TimesFM 3.0 + covariates, 3,840 windows | 32 | 4.2 s | 537 s |

### Reproduce

```powershell
# TimesFM 3.0 zero-shot, full partition 8, 3% symbols
python src/predict_timesfm3.py --symbol-sample-rate 0.03 --batch-size 128

# TimesFM 3.0 with covariates, first 120 batches (about 9 minutes on an RTX 5070 Laptop GPU)
python src/predict_timesfm3.py --symbol-sample-rate 0.03 --batch-size 32 --max-batches 120 --covariates
```
