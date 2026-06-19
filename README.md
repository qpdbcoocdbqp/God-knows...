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
