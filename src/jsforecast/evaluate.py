"""
evaluate.py
-----------
The single prediction -> metrics path used by training validation, backtests and
zero-shot runs alike:

    loader --Predictor--> prediction table --Evaluator--> Metrics

* `Predictor.run`   runs any `Forecaster` over a loader and returns the standard table
                    (symbol, date, time, weight, <target>_true, <target>_pred).
* `Evaluator.score` weighted R^2 plus a simple daily-PnL simulation
                    (position = sign(prediction)).
* `Backtester`      Predictor + Evaluator + printing + saving to `output_dir`.
"""

import itertools
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from .config import BacktestConfig, DataConfig
from .models.base import Forecaster
from .utils import build_prediction_table, ensure_dir, jane_street_r2, prediction_schema


@dataclass(frozen=True)
class Metrics:
    r2: float           # sample-weighted, zero-mean R^2
    total_pnl: float    # sum of daily PnL
    sharpe: float       # annualized Sharpe-like utility
    n_days: int
    n_rows: int


class Predictor:
    def __init__(self, data_cfg: DataConfig):
        self.data_cfg = data_cfg

    @torch.no_grad()
    def run(
        self,
        forecaster: Forecaster,
        loader: Iterable,
        max_batches: Optional[int] = None,
        desc: str = "Predicting",
    ) -> pa.Table:
        forecaster.set_training(False)
        tables = []
        for batch in tqdm(itertools.islice(loader, max_batches), desc=desc):
            pred = forecaster.predict(batch).float().cpu().numpy()
            tables.append(build_prediction_table(
                self.data_cfg,
                symbol_id=batch.symbol_id.numpy(),
                date_id=batch.date_id.numpy(),
                time_id=batch.time_id.numpy(),
                weight=batch.weight.numpy(),
                target=batch.target.numpy(),
                pred=pred,
            ))
        return pa.concat_tables(tables) if tables else prediction_schema(self.data_cfg).empty_table()


class Evaluator:
    def __init__(self, data_cfg: DataConfig, annualization_days: int = 252):
        self.data_cfg = data_cfg
        self.annualization_days = annualization_days
        self.true_col = f"{data_cfg.target_col}_true"
        self.pred_col = f"{data_cfg.target_col}_pred"

    def _check(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            raise ValueError("No windows to score: the split produced no predictions "
                             "(too few rows per symbol for context_len + horizon_len?)")

    def daily_pnl(self, table: pa.Table) -> pa.Table:
        """Daily PnL and cumulative equity; position = sign(prediction)."""
        self._check(table)
        cfg = self.data_cfg
        position = np.sign(table.column(self.pred_col).to_numpy())
        pnl = table.column(cfg.weight_col).to_numpy() * position * table.column(self.true_col).to_numpy()
        per_row = pa.table({cfg.date_col: table.column(cfg.date_col), "pnl": pnl})
        daily = (per_row.group_by(cfg.date_col).aggregate([("pnl", "sum")])
                 .select([cfg.date_col, "pnl_sum"])
                 .rename_columns([cfg.date_col, "pnl"])
                 .sort_by([(cfg.date_col, "ascending")]))
        return daily.append_column("equity", pc.cumulative_sum(daily["pnl"]))

    def score(self, table: pa.Table) -> Metrics:
        daily = self.daily_pnl(table)
        daily_pnl = daily["pnl"].to_numpy()
        n_days = len(daily_pnl)
        total = float(daily_pnl.sum())
        denom = float(np.sqrt((daily_pnl ** 2).sum())) + 1e-12
        sharpe = (total / denom) * np.sqrt(self.annualization_days / max(1, n_days))
        r2 = jane_street_r2(
            table.column(self.true_col).to_numpy(),
            table.column(self.pred_col).to_numpy(),
            table.column(self.data_cfg.weight_col).to_numpy(),
        )
        return Metrics(r2=r2, total_pnl=total, sharpe=float(sharpe), n_days=n_days, n_rows=table.num_rows)


class Backtester:
    """Predicts a split, prints the metrics and saves predictions / daily PnL under `output_dir`."""

    def __init__(self, data_cfg: DataConfig, bt_cfg: BacktestConfig):
        self.bt_cfg = bt_cfg
        self.predictor = Predictor(data_cfg)
        self.evaluator = Evaluator(data_cfg, bt_cfg.annualization_days)

    def run(self, forecaster: Forecaster, loader: Iterable, label: str) -> Metrics:
        table = self.predictor.run(forecaster, loader, max_batches=self.bt_cfg.max_batches, desc=label)
        metrics = self.evaluator.score(table)
        print(f"[{label}] Backtest results over {metrics.n_days} days:")
        print(f"  weighted R^2      : {metrics.r2:.6f}")
        print(f"  total PnL         : {metrics.total_pnl:.4f}")
        print(f"  utility (Sharpe-like, annualized): {metrics.sharpe:.4f}")

        out_dir = ensure_dir(self.bt_cfg.output_dir)
        pq.write_table(table, os.path.join(out_dir, f"predictions_{label}.parquet"))
        pq.write_table(self.evaluator.daily_pnl(table), os.path.join(out_dir, f"daily_pnl_{label}.parquet"))
        return metrics
