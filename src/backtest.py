"""
backtest.py
-----------
Backtesting utilities for the Jane Street task, aligned with the
transformers + peft LoRA pipeline in model.py / train.py.

1. `jane_street_r2` -- official-style leaderboard metric: sample-weighted,
   zero-mean R^2:
       R2 = 1 - sum(w_i * (y_i - yhat_i)^2) / sum(w_i * y_i^2)

2. `generate_predictions` -- runs a loaded model (base or LoRA-adapted)
   over a dataset via `model(past_values=...)` and returns a tidy
   DataFrame of predictions joined with truth/weight/date_id.

3. `simulate_pnl` -- turns predictions into a simple daily trading
   simulation (position = sign(prediction)) and computes an equity curve
   plus a Sharpe-like daily utility score.

4. `run_backtest` -- evaluates the fine-tuned (LoRA) model and, optionally,
   the zero-shot base model on the same split for comparison -- mirroring
   the official finetune_lora.py's own zero-shot-vs-finetuned `evaluate()`.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.compute as pc
import pyarrow.parquet as pq

# On Windows, importing pyarrow.dataset after torch can trigger a native
# access violation in some environments. Preload it before torch.
import pyarrow.dataset  # noqa: F401
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BacktestConfig, DataConfig, ModelConfig, TrainConfig
from data import build_datasets
from model import effective_context_len, load_adapter, load_base_model, load_feature_head



def jane_street_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, eps: float = 1e-12) -> float:
    """Sample-weighted, zero-mean R^2 (no demeaning of y_true, per competition spec)."""
    num = np.sum(weight * (y_true - y_pred) ** 2)
    den = np.sum(weight * (y_true ** 2)) + eps
    return float(1.0 - num / den)


def _collate(batch):
    context = torch.stack([b["context"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    context_features = torch.stack([b["context_features"] for b in batch])

    weight = torch.stack([b["weight"] for b in batch])
    symbol_id = torch.tensor([b["symbol_id"] for b in batch])
    date_id = torch.tensor([b["date_id"] for b in batch])
    time_id = torch.tensor([b["time_id"] for b in batch])

    return context, target, context_features, weight, symbol_id, date_id, time_id


# model = ft_model
# feature_head = ft_feature_head
# batch_size = 1024

@torch.no_grad()
def generate_predictions(
    model,
    feature_head,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    context_len: int,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.eval()
    if feature_head is not None:
        _ = feature_head.eval()

    _train , _valid, test_ds = build_datasets(data_cfg, context_len, model_cfg.horizon_len, load_test=True)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    rows = []
    with torch.inference_mode():
        for context, target, context_features, weight, symbol_id, date_id, time_id in tqdm(loader):
            context = context.to(device)
            base_pred = model(past_values=context, forecast_context_len=context_len).mean_predictions[:, :model_cfg.horizon_len]
            if feature_head is not None:
                context_features = context_features.to(device).float()
                delta = feature_head(context_features)
                pred = (base_pred + delta).float().cpu().detach().numpy()
            else:
                pred = base_pred.float().cpu().detach().numpy()
            target = target.numpy()
            weight = weight.numpy()
            h = pred.shape[1]
            for t in range(h): # t=0
                tmp = pa.table({
                    data_cfg.symbol_col: symbol_id.tolist(),
                    data_cfg.date_col: date_id.tolist(),
                    data_cfg.time_col: time_id.tolist(),
                    data_cfg.weight_col: weight[:, t].tolist(),
                    f"{data_cfg.target_col}_true": target[:, t].tolist(),
                    f"{data_cfg.target_col}_pred": pred[:, t].tolist(),    
                })
                if feature_head is not None:
                    tmp = tmp.append_column("delta_pred", [delta[:, t].tolist()])
                rows.append(tmp)
    output = pa.concat_tables(rows)
    return output

def simulate_pnl(df: pa.Table, data_cfg: DataConfig, bt_cfg: BacktestConfig, label: str = "model_lora") -> pd.DataFrame:
    """
    position = sign(prediction); daily_pnl = sum(weight * position * true_value) per date_id.
    utility = (sum(daily_pnl) / sqrt(sum(daily_pnl^2))) * sqrt(annualization_days / n_days)
    """
    pred_col = f"{data_cfg.target_col}_pred"
    true_col = f"{data_cfg.target_col}_true"
    df = df.append_column("position", [np.sign(df.column(pred_col))])
    df = df.append_column("pnl", pc.multiply(pc.multiply(df.column(data_cfg.weight_col), df.column("position")), df.column(true_col)))

    daily = df.group_by(data_cfg.date_col).aggregate([("pnl", "sum")]).sort_by([(data_cfg.date_col, "ascending")]).rename_columns(["date_id", "pnl"])
    daily = daily.append_column("equity", pc.cumulative_sum(daily["pnl"]))

    # 4. 統計指標計算
    n_days = len(daily)
    sum_pnl = pc.sum(daily["pnl"]).as_py()
    sum_sq = np.sqrt(pc.sum(pc.power(daily["pnl"], 2)).as_py()) + 1e-12
    days_ratio = bt_cfg.annualization_days / max(1, n_days)
    utility = (sum_pnl / sum_sq) * np.sqrt(days_ratio)

    overall_r2 = jane_street_r2(
        df[true_col].to_numpy(), df[pred_col].to_numpy(), df[data_cfg.weight_col].to_numpy()
    )

    print(f"[{label}] Backtest results over {n_days} days:")
    print(f"  weighted R^2      : {overall_r2:.6f}")
    print(f"  total PnL         : {sum_pnl:.4f}")
    print(f"  utility (Sharpe-like, annualized): {utility:.4f}")

    os.makedirs(bt_cfg.output_dir, exist_ok=True)
    pq.write_table(df, os.path.join(bt_cfg.output_dir, f"predictions_{label}.parquet"))
    pq.write_table(daily, os.path.join(bt_cfg.output_dir, f"daily_pnl_{label}.parquet"))

    return daily


def run_backtest(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    bt_cfg: BacktestConfig,
    adapter_dir: Optional[str] = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if adapter_dir and os.path.isdir(adapter_dir):
        ft_model = load_adapter(model_cfg, adapter_dir, device)
        try:
            ft_feature_head = load_feature_head(adapter_dir, device)
        except:
            ft_feature_head = None
        context_len = effective_context_len(ft_model, model_cfg.context_len)
        ft_pred_df = generate_predictions(ft_model, ft_feature_head, data_cfg, model_cfg, context_len)
        ft_daily = simulate_pnl(ft_pred_df, data_cfg, bt_cfg, label="finetuned")
    else:
        print(f"WARNING: no adapter found at {adapter_dir!r} -- skipping fine-tuned evaluation.")
        ft_pred_df, ft_daily = None, None

    if bt_cfg.compare_zero_shot:
        base_model = load_base_model(model_cfg, device)
        context_len = effective_context_len(base_model, model_cfg.context_len)
        base_pred_df = generate_predictions(base_model, None, data_cfg, model_cfg, context_len)
        simulate_pnl(base_pred_df, data_cfg, bt_cfg, label="zero_shot")

    return ft_pred_df, ft_daily
