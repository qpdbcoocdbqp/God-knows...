"""
utils.py
--------
Small helpers shared across the package (device, seeding, metrics, prediction tables).
"""

import os
import random
from typing import Sequence

import numpy as np
import pyarrow as pa
import torch

from .config import DataConfig


def get_device(preference: str = "auto") -> str:
    """Resolves "auto"/"cuda"/"cpu" to a usable device string (cuda falls back to cpu)."""
    if preference in ("auto", "cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference


def resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[name]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def jane_street_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, eps: float = 1e-12) -> float:
    """Sample-weighted, zero-mean R^2 (no demeaning of y_true, per competition spec)."""
    num = np.sum(weight * (y_true - y_pred) ** 2)
    den = np.sum(weight * (y_true ** 2)) + eps
    return float(1.0 - num / den)


def prediction_columns(data_cfg: DataConfig) -> Sequence[str]:
    return (
        data_cfg.symbol_col,
        data_cfg.date_col,
        data_cfg.time_col,
        data_cfg.weight_col,
        f"{data_cfg.target_col}_true",
        f"{data_cfg.target_col}_pred",
    )


def prediction_schema(data_cfg: DataConfig) -> pa.Schema:
    sym, date, time, weight, true, pred = prediction_columns(data_cfg)
    return pa.schema([
        (sym, pa.int64()), (date, pa.int64()), (time, pa.int64()),
        (weight, pa.float64()), (true, pa.float64()), (pred, pa.float64()),
    ])


def build_prediction_table(
    data_cfg: DataConfig,
    symbol_id: np.ndarray,
    date_id: np.ndarray,
    time_id: np.ndarray,
    weight: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
) -> pa.Table:
    """
    Builds the standard prediction table for one batch.

    `weight`, `target`, `pred` are (batch, horizon); the id arrays are (batch,).
    Rows are emitted horizon-step major (all batch rows of step 0, then step 1, ...).
    """
    horizon = pred.shape[1]
    sym, date, time, w, true, p = prediction_columns(data_cfg)
    columns = {
        sym: np.tile(np.asarray(symbol_id, dtype=np.int64), horizon),
        date: np.tile(np.asarray(date_id, dtype=np.int64), horizon),
        time: np.tile(np.asarray(time_id, dtype=np.int64), horizon),
        w: np.asarray(weight, dtype=np.float64).T.reshape(-1),
        true: np.asarray(target, dtype=np.float64).T.reshape(-1),
        p: np.asarray(pred, dtype=np.float64).T.reshape(-1),
    }
    return pa.table(columns, schema=prediction_schema(data_cfg))
