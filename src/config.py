"""
config.py
---------
Centralized configuration for the Jane Street + TimesFM pipeline.

Training now follows Google's official fine-tuning reference
(google-research/timesfm: timesfm-forecasting/examples/finetuning/finetune_lora.py):
LoRA via HuggingFace `transformers` + `peft`, NOT the raw `timesfm` package's
internal classes (which have no public fine-tuning entrypoint at all).
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # Directory containing the Jane Street "Real-Time Market Data Forecasting"
    # parquet partitions, e.g. .../train.parquet/partition_id=0/part-0.parquet ...
    data_dir: str = "./data/jane-street-real-time-market-data-forecasting/train.parquet"

    # Which partitions to load. Jane Street data is split into partition_id=0..9.
    train_partitions: List[int] = field(default_factory=lambda: [0])
    val_partitions: List[int] = field(default_factory=lambda: [1])
    test_partitions: List[int] = field(default_factory=lambda: [2])

    # Target responder to forecast (responder_6 is the competition's primary target).
    target_col: str = "responder_6"
    weight_col: str = "weight"
    date_col: str = "date_id"
    time_col: str = "time_id"
    symbol_col: str = "symbol_id"
    feature_cols: List[str] = field(default_factory=lambda: [f"feature_{i:02d}" for i in range(79)])

    # Optional: cap rows per symbol for fast local iteration / debugging
    # (counts streamed rows per symbol_id, not a pre-materialized slice).
    max_rows_per_symbol: Optional[int] = None

    # Fraction of unique symbol_ids to keep for training/validation/testing.
    # 1.0 means keep all symbols; 0.1 means keep ~10% of unique symbols (chosen
    # deterministically via hashing).
    symbol_sample_rate: float = 1.0

    # --- Streaming (pyarrow.dataset) settings ---
    # Number of rows pyarrow reads per RecordBatch while scanning (NOT the
    # training batch_size -- this just controls how much is pulled off disk
    # at a time during the lazy scan). Lower it if you're still memory
    # constrained; raise it for faster scanning when RAM allows.
    arrow_batch_size: int = 65536

    # How many *new* rows must stream past a symbol between emitted training
    # windows (1 = every possible window, i.e. maximum overlap/data reuse).
    train_stride: int = 1
    # Stride for val/test windows. None defaults to horizon_len (non-
    # overlapping windows), matching the previous fixed-window evaluation.
    eval_stride: Optional[int] = None

    # Streaming shuffle buffer size for training: instead of pre-sampling a
    # fixed number of random windows into memory, we keep a small reservoir
    # of `shuffle_buffer_size` in-flight windows and randomly swap new ones
    # in -- a standard streaming-shuffle approximation to full shuffling.
    shuffle_buffer_size: int = 4096

    random_seed: int = 42


@dataclass
class ModelConfig:
    # The TRANSFORMERS-format checkpoint (not the raw "-pytorch" repo used by
    # the standalone `timesfm` package). Required for the LoRA fine-tuning
    # path, which depends on `transformers.TimesFm2_5ModelForPrediction`.
    model_id: str = "google/timesfm-2.5-200m-transformers"

    context_len: int = 512          # capped at model.config.context_length at load time
    horizon_len: int = 1            # next-step prediction, matching the Jane Street task
    infer_is_positive: bool = False # returns/responders can be negative
    force_flip_invariance: bool = True
    torch_dtype: str = "bfloat16"   # matches the official example; use "float32" on CPU-only setups

    # FeatureCorrectionHead hyper-parameters
    feature_head_hidden_dim: int = 128   # width of the two MLP hidden layers
    feature_head_dropout: float = 0.1    # dropout in the feature head


@dataclass
class LoraConfigOpts:
    r: int = 4
    alpha: int = 8
    dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"


@dataclass
class TrainConfig:
    epochs: int = 2
    patience: int = 3              # early-stopping patience (epochs without val improvement)
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    device: str = "cuda"           # falls back to "cpu" automatically if unavailable
    output_dir: str = "./checkpoints/jane_street_lora_adapter"
    log_every: int = 50
    use_lora: bool = True          # set False to fully fine-tune all weights instead


@dataclass
class BacktestConfig:
    annualization_days: int = 252
    output_dir: str = "./output/backtest_results"
    # If True, also report the zero-shot (no LoRA adapter) base model's
    # R^2 on the same split, for comparison -- mirrors finetune_lora.py's
    # own --eval_only zero-shot-vs-finetuned comparison.
    compare_zero_shot: bool = True
