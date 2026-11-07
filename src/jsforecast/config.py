"""
config.py
---------
Centralized configuration for the Jane Street + TimesFM pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# Hugging Face repo per model type. TimesFM 2.5 uses the `transformers`-format
# checkpoint (needed for LoRA); TimesFM 3.0 only ships in the standalone
# `timesfm` package format.
MODEL_IDS = {
    "timesfm2_5": "google/timesfm-2.5-200m-transformers",
    "timesfm3": "google/timesfm-3.0-pytorch",
}


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

    # Fraction of unique symbol_ids to keep (chosen deterministically from the
    # seeded shuffle). 1.0 keeps all symbols; the dry runs use 0.03 (= 1 symbol).
    symbol_sample_rate: float = 1.0

    # --- Streaming (pyarrow.dataset) settings ---
    # Rows pyarrow reads per RecordBatch while scanning (NOT the training batch size).
    arrow_batch_size: int = 65536

    # How many *new* rows must stream past a symbol between emitted training
    # windows (1 = every possible window, i.e. maximum overlap/data reuse).
    train_stride: int = 1
    # Stride for val/test windows. None defaults to horizon_len (non-overlapping).
    eval_stride: Optional[int] = None

    # Streaming shuffle buffer size for training (reservoir-style local shuffle).
    shuffle_buffer_size: int = 4096

    random_seed: int = 42
    # Seed for the symbol sampling of the val/test splits. Kept separate from
    # `random_seed` so the evaluation symbols stay fixed across training seeds.
    eval_seed: int = 0


@dataclass
class ModelConfig:
    model_type: str = "timesfm2_5"          # key of MODEL_IDS
    model_id: Optional[str] = None          # None -> MODEL_IDS[model_type]

    context_len: int = 512                  # capped at the model's own limit at load time
    horizon_len: int = 1                    # next-step prediction, matching the Jane Street task
    torch_dtype: str = "bfloat16"           # TimesFM 2.5 only; use "float32" on CPU-only setups
    device: str = "auto"                    # "auto" -> cuda if available, else cpu

    # --- fine-tuning (both models) ---
    use_lora: bool = True                   # False -> fully fine-tune all weights (TimesFM 2.5 only)
    use_xreg: bool = True                   # attach the FeatureCorrectionHead when training
    feature_head_hidden_dim: int = 128      # width of the two MLP hidden layers
    feature_head_dropout: float = 0.1       # dropout in the feature head

    # --- TimesFM 3.0 ---
    symmetric: bool = True                  # average forecast(x) and -forecast(-x); doubles compute
    use_covariates: bool = False            # past-only covariates; zero-shot only (not with training/adapters)
    per_core_batch_size: int = 32           # internal inference chunk size (does not change results)

    def __post_init__(self):
        if self.model_type not in MODEL_IDS:
            raise ValueError(f"Unknown model_type {self.model_type!r}; expected one of {sorted(MODEL_IDS)}")

    @property
    def resolved_model_id(self) -> str:
        return self.model_id or MODEL_IDS[self.model_type]


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
    batch_size: int = 64
    eval_batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    output_dir: str = "./checkpoints/jane_street_lora_adapter"
    log_every: int = 50
    # False skips the per-epoch validation: the last epoch is saved instead of the best one.
    validate: bool = True
    # Debug/smoke-test caps (None = a full pass over the streamed partitions).
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None


@dataclass
class BacktestConfig:
    annualization_days: int = 252
    output_dir: str = "./output/backtest_results"
    # Also score the zero-shot (no adapter) base model next to the fine-tuned one.
    compare_zero_shot: bool = True
    # Batch size and cap used when predicting.
    batch_size: int = 128
    max_batches: Optional[int] = None


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfigOpts = field(default_factory=LoraConfigOpts)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
