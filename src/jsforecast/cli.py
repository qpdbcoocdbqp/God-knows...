"""
cli.py
------
Command line entry point: `python -m jsforecast <train|backtest> ...` (or `jsforecast ...`).

Examples
--------
# Fine-tune TimesFM 2.5 (LoRA + feature head), then backtest the saved adapter on the test split
jsforecast train --backtest

# Backtest a saved adapter, next to the zero-shot base model
jsforecast backtest --adapter-dir ./checkpoints/jane_street_lora_adapter

# Zero-shot TimesFM 3.0 dry run on 3% of the symbols (with / without covariates)
jsforecast backtest --model timesfm3 --symbol-sample-rate 0.03 --test-partitions 8
jsforecast backtest --model timesfm3 --symbol-sample-rate 0.03 --test-partitions 8 --covariates --max-batches 120
"""

import argparse
import gc
from typing import Optional, Sequence

import torch

from .config import MODEL_IDS, PipelineConfig
from .data import JaneStreetDataModule
from .evaluate import Backtester
from .models import build_forecaster
from .trainer import Trainer
from .utils import seed_everything


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", dest="model_type", choices=sorted(MODEL_IDS), default=None)
    p.add_argument("--model-id", default=None, help="Override the Hugging Face repo id")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--context-len", type=int, default=None)
    p.add_argument("--horizon-len", type=int, default=None)
    p.add_argument("--symbol-sample-rate", type=float, default=None)
    p.add_argument("--train-partitions", type=int, nargs="+", default=None)
    p.add_argument("--val-partitions", type=int, nargs="+", default=None)
    p.add_argument("--test-partitions", type=int, nargs="+", default=None)
    p.add_argument("--device", default=None, help="auto | cuda | cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jsforecast", description="TimesFM x Jane Street pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Fine-tune (LoRA; TimesFM 2.5 also full) and save the best checkpoint")
    _add_common_args(t)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch-size", dest="train_batch_size", type=int, default=None)
    t.add_argument("--eval-batch-size", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--output-dir", dest="checkpoint_dir", default=None, help="Where the best adapter is saved")
    t.add_argument("--xreg", action=argparse.BooleanOptionalAction, default=None,
                   help="Attach the feature correction head")
    t.add_argument("--lora", action=argparse.BooleanOptionalAction, default=None,
                   help="LoRA adapter (--no-lora fine-tunes all weights)")
    t.add_argument("--max-train-batches", type=int, default=None)
    t.add_argument("--max-val-batches", type=int, default=None)
    t.add_argument("--backtest", action="store_true", help="Backtest the saved checkpoint on the test split afterwards")

    b = sub.add_parser("backtest", help="Score models on the val/test split")
    _add_common_args(b)
    b.add_argument("--split", choices=["val", "test"], default="test")
    b.add_argument("--adapter-dir", default=None, help="Saved checkpoint to score")
    b.add_argument("--zero-shot", action=argparse.BooleanOptionalAction, default=None,
                   help="Also score the zero-shot base model (always scored when no --adapter-dir)")
    b.add_argument("--covariates", action=argparse.BooleanOptionalAction, default=None,
                   help="TimesFM 3: feed feature columns as past-only covariates")
    b.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=None,
                   help="TimesFM 3: average forecast(x) and -forecast(-x)")
    b.add_argument("--batch-size", dest="backtest_batch_size", type=int, default=None)
    b.add_argument("--max-batches", type=int, default=None)
    b.add_argument("--output-dir", dest="results_dir", default=None, help="Where predictions / daily PnL are saved")
    return parser


def _apply(section, **overrides) -> None:
    """Sets every override that was actually given on the command line."""
    for key, value in overrides.items():
        if value is not None:
            setattr(section, key, value)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    g = vars(args).get
    _apply(cfg.data, data_dir=g("data_dir"), symbol_sample_rate=g("symbol_sample_rate"),
           train_partitions=g("train_partitions"), val_partitions=g("val_partitions"),
           test_partitions=g("test_partitions"))
    _apply(cfg.model, model_type=g("model_type"), model_id=g("model_id"), context_len=g("context_len"),
           horizon_len=g("horizon_len"), device=g("device"), use_xreg=g("xreg"), use_lora=g("lora"),
           use_covariates=g("covariates"), symmetric=g("symmetric"))
    _apply(cfg.train, epochs=g("epochs"), batch_size=g("train_batch_size"),
           eval_batch_size=g("eval_batch_size"), lr=g("lr"), output_dir=g("checkpoint_dir"),
           max_train_batches=g("max_train_batches"), max_val_batches=g("max_val_batches"))
    _apply(cfg.backtest, compare_zero_shot=g("zero_shot"), batch_size=g("backtest_batch_size"),
           max_batches=g("max_batches"), output_dir=g("results_dir"))
    cfg.model.__post_init__()  # re-validate after overrides
    return cfg


def run_train(cfg: PipelineConfig) -> str:
    seed_everything(cfg.data.random_seed)
    forecaster = build_forecaster(cfg.model, lora_cfg=cfg.lora, data_cfg=cfg.data, trainable=True)
    datamodule = JaneStreetDataModule(cfg.data, forecaster.context_len, cfg.model.horizon_len)
    result = Trainer(forecaster, datamodule, cfg.train, cfg.data).fit()
    return result.output_dir


def run_backtest(cfg: PipelineConfig, split: str, adapter_dir: Optional[str] = None) -> None:
    """Scores the fine-tuned checkpoint and/or the zero-shot model, one after the other."""
    variants = []
    if adapter_dir:
        variants.append(("finetuned", adapter_dir))
    if adapter_dir is None or cfg.backtest.compare_zero_shot:
        variants.append(("zero_shot", None))

    backtester = Backtester(cfg.data, cfg.backtest)
    for variant, variant_adapter in variants:
        forecaster = build_forecaster(cfg.model, adapter_dir=variant_adapter)
        datamodule = JaneStreetDataModule(cfg.data, forecaster.context_len, cfg.model.horizon_len)
        label = f"{cfg.model.model_type}_{variant}" + ("_xreg" if cfg.model.use_covariates else "")
        backtester.run(forecaster, datamodule.loader(split, cfg.backtest.batch_size), label)
        # release GPU memory before loading the next model
        del forecaster
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = build_config(args)

    if args.command == "train":
        output_dir = run_train(cfg)
        if args.backtest:
            run_backtest(cfg, split="test", adapter_dir=output_dir)
    else:
        run_backtest(cfg, split=args.split, adapter_dir=args.adapter_dir)


if __name__ == "__main__":
    main()
