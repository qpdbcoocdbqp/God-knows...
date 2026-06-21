"""
main.py
-------
CLI entry point.

Examples
--------
# Train (LoRA) then backtest against the held-out test partitions
python main.py --stage all

# Only train
python main.py --stage train

# Only backtest using a previously saved LoRA adapter
python main.py --stage backtest --adapter-dir ./checkpoints/jane_street_lora_adapter
"""

import argparse

from config import BacktestConfig, DataConfig, LoraConfigOpts, ModelConfig, TrainConfig
from train import train
from backtest import run_backtest


def parse_args():
    p = argparse.ArgumentParser(description="TimesFM 2.5 (LoRA) x Jane Street pipeline")
    p.add_argument("--stage", choices=["train", "backtest", "all"], default="all")
    p.add_argument("--data-dir", type=str, default=None, help="Override DataConfig.data_dir")
    p.add_argument("--adapter-dir", type=str, default=None,
                   help="LoRA adapter directory to evaluate (defaults to TrainConfig.output_dir)")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"],
                   help="Which split to backtest on")
    p.add_argument("--no-zero-shot", action="store_true",
                   help="Skip the zero-shot base-model comparison during backtest")
    return p.parse_args()


def main():
    args = parse_args()

    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    lora_cfg = LoraConfigOpts()
    bt_cfg = BacktestConfig()

    if args.data_dir:
        data_cfg.data_dir = args.data_dir
    if args.no_zero_shot:
        bt_cfg.compare_zero_shot = False

    adapter_dir = args.adapter_dir or train_cfg.output_dir

    if args.stage in ("train", "all"):
        adapter_dir, best_val_r2 = train(data_cfg, model_cfg, train_cfg, lora_cfg)
        print(f"Training complete. Best validation weighted R^2: {best_val_r2:.6f}")

    if args.stage in ("backtest", "all"):
        run_backtest(data_cfg, model_cfg, bt_cfg, adapter_dir=adapter_dir, split=args.split)


if __name__ == "__main__":
    main()
