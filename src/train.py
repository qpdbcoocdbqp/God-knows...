"""
train.py
--------
Fine-tunes TimesFM 2.5 with LoRA on Jane Street data, streamed lazily from
parquet via `data.JaneStreetStreamDataset` (pyarrow.dataset-based -- no
pandas, no full-split materialization).

Differences from Google's official `finetune_lora.py` reference:
  1. Loss: `outputs.loss` (the model's own, unweighted) is NOT used --
     Jane Street's evaluation is sample-weighted by the `weight` column.
     We take `outputs.mean_predictions` and apply `JaneStreetWeightedMSELoss`.
  2. Best-adapter selection / early stopping is on validation weighted R^2
     (the competition's actual metric), not raw loss.
  3. Because the training set is a streaming `IterableDataset` (no known
     length), the LR schedule uses `ReduceLROnPlateau` keyed off validation
     R^2 instead of a step-count-based `CosineAnnealingLR`.
  4. "One epoch" = one full sequential pass over the streamed train
     partitions (with a shuffle buffer for local randomization), rather
     than a fixed pre-sampled window count.
"""
import sys
sys.path.append("src/")

import os

# On Windows, importing pyarrow.dataset after torch can trigger a native
# access violation in some environments. Preload it before torch.
import pyarrow.dataset  # noqa: F401
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DataConfig, LoraConfigOpts, ModelConfig, TrainConfig
from data import build_datasets
from losses import JaneStreetWeightedMSELoss
from model import apply_lora, build_feature_head, effective_context_len, load_base_model
from model import save_feature_head
from backtest import jane_street_r2


def _device(train_cfg: TrainConfig) -> str:
    if train_cfg.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _collate(batch):
    context = torch.stack([b["context"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    weight = torch.stack([b["weight"] for b in batch])
    context_features = torch.stack([b["context_features"] for b in batch])
    target_features = torch.stack([b["target_features"] for b in batch])
    return context, target, weight, context_features, target_features


@torch.no_grad()
def _evaluate(model, feature_head, loader: DataLoader, device: str, context_len: int, horizon_len: int) -> float:
    model.eval()
    if feature_head is not None:
        feature_head.eval()
    preds, trues, weights = [], [], []
    for context, target, weight, context_features, target_features in tqdm(loader, desc="Evaluating"):
        context = context.to(device)
        target = target.to(device)
        context_features = context_features.to(device).float()
        base_pred = model(past_values=context, forecast_context_len=context_len).mean_predictions[:, :horizon_len].float()
        if feature_head is not None:
            delta = feature_head(context_features)
            pred = (base_pred + delta).cpu()
        else:
            pred = base_pred.cpu()
        preds.append(pred)
        trues.append(target.cpu())
        weights.append(weight)
    if not preds:
        return float("-inf")
    preds = torch.cat(preds).numpy().reshape(-1)
    trues = torch.cat(trues).numpy().reshape(-1)
    weights = torch.cat(weights).numpy().reshape(-1)
    return jane_street_r2(trues, preds, weights)

def train(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    lora_cfg: LoraConfigOpts,
):
    device = _device(train_cfg)

    model = load_base_model(model_cfg, device)
    context_len = effective_context_len(model, model_cfg.context_len)
    horizon_len = model_cfg.horizon_len
    all_trainable_params = []
    if train_cfg.use_lora:
        model = apply_lora(model, lora_cfg)
        all_trainable_params += [p for p in model.parameters() if p.requires_grad]
    else:
        all_trainable_params += list(model.parameters())
        for p in all_trainable_params:
            p.requires_grad_(True)

    if train_cfg.use_xreg:
        # Build the feature correction head and add its params to the optimizer.
        feature_head = build_feature_head(model_cfg, data_cfg, context_len, horizon_len).to(device)
        all_trainable_params += list(feature_head.parameters())

    train_ds, val_ds, _ = build_datasets(data_cfg, context_len, horizon_len)
    # IterableDataset (streaming) -- do NOT pass shuffle=True; randomization
    # is handled inside JaneStreetStreamDataset via its shuffle buffer.
    train_loader = DataLoader(train_ds, batch_size=64, # train_cfg.batch_size,
                               drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=128, # train_cfg.batch_size,
                             collate_fn=_collate)

    loss_fn = JaneStreetWeightedMSELoss()
    optimizer = torch.optim.AdamW(all_trainable_params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    # No fixed T_max here: the streaming train_loader has no __len__, so we
    # can't precompute "epochs * steps_per_epoch" the way a CosineAnnealingLR
    # schedule needs. ReduceLROnPlateau on validation R^2 works fine without
    # knowing the dataset size upfront.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    os.makedirs(train_cfg.output_dir, exist_ok=True)
    best_r2 = -float("inf")
    epochs_no_improve = 0
    # epoch = 1
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        if train_cfg.use_xreg:
            feature_head.train()
        epoch_loss, n_batches = 0.0, 0

        for step, (context, target, weight, context_features, target_features) in tqdm(enumerate(train_loader, start=1)):
            # print(context.shape, target.shape, weight.shape, context_features.shape, target_features.shape)
            # break
            context = context.to(device)
            target = target.to(device)
            weight = weight.to(device)
            outputs = model(past_values=context, future_values=target, forecast_context_len=context_len)
            base_pred = outputs.mean_predictions[:, :model_cfg.horizon_len].float()
            if train_cfg.use_xreg:
                context_features = context_features.to(device).float()
                delta = feature_head(context_features)
                pred = base_pred + delta
            else:
                pred = base_pred

            loss = loss_fn(pred, target.float(), weight.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable_params, train_cfg.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            if step % train_cfg.log_every == 0:
                print(f"[epoch {epoch}] step {step} (streamed) loss={epoch_loss / step:.6f}", flush=True)

        avg_train_loss = epoch_loss / max(1, n_batches)
        if train_cfg.use_xreg:
            val_r2 = _evaluate(model, feature_head, val_loader, device, context_len, model_cfg.horizon_len)
        else:
            val_r2 = _evaluate(model, None, val_loader, device, context_len, model_cfg.horizon_len)
        scheduler.step(val_r2)
        print(f"[epoch {epoch}/{train_cfg.epochs}] batches={n_batches} "
              f"train_weighted_loss={avg_train_loss:.6f} val_weighted_R2={val_r2:.6f}", flush=True)

        if val_r2 > best_r2:
            best_r2 = val_r2
            epochs_no_improve = 0
            model.save_pretrained(train_cfg.output_dir)
            if train_cfg.use_xreg:
                save_feature_head(feature_head, train_cfg.output_dir)
                print(f"  -> new best (val_R2={best_r2:.6f}), saved adapter + feature head to {train_cfg.output_dir}")
            else:
                print(f"  -> new best (val_R2={best_r2:.6f}), saved adapter to {train_cfg.output_dir}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {train_cfg.patience} epochs).")
                break

    print(f"Training complete. Best validation weighted R^2: {best_r2:.6f}")
    return train_cfg.output_dir, best_r2


if __name__ == "__main__":
    data_cfg = DataConfig(symbol_sample_rate=0.03)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(use_xreg=False)
    lora_cfg = LoraConfigOpts()
    train(data_cfg, model_cfg, train_cfg, lora_cfg)
