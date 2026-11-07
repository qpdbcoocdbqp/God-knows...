"""
trainer.py
----------
Fine-tunes a trainable `Forecaster` on Jane Street data (streamed from parquet).

Differences from Google's official `finetune_lora.py`:
  1. The loss is `JaneStreetWeightedMSELoss` on `Forecaster.predict`, not the model's
     own unweighted loss: the competition metric is weighted by the `weight` column.
  2. The best checkpoint / early stopping use validation weighted R^2 (the actual
     metric), scored through the same `Predictor` + `Evaluator` path as backtests.
  3. The train set is a streaming `IterableDataset` (no length), so the LR schedule is
     `ReduceLROnPlateau` on validation R^2 instead of a step-count-based schedule.
  4. "One epoch" = one full pass over the streamed train partitions.
"""

import itertools
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm  # notebook widget in Jupyter / interactive window, text bar in a terminal

from .config import DataConfig, TrainConfig
from .data import JaneStreetDataModule
from .evaluate import Evaluator, Predictor
from .models import Forecaster, JaneStreetWeightedMSELoss
from .utils import ensure_dir


class EarlyStopping:
    """Tracks the best score; `update` returns True when it improved."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        if score > self.best:
            self.best, self.bad_epochs = score, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


@dataclass(frozen=True)
class TrainResult:
    best_val_r2: float
    output_dir: str
    epochs_run: int


class Trainer:
    def __init__(
        self,
        forecaster: Forecaster,
        datamodule: JaneStreetDataModule,
        train_cfg: TrainConfig,
        data_cfg: DataConfig,
    ):
        if not forecaster.is_trainable:
            raise ValueError(f"{type(forecaster).__name__} is not trainable")
        self.forecaster = forecaster
        self.datamodule = datamodule
        self.cfg = train_cfg
        self.predictor = Predictor(data_cfg)
        self.evaluator = Evaluator(data_cfg)
        self.loss_fn = JaneStreetWeightedMSELoss()

    def _validate(self, val_loader) -> float:
        table = self.predictor.run(
            self.forecaster, val_loader, max_batches=self.cfg.max_val_batches, desc="Evaluating")
        return self.evaluator.score(table).r2

    def _train_epoch(self, epoch: int, loader, optimizer, params) -> tuple:
        """One pass over the train loader; returns (avg_loss, n_batches)."""
        self.forecaster.set_training(True)
        device = self.forecaster.device
        total_loss, n_batches = 0.0, 0
        batches = itertools.islice(loader, self.cfg.max_train_batches)
        progress = tqdm(batches, desc=f"Epoch {epoch}", total=self.cfg.max_train_batches)
        for step, batch in enumerate(progress, start=1):
            pred = self.forecaster.predict(batch)
            loss = self.loss_fn(pred, batch.target.to(device).float(), batch.weight.to(device).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.cfg.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.6f}", avg=f"{total_loss / step:.6f}")
            if step % self.cfg.log_every == 0:
                progress.write(f"[epoch {epoch}] step {step} (streamed) loss={total_loss / step:.6f}")
        return total_loss / max(1, n_batches), n_batches

    def fit(self) -> TrainResult:
        cfg = self.cfg
        params = self.forecaster.trainable_parameters()
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
        train_loader = self.datamodule.loader("train", cfg.batch_size)
        val_loader = self.datamodule.loader("val", cfg.eval_batch_size)
        stopper = EarlyStopping(cfg.patience)
        ensure_dir(cfg.output_dir)

        epochs_run = 0
        for epoch in range(1, cfg.epochs + 1):
            epochs_run = epoch
            avg_loss, n_batches = self._train_epoch(epoch, train_loader, optimizer, params)
            if not cfg.validate:
                # no validation: no best-checkpoint selection, keep the latest epoch
                print(f"[epoch {epoch}/{cfg.epochs}] batches={n_batches} train_weighted_loss={avg_loss:.6f}", flush=True)
                self.forecaster.save(cfg.output_dir)
                continue
            val_r2 = self._validate(val_loader)
            scheduler.step(val_r2)
            print(f"[epoch {epoch}/{cfg.epochs}] batches={n_batches} "
                  f"train_weighted_loss={avg_loss:.6f} val_weighted_R2={val_r2:.6f}", flush=True)

            if stopper.update(val_r2):
                self.forecaster.save(cfg.output_dir)
                print(f"  -> new best (val_R2={val_r2:.6f}), saved to {cfg.output_dir}")
            elif stopper.should_stop:
                print(f"Early stopping at epoch {epoch} (no improvement for {cfg.patience} epochs).")
                break

        if not cfg.validate:
            print(f"Training complete (validation off), saved to {cfg.output_dir}")
            return TrainResult(best_val_r2=float("nan"), output_dir=cfg.output_dir, epochs_run=epochs_run)
        print(f"Training complete. Best validation weighted R^2: {stopper.best:.6f}")
        return TrainResult(best_val_r2=stopper.best, output_dir=cfg.output_dir, epochs_run=epochs_run)
