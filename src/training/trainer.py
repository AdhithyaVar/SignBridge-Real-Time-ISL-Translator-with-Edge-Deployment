"""
trainer.py
----------
Core training loop for the SignBridge CNN-LSTM model.

Features
--------
* Automatic GPU / CPU selection (AMP on GPU, plain float32 on CPU)
* Mixed precision training (torch.amp) on CUDA for 2× speed
* Per-epoch train + validation passes with tqdm progress bars
* Top-1 and Top-3 accuracy computed every epoch
* Gradient clipping (prevent exploding gradients in LSTM)
* Full callback chain invoked after every epoch
* Clean training history dict returned at end
* Resumes from checkpoint if provided
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.scheduler import WarmupCosineScheduler, EarlyStopping
from src.training.callbacks import Callback
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training lifecycle for SignBridgeCNNLSTM.

    Parameters
    ----------
    model          : nn.Module
    train_loader   : DataLoader
    val_loader     : DataLoader
    optimizer      : torch optimizer
    scheduler      : WarmupCosineScheduler
    criterion      : loss function (CrossEntropyLoss with label smoothing)
    early_stopping : EarlyStopping
    callbacks      : list of Callback objects
    device         : torch.device
    max_epochs     : int
    grad_clip      : float — max gradient norm for clipping
    use_amp        : bool  — enable mixed precision on GPU
    log_every_n    : int   — print progress every N batches
    """

    def __init__(
        self,
        model:          nn.Module,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        optimizer:      optim.Optimizer,
        scheduler:      WarmupCosineScheduler,
        criterion:      nn.Module,
        early_stopping: EarlyStopping,
        callbacks:      list[Callback],
        device:         torch.device,
        max_epochs:     int   = 150,
        grad_clip:      float = 1.0,
        use_amp:        bool  = True,
        log_every_n:    int   = 10,
    ) -> None:
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.criterion      = criterion
        self.early_stopping = early_stopping
        self.callbacks      = callbacks
        self.device         = device
        self.max_epochs     = max_epochs
        self.grad_clip      = grad_clip
        self.log_every_n    = log_every_n

        # AMP only makes sense on CUDA
        self.use_amp = use_amp and (device.type == "cuda")
        if use_amp and device.type != "cuda":
            logger.info("AMP requested but device is CPU — disabling AMP.")

        # GradScaler for AMP
        self._scaler = torch.amp.GradScaler(
            device=device.type,
            enabled=self.use_amp,
        )

        # Training state
        self._start_epoch = 1
        self.history: dict[str, list] = {
            "epoch":          [],
            "train_loss":     [],
            "val_loss":       [],
            "train_accuracy": [],
            "val_accuracy":   [],
            "top3_accuracy":  [],
            "lr":             [],
            "epoch_time_s":   [],
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> dict[str, list]:
        """
        Run the full training loop.

        Returns
        -------
        dict — training history with one value per epoch for each metric.
        """
        logger.info(
            f"\n{'='*58}\n"
            f"  Starting Training\n"
            f"  Device     : {self.device}\n"
            f"  AMP        : {'enabled' if self.use_amp else 'disabled'}\n"
            f"  Max epochs : {self.max_epochs}\n"
            f"  Grad clip  : {self.grad_clip}\n"
            f"{'='*58}"
        )

        best_val_acc = 0.0

        for epoch in range(self._start_epoch, self.max_epochs + 1):
            epoch_start = time.time()

            # ── Training pass ────────────────────────────────────────
            train_loss, train_acc = self._train_epoch(epoch)

            # ── Validation pass ───────────────────────────────────────
            val_loss, val_acc, top3_acc = self._val_epoch(epoch)

            # ── Scheduler step ────────────────────────────────────────
            current_lr = self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # ── Collect metrics ───────────────────────────────────────
            metrics = {
                "train_loss":     train_loss,
                "val_loss":       val_loss,
                "train_accuracy": train_acc,
                "val_accuracy":   val_acc,
                "top3_accuracy":  top3_acc,
                "lr":             current_lr,
                "epoch_time_s":   epoch_time,
                "overfit_gap":    val_loss - train_loss,
            }

            # ── Log epoch summary ─────────────────────────────────────
            self._log_epoch(epoch, metrics)

            # ── Update history ────────────────────────────────────────
            self.history["epoch"].append(epoch)
            for k in ("train_loss", "val_loss", "train_accuracy",
                      "val_accuracy", "top3_accuracy", "lr", "epoch_time_s"):
                self.history[k].append(metrics[k])

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # ── Run callbacks ─────────────────────────────────────────
            for cb in self.callbacks:
                cb.on_epoch_end(epoch, metrics, self.model)

            # ── Early stopping check ──────────────────────────────────
            stop = self.early_stopping(val_acc, self.model, epoch)
            if stop:
                logger.info(
                    f"\nTraining stopped early at epoch {epoch}. "
                    f"Best val_accuracy: {self.early_stopping.best_score:.4f} "
                    f"at epoch {self.early_stopping.best_epoch}."
                )
                break

        # ── Training complete ─────────────────────────────────────────
        logger.info(
            f"\n{'='*58}\n"
            f"  Training Complete\n"
            f"  Best val_accuracy : {self.early_stopping.best_score:.4f}\n"
            f"  Best epoch        : {self.early_stopping.best_epoch}\n"
            f"{'='*58}\n"
        )

        for cb in self.callbacks:
            cb.on_training_end(
                best_epoch=  self.early_stopping.best_epoch,
                best_metric= self.early_stopping.best_score or 0.0,
            )

        return self.history

    def resume_from_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load a checkpoint to resume training.
        Must be called before train().
        """
        from src.models.model_factory import load_checkpoint
        self.model, start_epoch, metrics = load_checkpoint(
            self.model, checkpoint_path,
            optimizer=self.optimizer,
            device=self.device,
        )
        self._start_epoch = start_epoch + 1
        logger.info(f"Resuming from epoch {self._start_epoch}")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> tuple[float, float]:
        """
        One full pass over the training DataLoader.

        Returns
        -------
        (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [TRAIN]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward + loss (with optional AMP)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                logits = self.model(X)              # (batch, num_classes)
                loss   = self.criterion(logits, y)

            # Backward
            self._scaler.scale(loss).backward()

            # Gradient clipping (unscale first for AMP)
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )

            # Optimizer step
            self._scaler.step(self.optimizer)
            self._scaler.update()

            # Accumulate metrics
            batch_size  = X.size(0)
            total_loss += loss.item() * batch_size
            preds       = logits.detach().argmax(dim=-1)
            correct    += (preds == y).sum().item()
            total      += batch_size

            # Progress bar suffix
            if (batch_idx + 1) % self.log_every_n == 0:
                pbar.set_postfix(
                    loss=f"{total_loss/total:.4f}",
                    acc= f"{correct/total:.4f}",
                )

        avg_loss = total_loss / max(total, 1)
        accuracy = correct   / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    def _val_epoch(
        self,
        epoch: int,
    ) -> tuple[float, float, float]:
        """
        One full pass over the validation DataLoader.

        Returns
        -------
        (avg_loss, top1_accuracy, top3_accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_1  = 0
        correct_3  = 0
        total      = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d} [VAL  ]",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for X, y in pbar:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    logits = self.model(X)
                    loss   = self.criterion(logits, y)

                batch_size  = X.size(0)
                total_loss += loss.item() * batch_size

                # Top-1 accuracy
                pred_1  = logits.argmax(dim=-1)
                correct_1 += (pred_1 == y).sum().item()

                # Top-3 accuracy
                top3    = logits.topk(min(3, logits.size(-1)), dim=-1).indices
                correct_3 += sum(
                    y[i].item() in top3[i].tolist()
                    for i in range(batch_size)
                )

                total += batch_size

        avg_loss  = total_loss / max(total, 1)
        top1_acc  = correct_1  / max(total, 1)
        top3_acc  = correct_3  / max(total, 1)
        return avg_loss, top1_acc, top3_acc

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(self, epoch: int, metrics: dict) -> None:
        """Print a clean one-line epoch summary."""
        logger.info(
            f"Epoch {epoch:03d}/{self.max_epochs} | "
            f"loss {metrics['train_loss']:.4f}→{metrics['val_loss']:.4f} | "
            f"acc  {metrics['train_accuracy']:.4f}→{metrics['val_accuracy']:.4f} | "
            f"top3 {metrics['top3_accuracy']:.4f} | "
            f"lr {metrics['lr']:.2e} | "
            f"{metrics['epoch_time_s']:.1f}s"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_trainer(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    training_cfg: dict,
    dataset_cfg:  dict,
    device:       torch.device,
    callbacks:    list[Callback],
) -> Trainer:
    """
    Build a fully configured Trainer from parsed YAML configs.

    Parameters
    ----------
    model, train_loader, val_loader : from Phases 3 & 4
    training_cfg : parsed training.yaml
    dataset_cfg  : parsed dataset.yaml
    device       : from device.get_device()
    callbacks    : from callbacks.build_callbacks()

    Returns
    -------
    Trainer (ready to call .train())
    """
    from src.training.scheduler import build_scheduler, build_early_stopping

    opt_cfg   = training_cfg.get("optimizer",  {})
    train_cfg = training_cfg.get("training",   {})
    hw_cfg    = training_cfg.get("hardware",   {})
    loss_cfg  = training_cfg.get("loss",       {})
    reg_cfg   = training_cfg.get("", {})

    # ── Optimizer (AdamW) ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=           float(opt_cfg.get("lr",            1e-3)),
        betas=  tuple(      opt_cfg.get("betas",         [0.9, 0.999])),
        eps=          float(opt_cfg.get("eps",           1e-8)),
        weight_decay= float(opt_cfg.get("weight_decay",  1e-4)),
    )

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler = build_scheduler(optimizer, training_cfg)

    # ── Loss (CrossEntropy + label smoothing) ─────────────────────────
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.1))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )
    logger.info(f"Loss: CrossEntropyLoss(label_smoothing={label_smoothing})")

    # ── Early stopping ────────────────────────────────────────────────
    early_stopping = build_early_stopping(training_cfg)

    # ── AMP / mixed precision ─────────────────────────────────────────
    use_amp = bool(hw_cfg.get("mixed_precision", True))

    # ── Gradient clip ─────────────────────────────────────────────────
    from src.utils.config_loader import load_config as _lc
    model_cfg   = _lc("model", resolve_paths=False)
    grad_clip   = float(
        model_cfg.get("regularization", {}).get("gradient_clip", 1.0)
    )

    return Trainer(
        model=          model,
        train_loader=   train_loader,
        val_loader=     val_loader,
        optimizer=      optimizer,
        scheduler=      scheduler,
        criterion=      criterion,
        early_stopping= early_stopping,
        callbacks=      callbacks,
        device=         device,
        max_epochs=     int(train_cfg.get("epochs",    150)),
        grad_clip=      grad_clip,
        use_amp=        use_amp,
        log_every_n=    int(training_cfg.get("logging", {}).get("log_every_n_steps", 10)),
    )
