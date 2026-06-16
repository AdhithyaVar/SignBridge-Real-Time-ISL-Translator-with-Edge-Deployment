"""
callbacks.py
------------
Training callbacks for the SignBridge training pipeline.

Callbacks are invoked by the Trainer at the end of every epoch.
Each callback receives the full metrics dict and the model, and performs
a side-effect (save file, log metric, print warning, etc).

Available callbacks
-------------------
CheckpointCallback      — saves per-epoch .pth files, keeps last N, detects best
CSVLoggerCallback       — appends epoch metrics to a CSV file
TensorBoardCallback     — writes scalars to TensorBoard event files
OverfittingMonitorCallback — warns when val_loss - train_loss > threshold
"""

from __future__ import annotations

import csv
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base callback
# ---------------------------------------------------------------------------

class Callback(ABC):
    """Abstract base class for all training callbacks."""

    @abstractmethod
    def on_epoch_end(
        self,
        epoch:   int,
        metrics: dict[str, float],
        model:   nn.Module,
    ) -> None:
        """
        Called by Trainer at the end of each epoch.

        Parameters
        ----------
        epoch   : int  — current epoch (1-indexed)
        metrics : dict — keys like 'train_loss', 'val_loss',
                        'train_accuracy', 'val_accuracy', 'lr'
        model   : nn.Module — current model state
        """

    def on_training_end(
        self,
        best_epoch:  int,
        best_metric: float,
    ) -> None:
        """Called once when training completes (or early-stops)."""


# ---------------------------------------------------------------------------
# Checkpoint callback
# ---------------------------------------------------------------------------

class CheckpointCallback(Callback):
    """
    Saves a .pth checkpoint after every N epochs and tracks the best model.

    Parameters
    ----------
    checkpoint_dir : Path — where per-epoch checkpoints go
    best_dir       : Path — where best_model.pth is saved
    save_every     : int  — save a checkpoint every N epochs (5)
    keep_last_n    : int  — delete old checkpoints, keep the N most recent (3)
    monitor        : str  — metric name to track for best model
    mode           : str  — 'max' or 'min'
    optimizer      : optimizer whose state is saved alongside model
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        best_dir:       Path,
        save_every:     int   = 5,
        keep_last_n:    int   = 3,
        monitor:        str   = "val_accuracy",
        mode:           str   = "max",
        optimizer:      torch.optim.Optimizer | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_dir       = Path(best_dir)
        self.save_every     = save_every
        self.keep_last_n    = keep_last_n
        self.monitor        = monitor
        self.mode           = mode
        self.optimizer      = optimizer

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        self._best_score: float | None = None
        self._saved_files: list[Path]  = []

    def on_epoch_end(
        self,
        epoch:   int,
        metrics: dict[str, float],
        model:   nn.Module,
    ) -> None:
        score    = metrics.get(self.monitor, 0.0)
        is_best  = self._is_best(score)

        # Save periodic checkpoint
        if epoch % self.save_every == 0:
            filename  = f"epoch_{epoch:03d}_val{metrics.get('val_accuracy', 0):.4f}.pth"
            save_path = self.checkpoint_dir / filename
            self._save(model, epoch, metrics, save_path)
            self._saved_files.append(save_path)
            self._prune_old_checkpoints()

        # Save best model
        if is_best:
            self._best_score = score
            best_path = self.best_dir / "best_model.pth"
            self._save(model, epoch, metrics, best_path)
            logger.info(
                f"  ★ New best model | epoch={epoch} | "
                f"{self.monitor}={score:.4f}"
            )

    def _is_best(self, score: float) -> bool:
        if self._best_score is None:
            return True
        if self.mode == "max":
            return score > self._best_score
        return score < self._best_score

    def _save(
        self,
        model:    nn.Module,
        epoch:    int,
        metrics:  dict,
        path:     Path,
    ) -> None:
        checkpoint = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "metrics":              metrics,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        logger.debug(f"  Checkpoint saved: {path.name}")

    def _prune_old_checkpoints(self) -> None:
        """Delete oldest checkpoints, keeping only the last N."""
        while len(self._saved_files) > self.keep_last_n:
            old = self._saved_files.pop(0)
            if old.exists():
                old.unlink()
                logger.debug(f"  Pruned old checkpoint: {old.name}")


# ---------------------------------------------------------------------------
# CSV logger callback
# ---------------------------------------------------------------------------

class CSVLoggerCallback(Callback):
    """
    Appends one row per epoch to a CSV log file.

    Columns: epoch, train_loss, val_loss, train_accuracy,
             val_accuracy, top3_accuracy, lr

    Parameters
    ----------
    log_dir  : Path — directory to write CSV file
    filename : str  — filename (default: training_log.csv)
    """

    def __init__(
        self,
        log_dir:  Path,
        filename: str = "training_log.csv",
    ) -> None:
        self.log_path = Path(log_dir) / filename
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = False

        # If file exists from a previous run, preserve it (append mode)
        if self.log_path.exists():
            self._header_written = True

    def on_epoch_end(
        self,
        epoch:   int,
        metrics: dict[str, float],
        model:   nn.Module,
    ) -> None:
        row = {"epoch": epoch, **metrics}

        with open(self.log_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in row.items()})


# ---------------------------------------------------------------------------
# TensorBoard callback
# ---------------------------------------------------------------------------

class TensorBoardCallback(Callback):
    """
    Writes training metrics to TensorBoard event files.

    Parameters
    ----------
    log_dir : Path — TensorBoard log directory (logs/tensorboard/)
    """

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None

    def _get_writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                logger.warning(
                    "TensorBoard not available. "
                    "Install with: pip install tensorboard"
                )
                self._writer = False   # Mark as unavailable
        return self._writer if self._writer is not False else None

    def on_epoch_end(
        self,
        epoch:   int,
        metrics: dict[str, float],
        model:   nn.Module,
    ) -> None:
        writer = self._get_writer()
        if writer is None:
            return

        # Scalars
        scalar_map = {
            "Loss/train":       "train_loss",
            "Loss/val":         "val_loss",
            "Accuracy/train":   "train_accuracy",
            "Accuracy/val":     "val_accuracy",
            "Accuracy/top3":    "top3_accuracy",
            "LearningRate":     "lr",
            "Overfitting/gap":  "overfit_gap",
        }
        for tb_key, metric_key in scalar_map.items():
            if metric_key in metrics:
                writer.add_scalar(tb_key, metrics[metric_key], epoch)

        writer.flush()

    def on_training_end(self, best_epoch: int, best_metric: float) -> None:
        if self._writer and self._writer is not False:
            self._writer.close()


# ---------------------------------------------------------------------------
# Overfitting monitor callback
# ---------------------------------------------------------------------------

class OverfittingMonitorCallback(Callback):
    """
    Monitors the gap between training and validation loss.
    Logs a warning when overfitting or underfitting is detected.

    Parameters
    ----------
    overfit_threshold   : float — warn if val_loss - train_loss > threshold
    underfit_threshold  : float — warn if train_accuracy < threshold at epoch 10+
    underfit_start_epoch: int   — don't check underfitting before this epoch
    """

    def __init__(
        self,
        overfit_threshold:    float = 0.05,
        underfit_threshold:   float = 0.50,
        underfit_start_epoch: int   = 10,
    ) -> None:
        self.overfit_threshold    = overfit_threshold
        self.underfit_threshold   = underfit_threshold
        self.underfit_start_epoch = underfit_start_epoch
        self._overfit_count       = 0

    def on_epoch_end(
        self,
        epoch:   int,
        metrics: dict[str, float],
        model:   nn.Module,
    ) -> None:
        train_loss = metrics.get("train_loss",     0.0)
        val_loss   = metrics.get("val_loss",       0.0)
        train_acc  = metrics.get("train_accuracy", 0.0)
        gap        = val_loss - train_loss

        # Record gap in metrics for TensorBoard
        metrics["overfit_gap"] = gap

        # Overfitting check
        if gap > self.overfit_threshold:
            self._overfit_count += 1
            if self._overfit_count >= 3:
                logger.warning(
                    f"  ⚠ Overfitting detected (epoch {epoch}): "
                    f"val_loss - train_loss = {gap:.4f} "
                    f"(threshold {self.overfit_threshold}) "
                    f"[{self._overfit_count} consecutive epochs]"
                )
        else:
            self._overfit_count = 0

        # Underfitting check (only after warm-up period)
        if epoch >= self.underfit_start_epoch:
            if train_acc < self.underfit_threshold:
                logger.warning(
                    f"  ⚠ Possible underfitting (epoch {epoch}): "
                    f"train_accuracy = {train_acc:.4f} "
                    f"(below threshold {self.underfit_threshold})"
                )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_callbacks(
    training_cfg: dict,
    optimizer:    torch.optim.Optimizer,
) -> list[Callback]:
    """
    Build the default callback stack from training.yaml config.

    Returns
    -------
    list[Callback]
    """
    from pathlib import Path
    from src.utils.config_loader import get_project_root

    root = get_project_root()

    ckpt_cfg = training_cfg.get("checkpoints", {})
    log_cfg  = training_cfg.get("logging",     {})
    diag_cfg = training_cfg.get("diagnostics", {})

    callbacks: list[Callback] = []

    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        checkpoint_dir=root / ckpt_cfg.get("save_dir", "models/checkpoints"),
        best_dir=      root / ckpt_cfg.get("best_dir", "models/best"),
        save_every=    int(ckpt_cfg.get("save_every_n_epochs", 5)),
        keep_last_n=   int(ckpt_cfg.get("keep_last_n", 3)),
        monitor=            ckpt_cfg.get("monitor", "val_accuracy"),
        mode=               ckpt_cfg.get("mode",    "max"),
        optimizer=     optimizer,
    ))

    # CSV logger
    if log_cfg.get("csv_log", True):
        callbacks.append(CSVLoggerCallback(
            log_dir=  root / log_cfg.get("log_dir", "logs"),
            filename= log_cfg.get("csv_filename", "training_log.csv"),
        ))

    # TensorBoard
    if log_cfg.get("tensorboard", True):
        callbacks.append(TensorBoardCallback(
            log_dir=root / log_cfg.get("log_dir", "logs") / "tensorboard",
        ))

    # Overfitting monitor
    callbacks.append(OverfittingMonitorCallback(
        overfit_threshold=  float(diag_cfg.get("overfitting_threshold",  0.05)),
        underfit_threshold= float(diag_cfg.get("underfitting_threshold", 0.50)),
    ))

    return callbacks
