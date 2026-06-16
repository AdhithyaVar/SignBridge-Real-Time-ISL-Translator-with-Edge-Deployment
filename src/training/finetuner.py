"""
finetuner.py
------------
Progressive unfreezing fine-tuner for the SignBridge CNN-LSTM model.

Progressive unfreezing strategy
---------------------------------
Training from scratch: all 4.06M params trained simultaneously.
Fine-tuning risk: adapting ALL layers to new data causes catastrophic
forgetting of the original signer's learned features.

Solution: progressive unfreezing
  Stage 1 → Only classifier head trainable (168K params)
  Stage 2 → Attention also unfrozen (+1.12M params)
  Stage 3 → LSTM also unfrozen (+2.63M params)
  Stage 4 → CNN also unfrozen (full 4.06M params, very low LR)

Each stage uses a lower learning rate than the previous — the lower
layers retain more of their learned representation.

Hard example mining
--------------------
The S/M/N classes are consistently confused (1 error per run observed).
During fine-tuning their loss contribution is weighted 2.5×, forcing
the model to spend more gradient budget distinguishing them.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.models.cnn_lstm import SignBridgeCNNLSTM
from src.utils.class_labels import CLASS_LABELS, CLASS_TO_IDX
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Layer group names → parameter filter
# ---------------------------------------------------------------------------
_LAYER_GROUPS = {
    "cnn":        lambda m: m.cnn,
    "lstm":       lambda m: m.lstm,
    "attention":  lambda m: m.attention,
    "classifier": lambda m: m.classifier,
    "input_norm": lambda m: m.input_norm,
}


def freeze_all(model: SignBridgeCNNLSTM) -> None:
    """Freeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = False
    logger.info("All model parameters frozen.")


def unfreeze_layers(
    model:  SignBridgeCNNLSTM,
    layers: list[str],
) -> int:
    """
    Unfreeze the named layer groups.

    Parameters
    ----------
    model  : SignBridgeCNNLSTM
    layers : list of layer group names (keys of _LAYER_GROUPS)

    Returns
    -------
    int — number of newly trainable parameters
    """
    newly_trainable = 0
    for name in layers:
        if name not in _LAYER_GROUPS:
            logger.warning(f"Unknown layer group '{name}' — skipping.")
            continue
        module = _LAYER_GROUPS[name](model)
        for param in module.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                newly_trainable += param.numel()

    trainable_total = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(
        f"Unfrozen: {layers} | "
        f"Newly trainable: {newly_trainable:,} | "
        f"Total trainable: {trainable_total:,}"
    )
    return newly_trainable


def get_trainable_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Weighted loss for hard example mining
# ---------------------------------------------------------------------------

def build_weighted_criterion(
    hard_classes:      list[str],
    hard_class_weight: float,
    num_classes:       int,
    label_smoothing:   float,
    device:            torch.device,
) -> nn.CrossEntropyLoss:
    """
    Build CrossEntropyLoss with per-class weights for hard example mining.

    Parameters
    ----------
    hard_classes      : list of class names to up-weight (e.g. ['S','M','N'])
    hard_class_weight : weight multiplier for hard classes
    num_classes       : total number of classes
    label_smoothing   : smoothing factor
    device            : target device for weight tensor

    Returns
    -------
    nn.CrossEntropyLoss with weight tensor
    """
    weights = torch.ones(num_classes, dtype=torch.float32)
    for cls in hard_classes:
        if cls in CLASS_TO_IDX:
            weights[CLASS_TO_IDX[cls]] = hard_class_weight
            logger.info(
                f"Hard mining: class '{cls}' (idx {CLASS_TO_IDX[cls]}) "
                f"weight = {hard_class_weight}"
            )

    return nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=label_smoothing,
    )


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

class FineTuner:
    """
    Runs progressive unfreezing fine-tuning on a loaded checkpoint.

    Parameters
    ----------
    model        : SignBridgeCNNLSTM loaded from best_model.pth
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    finetune_cfg : parsed finetune.yaml config
    device       : torch.device
    """

    def __init__(
        self,
        model:        SignBridgeCNNLSTM,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        finetune_cfg: dict,
        device:       torch.device,
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = finetune_cfg
        self.device       = device

        self.stages_cfg   = finetune_cfg["stages"]
        self.opt_cfg      = finetune_cfg.get("optimizer",  {})
        self.loss_cfg     = finetune_cfg.get("loss",       {})
        self.hard_cfg     = finetune_cfg.get("hard_mining",{})
        self.ckpt_cfg     = finetune_cfg.get("checkpoints",{})
        self.log_cfg      = finetune_cfg.get("logging",    {})

        self.use_amp      = (
            finetune_cfg.get("hardware", {}).get("mixed_precision", True)
            and device.type == "cuda"
        )
        self._scaler = torch.amp.GradScaler(
            device=device.type, enabled=self.use_amp
        )

        # Checkpoint dirs
        from src.utils.config_loader import get_project_root
        root = get_project_root()
        self.save_dir = root / self.ckpt_cfg.get("save_dir", "models/finetune_checkpoints")
        self.best_dir = root / self.ckpt_cfg.get("best_dir", "models/finetuned")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        log_dir = root / self.log_cfg.get("log_dir", "logs/finetune")
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = log_dir / self.log_cfg.get("csv_filename", "finetune_log.csv")
        self._csv_rows: list[dict] = []

        # Best model tracking
        self._global_best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_all_stages(self) -> dict[str, list]:
        """
        Execute all progressive unfreezing stages in order.

        Returns
        -------
        dict with full training history across all stages.
        """
        history = {
            "stage": [], "epoch": [], "train_loss": [],
            "val_loss": [], "train_accuracy": [], "val_accuracy": [],
        }

        logger.info(
            f"\n{'='*58}\n"
            f"  Starting Fine-Tuning (Progressive Unfreezing)\n"
            f"  Device   : {self.device}\n"
            f"  AMP      : {'enabled' if self.use_amp else 'disabled'}\n"
            f"  Stages   : {len(self.stages_cfg)}\n"
            f"{'='*58}"
        )

        # Freeze all parameters first
        freeze_all(self.model)

        for stage_key, stage_cfg in self.stages_cfg.items():
            stage_history = self._run_stage(stage_key, stage_cfg)
            for k, v in stage_history.items():
                if k in history:
                    history[k].extend(v)

        # Save CSV log
        self._save_csv()

        logger.info(
            f"\n{'='*58}\n"
            f"  Fine-Tuning Complete\n"
            f"  Best val accuracy : {self._global_best_val_acc:.4f}\n"
            f"  Best model saved  : {self.best_dir}/finetuned_model.pth\n"
            f"{'='*58}\n"
        )
        return history

    def run_single_stage(self, stage_key: str) -> dict:
        """Run only one specific stage (for resuming or experimenting)."""
        if stage_key not in self.stages_cfg:
            raise ValueError(
                f"Stage '{stage_key}' not found. "
                f"Available: {list(self.stages_cfg.keys())}"
            )
        return self._run_stage(stage_key, self.stages_cfg[stage_key])

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def _run_stage(self, stage_key: str, stage_cfg: dict) -> dict:
        """Execute one fine-tuning stage."""
        name        = stage_cfg.get("name", stage_key)
        epochs      = int(stage_cfg.get("epochs", 20))
        lr          = float(stage_cfg.get("lr", 1e-4))
        unfreeze    = stage_cfg.get("unfreeze", ["classifier"])
        batch_size  = int(stage_cfg.get("batch_size", 32))

        logger.info(
            f"\n{'─'*58}\n"
            f"  Stage: {stage_key} — {name}\n"
            f"  Unfreezing : {unfreeze}\n"
            f"  LR         : {lr}\n"
            f"  Epochs     : {epochs}\n"
            f"{'─'*58}"
        )

        # Unfreeze layers for this stage
        unfreeze_layers(self.model, unfreeze)
        trainable = get_trainable_count(self.model)
        logger.info(f"  Trainable params: {trainable:,} / 4,063,638")

        # Build optimizer (only trainable params)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=float(self.opt_cfg.get("weight_decay", 5e-4)),
        )

        # Cosine LR scheduler within stage
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1
        )

        # Loss with hard example mining
        hard_mining_enabled = self.hard_cfg.get("enabled", True)
        if hard_mining_enabled:
            criterion = build_weighted_criterion(
                hard_classes=      self.hard_cfg.get("hard_classes", ["S","M","N"]),
                hard_class_weight= float(self.hard_cfg.get("hard_class_weight", 2.5)),
                num_classes=       len(CLASS_LABELS),
                label_smoothing=   float(self.loss_cfg.get("label_smoothing", 0.15)),
                device=            self.device,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                label_smoothing=float(self.loss_cfg.get("label_smoothing", 0.15))
            )

        # Early stopping for this stage
        es_cfg      = self.cfg.get("early_stopping", {})
        es_patience = int(es_cfg.get("patience", 10))
        es_best     = 0.0
        es_counter  = 0

        history = {
            "stage": [], "epoch": [], "train_loss": [],
            "val_loss": [], "train_accuracy": [], "val_accuracy": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._train_epoch(optimizer, criterion)
            val_loss,   val_acc   = self._val_epoch(criterion)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                f"  [{stage_key}] Ep {epoch:03d}/{epochs} | "
                f"loss {train_loss:.4f}→{val_loss:.4f} | "
                f"acc {train_acc:.4f}→{val_acc:.4f} | "
                f"lr {current_lr:.2e}"
            )

            # Record history
            history["stage"].append(stage_key)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            # CSV row
            self._csv_rows.append({
                "stage": stage_key, "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "train_accuracy": f"{train_acc:.6f}",
                "val_accuracy": f"{val_acc:.6f}",
                "lr": f"{current_lr:.2e}",
            })

            # Save best model (global across all stages)
            if val_acc > self._global_best_val_acc:
                self._global_best_val_acc = val_acc
                self._save_best(epoch, val_acc, stage_key)

            # Stage-level early stopping
            if val_acc > es_best + float(es_cfg.get("min_delta", 0.001)):
                es_best    = val_acc
                es_counter = 0
            else:
                es_counter += 1
                if es_counter >= es_patience:
                    logger.info(
                        f"  Early stopping at epoch {epoch} "
                        f"(stage {stage_key})"
                    )
                    break

        logger.info(
            f"  Stage {stage_key} done | "
            f"best val_acc = {max(history['val_accuracy']):.4f}"
        )
        return history

    # ------------------------------------------------------------------
    # Train / Val epochs
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for X, y in self.train_loader:
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                logits = self.model(X)
                loss   = criterion(logits, y)

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                max_norm=1.0,
            )
            self._scaler.step(optimizer)
            self._scaler.update()

            bs          = X.size(0)
            total_loss += loss.item() * bs
            correct    += (logits.detach().argmax(-1) == y).sum().item()
            total      += bs

        return total_loss / max(total, 1), correct / max(total, 1)

    def _val_epoch(
        self,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    logits = self.model(X)
                    loss   = criterion(logits, y)

                bs          = X.size(0)
                total_loss += loss.item() * bs
                correct    += (logits.argmax(-1) == y).sum().item()
                total      += bs

        return total_loss / max(total, 1), correct / max(total, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_best(
        self,
        epoch:     int,
        val_acc:   float,
        stage_key: str,
    ) -> None:
        checkpoint = {
            "epoch":             epoch,
            "stage":             stage_key,
            "model_state_dict":  self.model.state_dict(),
            "val_accuracy":      val_acc,
        }
        best_path = self.best_dir / self.ckpt_cfg.get(
            "best_filename", "finetuned_model.pth"
        )
        torch.save(checkpoint, best_path)
        logger.info(
            f"  ★ New best model | "
            f"stage={stage_key} epoch={epoch} "
            f"val_acc={val_acc:.4f} → {best_path.name}"
        )

    def _save_csv(self) -> None:
        if not self._csv_rows:
            return
        with open(self._csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(self._csv_rows)
        logger.info(f"Fine-tune log saved → {self._csv_path}")
