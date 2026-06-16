"""
model_factory.py
----------------
Build, save, and load the SignBridge model from YAML configuration.

All training and inference scripts call build_model() or load_model()
rather than instantiating SignBridgeCNNLSTM directly.  This ensures
that configuration changes in model.yaml propagate everywhere
automatically without touching Python source files.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.cnn_lstm import SignBridgeCNNLSTM
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Build model from config dict
# ---------------------------------------------------------------------------

def build_model(
    model_cfg:   dict,
    dataset_cfg: dict,
    device:      torch.device | None = None,
) -> SignBridgeCNNLSTM:
    """
    Instantiate a SignBridgeCNNLSTM from parsed YAML configs.

    Parameters
    ----------
    model_cfg   : dict — parsed model.yaml
    dataset_cfg : dict — parsed dataset.yaml (for num_classes, dims)
    device      : torch.device | None
        If provided, model is moved to device before returning.

    Returns
    -------
    SignBridgeCNNLSTM on the requested device.
    """
    # ── Input dims from dataset config ───────────────────────────────────
    num_classes  = int(dataset_cfg["num_classes"])           # 26
    seq_len      = int(dataset_cfg["recording"]["sequence_length"])  # 30
    feature_dim  = int(dataset_cfg["features"]["two_hand_features"]) # 126

    # ── CNN config ────────────────────────────────────────────────────────
    cnn_cfg      = model_cfg["cnn"]
    cnn_channels = [int(l["out_channels"]) for l in cnn_cfg["conv_layers"]]
    cnn_dropouts = [float(l.get("dropout", 0.0)) for l in cnn_cfg["conv_layers"]]

    # ── LSTM config ───────────────────────────────────────────────────────
    lstm_cfg     = model_cfg["lstm"]
    lstm_hidden  = int(  lstm_cfg["hidden_size"])
    lstm_layers  = int(  lstm_cfg["num_layers"])
    lstm_dropout = float(lstm_cfg["dropout"])

    # ── Attention config ──────────────────────────────────────────────────
    attn_cfg     = model_cfg["attention"]
    attn_enabled = bool( attn_cfg["enabled"])
    attn_heads   = int(  attn_cfg["num_heads"])
    attn_dim     = int(  attn_cfg["attention_dim"])
    attn_dropout = float(attn_cfg["dropout"])

    # ── Classifier config ─────────────────────────────────────────────────
    clf_cfg     = model_cfg["classifier"]
    clf_dims    = [int(  l["dim"])               for l in clf_cfg["hidden_layers"]]
    clf_dropout = [float(l.get("dropout", 0.0)) for l in clf_cfg["hidden_layers"]]

    # Log configuration summary
    logger.info(
        f"Building model | "
        f"classes={num_classes} | "
        f"seq={seq_len} | "
        f"feat={feature_dim} | "
        f"cnn={cnn_channels} | "
        f"lstm={lstm_hidden}×{lstm_layers}{'×bidir' if lstm_cfg['bidirectional'] else ''} | "
        f"attn={'on' if attn_enabled else 'off'}"
    )

    model = SignBridgeCNNLSTM(
        num_classes=     num_classes,
        sequence_length= seq_len,
        feature_dim=     feature_dim,
        cnn_channels=    cnn_channels,
        lstm_hidden=     lstm_hidden,
        lstm_layers=     lstm_layers,
        lstm_dropout=    lstm_dropout,
        attn_heads=      attn_heads,
        attn_dim=        attn_dim,
        attn_dropout=    attn_dropout,
        clf_dims=        clf_dims,
        clf_dropout=     clf_dropout,
    )

    # Parameter count breakdown
    param_counts = model.count_parameters()
    logger.info(
        f"Parameters | "
        f"CNN={param_counts['cnn']:,} | "
        f"LSTM={param_counts['lstm']:,} | "
        f"Attn={param_counts['attention']:,} | "
        f"CLF={param_counts['classifier']:,} | "
        f"TOTAL={param_counts['total']:,}"
    )

    if device is not None:
        model = model.to(device)
        logger.info(f"Model moved to {device}")

    return model


# ---------------------------------------------------------------------------
# Save / Load checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    metrics:   dict,
    save_path: Path,
    is_best:   bool = False,
    best_dir:  Path | None = None,
) -> None:
    """
    Save a full training checkpoint.

    Saved dict contains:
      epoch, model_state_dict, optimizer_state_dict, metrics

    Parameters
    ----------
    model      : model to save
    optimizer  : optimizer to save (for resuming training)
    epoch      : current epoch number
    metrics    : dict of metric values (val_accuracy, val_loss, etc.)
    save_path  : full path for this checkpoint file
    is_best    : if True, also copy to best_dir/best_model.pth
    best_dir   : directory to copy best checkpoint into
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "metrics":             metrics,
    }
    torch.save(checkpoint, save_path)
    logger.debug(f"Checkpoint saved: {save_path.name}")

    if is_best and best_dir is not None:
        best_dir  = Path(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        best_path = best_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        logger.info(
            f"New best model saved → {best_path}  "
            f"(epoch={epoch}, "
            f"val_acc={metrics.get('val_accuracy', 0):.4f})"
        )


def load_checkpoint(
    model:      nn.Module,
    checkpoint_path: Path,
    optimizer:  torch.optim.Optimizer | None = None,
    device:     torch.device | None          = None,
) -> tuple[nn.Module, int, dict]:
    """
    Load a checkpoint into an existing model.

    Parameters
    ----------
    model            : model instance (must match saved architecture)
    checkpoint_path  : path to .pth file
    optimizer        : if provided, optimizer state is also restored
    device           : if provided, checkpoint is mapped to this device

    Returns
    -------
    model, start_epoch, metrics
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device if device is not None else torch.device("cpu")
    checkpoint   = torch.load(checkpoint_path, map_location=map_location,
                               weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch   = int(checkpoint.get("epoch", 0))
    metrics = checkpoint.get("metrics", {})

    logger.info(
        f"Loaded checkpoint: {checkpoint_path.name} | "
        f"epoch={epoch} | "
        f"val_acc={metrics.get('val_accuracy', 'N/A')}"
    )
    return model, epoch, metrics


def load_best_model(
    model_cfg:   dict,
    dataset_cfg: dict,
    best_dir:    Path,
    device:      torch.device | None = None,
) -> tuple[SignBridgeCNNLSTM, dict]:
    """
    Build the model architecture and load the best checkpoint weights.

    Parameters
    ----------
    model_cfg, dataset_cfg : YAML config dicts
    best_dir               : directory containing best_model.pth
    device                 : target device

    Returns
    -------
    model, metrics_at_best_epoch
    """
    model       = build_model(model_cfg, dataset_cfg, device=device)
    best_path   = Path(best_dir) / "best_model.pth"
    model, _, metrics = load_checkpoint(model, best_path, device=device)
    model.eval()
    return model, metrics


# ---------------------------------------------------------------------------
# Model info printer
# ---------------------------------------------------------------------------

def print_model_summary(model: SignBridgeCNNLSTM) -> None:
    """Print a formatted model summary to stdout."""
    counts = model.count_parameters()
    print("\n" + "=" * 58)
    print("  SignBridge CNN-LSTM — Model Summary")
    print("=" * 58)
    print(f"  Input          : (batch, {model.sequence_length}, {model.feature_dim})")
    print(f"  CNN output     : (batch, {model.sequence_length}, {model._cnn_out_dim})")
    print(f"  LSTM output    : (batch, {model.sequence_length}, {model._lstm_out_dim})")
    print(f"  Context vector : (batch, {model._lstm_out_dim})")
    print(f"  Output         : (batch, {model.num_classes})")
    print("-" * 58)
    print(f"  CNN params     : {counts['cnn']:>10,}")
    print(f"  LSTM params    : {counts['lstm']:>10,}")
    print(f"  Attention      : {counts['attention']:>10,}")
    print(f"  Classifier     : {counts['classifier']:>10,}")
    print(f"  TOTAL          : {counts['total']:>10,}")
    print("=" * 58 + "\n")
