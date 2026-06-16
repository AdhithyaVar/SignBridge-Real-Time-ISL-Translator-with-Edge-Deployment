"""
train.py
--------
CLI entry-point for the SignBridge CNN-LSTM training pipeline.

What this script does
---------------------
1.  Parses command-line arguments
2.  Loads dataset.yaml, model.yaml, training.yaml
3.  Detects GPU / CPU and logs hardware info
4.  Builds DataLoaders (runs preprocessing if splits missing)
5.  Builds SignBridgeCNNLSTM model from model.yaml
6.  Builds Trainer with optimizer, scheduler, callbacks
7.  Optionally resumes from a checkpoint
8.  Runs training loop (with early stopping)
9.  Prints final best accuracy and model location

Usage
-----
    # Standard training run
    python scripts/train.py

    # Force rebuild of dataset splits before training
    python scripts/train.py --rebuild_data

    # Resume from a saved checkpoint
    python scripts/train.py --resume models/checkpoints/epoch_050_val0.8500.pth

    # Override number of epochs
    python scripts/train.py --epochs 200

    # Force CPU even if GPU is available
    python scripts/train.py --device cpu

    # Debug run (2 epochs, small batch)
    python scripts/train.py --debug
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.device import get_device_from_config
from src.models.model_factory import build_model, print_model_summary
from src.preprocessing.dataset_builder import build_dataloaders
from src.training.trainer import build_trainer
from src.training.callbacks import build_callbacks

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — CNN-LSTM Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train.py
  python scripts/train.py --rebuild_data
  python scripts/train.py --resume models/checkpoints/epoch_050_val0.8500.pth
  python scripts/train.py --epochs 200 --device cpu
  python scripts/train.py --debug
        """,
    )
    parser.add_argument(
        "--rebuild_data",
        action="store_true",
        help="Force re-run of preprocessing pipeline before training.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pth checkpoint to resume training from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max training epochs from training.yaml.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override training batch size from training.yaml.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Force a specific device (default: auto from training.yaml).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override initial learning rate.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 2 epochs, batch_size=8, verbose logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_splits_exist(dataset_cfg: dict) -> bool:
    """Return True if preprocessed splits already exist."""
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    return (splits_dir / "X_train.npy").exists()


def print_training_plan(
    dataset_cfg:  dict,
    training_cfg: dict,
    device,
) -> None:
    """Print a human-readable summary of the training configuration."""
    train_cfg = training_cfg.get("training",  {})
    opt_cfg   = training_cfg.get("optimizer", {})

    print("\n" + "=" * 58)
    print("  SignBridge — Training Plan")
    print("=" * 58)
    print(f"  Device         : {device}")
    print(f"  Classes        : {dataset_cfg['num_classes']}")
    print(f"  Sequence len   : {dataset_cfg['recording']['sequence_length']}")
    print(f"  Feature dim    : {dataset_cfg['features']['two_hand_features']}")
    print(f"  Max epochs     : {train_cfg.get('epochs', 150)}")
    print(f"  Batch size     : {train_cfg.get('batch_size', 64)}")
    print(f"  Learning rate  : {opt_cfg.get('lr', 1e-3)}")
    print(f"  Weight decay   : {opt_cfg.get('weight_decay', 1e-4)}")
    print(f"  Best model     : models/best/best_model.pth")
    print(f"  Logs           : logs/training_log.csv")
    print(f"  TensorBoard    : logs/tensorboard/")
    print("=" * 58 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    get_logger("signbridge", log_level=log_level)

    # ── Load configs ──────────────────────────────────────────────────
    dataset_cfg  = load_config("dataset")
    model_cfg    = load_config("model",    resolve_paths=False)
    training_cfg = load_config("training", resolve_paths=False)

    # ── Apply CLI overrides ───────────────────────────────────────────
    if args.debug:
        training_cfg["training"]["epochs"]     = 2
        training_cfg["training"]["batch_size"] = 8
        training_cfg["early_stopping"]["patience"] = 2
        logger.info("Debug mode: 2 epochs, batch=8, patience=2")

    if args.epochs is not None:
        training_cfg["training"]["epochs"] = args.epochs
        logger.info(f"Override: epochs = {args.epochs}")

    if args.batch_size is not None:
        training_cfg["training"]["batch_size"] = args.batch_size
        logger.info(f"Override: batch_size = {args.batch_size}")

    if args.lr is not None:
        training_cfg["optimizer"]["lr"] = args.lr
        logger.info(f"Override: lr = {args.lr}")

    if args.device is not None:
        training_cfg["hardware"]["device"] = args.device
        logger.info(f"Override: device = {args.device}")

    # ── Device ───────────────────────────────────────────────────────
    device = get_device_from_config(training_cfg)
    print_training_plan(dataset_cfg, training_cfg, device)

    # ── Dataset / DataLoaders ─────────────────────────────────────────
    splits_exist = check_splits_exist(dataset_cfg)
    if not splits_exist or args.rebuild_data:
        if not splits_exist:
            logger.info("No preprocessed splits found — running preprocessing...")
        else:
            logger.info("Rebuilding dataset splits (--rebuild_data)...")

        # Run preprocessing inline
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/preprocess.py", "--force"],
            cwd=str(_PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error("Preprocessing failed. Run  python scripts/preprocess.py  to debug.")
            sys.exit(1)

    logger.info("Building DataLoaders...")
    train_loader, val_loader, _, scaler = build_dataloaders(
        dataset_cfg=  dataset_cfg,
        training_cfg= training_cfg,
        force_rebuild=False,
    )
    logger.info(
        f"DataLoaders ready | "
        f"train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────
    logger.info("Building model...")
    model = build_model(model_cfg, dataset_cfg, device=device)
    print_model_summary(model)

    # ── Optimizer (needed for callbacks before Trainer builds its own) ─
    import torch
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=           float(training_cfg["optimizer"].get("lr",           1e-3)),
        weight_decay= float(training_cfg["optimizer"].get("weight_decay", 1e-4)),
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = build_callbacks(training_cfg, optimizer)
    logger.info(
        f"Callbacks: {[type(cb).__name__ for cb in callbacks]}"
    )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = build_trainer(
        model=        model,
        train_loader= train_loader,
        val_loader=   val_loader,
        training_cfg= training_cfg,
        dataset_cfg=  dataset_cfg,
        device=       device,
        callbacks=    callbacks,
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    if args.resume:
        resume_path = _PROJECT_ROOT / args.resume
        if not resume_path.exists():
            logger.error(f"Checkpoint not found: {resume_path}")
            sys.exit(1)
        trainer.resume_from_checkpoint(resume_path)

    # ── Train ─────────────────────────────────────────────────────────
    t_start = time.time()
    history = trainer.train()
    t_total = time.time() - t_start

    # ── Final report ──────────────────────────────────────────────────
    best_acc   = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    best_epoch = history["val_accuracy"].index(best_acc) + 1 \
                 if history["val_accuracy"] else 0

    print("\n" + "=" * 58)
    print("  Training Complete")
    print("=" * 58)
    print(f"  Total time        : {t_total/60:.1f} min")
    print(f"  Epochs completed  : {len(history['epoch'])}")
    print(f"  Best val_accuracy : {best_acc:.4f}")
    print(f"  Best epoch        : {best_epoch}")
    print(f"  Best model saved  : models/best/best_model.pth")
    print(f"  Training log      : logs/training_log.csv")
    print(f"  TensorBoard       : tensorboard --logdir logs/tensorboard")
    print("=" * 58 + "\n")

    if best_acc >= 0.90:
        print("  ✓  Target accuracy (>90%) ACHIEVED!\n")
    else:
        gap = 0.90 - best_acc
        print(f"  ⚠  Target accuracy not yet reached ({best_acc:.4f}). "
              f"Gap: {gap:.4f}")
        print(f"     Consider: more data, longer training, or --epochs 200\n")


if __name__ == "__main__":
    main()
