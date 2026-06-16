"""
finetune.py
-----------
CLI entry-point for SignBridge model fine-tuning.

What this script does
---------------------
1.  Loads finetune.yaml, dataset.yaml, model.yaml
2.  Loads best_model.pth as the starting checkpoint
3.  Optionally rebuilds the dataset with stronger augmentation
4.  Runs 4-stage progressive unfreezing fine-tuning
5.  Saves the best fine-tuned model to models/finetuned/finetuned_model.pth
6.  Evaluates the fine-tuned model vs the original on test set
7.  Re-exports to ONNX FP32 + INT8 using the fine-tuned weights

Usage
-----
    # Full 4-stage fine-tuning (all stages, ~30-45 min CPU)
    python scripts/finetune.py

    # Fine-tune from new signer data only (after adding signers)
    python scripts/finetune.py --rebuild_data

    # Run only one specific stage
    python scripts/finetune.py --stage stage1

    # Quick test run (5 epochs per stage)
    python scripts/finetune.py --epochs_override 5

    # After fine-tuning, export new ONNX model
    python scripts/finetune.py --export_after

    # Use the fine-tuned model for the demo
    python scripts/run_demo.py --checkpoint models/finetuned/finetuned_model.pth
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.device import get_device_from_config
from src.utils.class_labels import CLASS_LABELS, CLASS_TO_IDX, NUM_CLASSES
from src.models.model_factory import build_model, load_checkpoint, print_model_summary
from src.preprocessing.dataset_builder import build_dataloaders
from src.preprocessing.augmentor import build_augmentor
from src.training.finetuner import FineTuner, freeze_all

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — Progressive Unfreezing Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/finetune.py                          # Full 4-stage
  python scripts/finetune.py --stage stage1           # Single stage
  python scripts/finetune.py --rebuild_data           # Fresh data splits
  python scripts/finetune.py --epochs_override 5      # Quick test
  python scripts/finetune.py --export_after           # Export ONNX after
  python scripts/finetune.py --skip_stages stage3 stage4  # Skip stages
        """,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["stage1", "stage2", "stage3", "stage4"],
        help="Run only this stage (default: all 4 stages).",
    )
    parser.add_argument(
        "--skip_stages",
        nargs="+",
        default=[],
        choices=["stage1", "stage2", "stage3", "stage4"],
        help="Skip these stages.",
    )
    parser.add_argument(
        "--rebuild_data",
        action="store_true",
        help="Force rebuild of dataset splits (use after adding new signers).",
    )
    parser.add_argument(
        "--epochs_override",
        type=int,
        default=None,
        help="Override epochs for all stages (useful for quick testing).",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help="Path to base checkpoint (default: models/best/best_model.pth).",
    )
    parser.add_argument(
        "--export_after",
        action="store_true",
        help="Export fine-tuned model to ONNX FP32 + INT8 after fine-tuning.",
    )
    parser.add_argument(
        "--eval_after",
        action="store_true",
        default=True,
        help="Evaluate fine-tuned vs original model after fine-tuning (default: True).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset builder with enhanced augmentation for fine-tuning
# ---------------------------------------------------------------------------

def build_finetune_loaders(
    dataset_cfg:  dict,
    training_cfg: dict,
    finetune_cfg: dict,
    force_rebuild:bool,
) -> tuple:
    """
    Build DataLoaders with fine-tuning augmentation settings.
    Applies stronger augmentation and hard-class oversampling.
    """
    # Override augmentation settings with finetune.yaml values
    aug_override = finetune_cfg.get("augmentation", {})
    if aug_override:
        orig_aug = dataset_cfg.get("augmentation", {})
        orig_aug.update({
            "rotation_range":    aug_override.get("rotation_range",   20.0),
            "scale_range":       aug_override.get("scale_range",       0.20),
            "translation_range": aug_override.get("translation_range", 0.08),
            "gaussian_noise_std":aug_override.get("gaussian_noise_std",0.008),
            "per_sample_copies": aug_override.get("per_sample_copies",    6),
            "time_warp":         True,
        })
        dataset_cfg["augmentation"] = orig_aug
        logger.info("Fine-tuning augmentation settings applied.")

    # Smaller batch size for fine-tuning
    ft_batch = int(
        finetune_cfg.get("stages", {})
        .get("stage1", {}).get("batch_size", 32)
    )
    training_cfg["training"]["batch_size"]     = ft_batch
    training_cfg["training"]["val_batch_size"] = ft_batch * 2

    train_loader, val_loader, test_loader, scaler = build_dataloaders(
        dataset_cfg=  dataset_cfg,
        training_cfg= training_cfg,
        force_rebuild=force_rebuild,
    )
    return train_loader, val_loader, test_loader, scaler


# ---------------------------------------------------------------------------
# Comparison evaluator
# ---------------------------------------------------------------------------

def compare_models(
    original_path:  Path,
    finetuned_path: Path,
    test_loader,
    model_cfg:  dict,
    dataset_cfg:dict,
    device:     torch.device,
) -> None:
    """
    Evaluate both models on the test set and print a comparison table.
    """
    from src.evaluation.evaluator import ModelEvaluator

    print("\n" + "=" * 62)
    print("  Fine-Tuning Results — Model Comparison")
    print("=" * 62)

    for label, ckpt_path in [
        ("Original  (best_model.pth)    ", original_path),
        ("Fine-tuned (finetuned_model.pth)", finetuned_path),
    ]:
        if not ckpt_path.exists():
            print(f"  {label}: NOT FOUND")
            continue

        model = build_model(model_cfg, dataset_cfg, device=device)
        model, epoch, metrics = load_checkpoint(model, ckpt_path, device=device)
        evaluator = ModelEvaluator(model=model, device=device)
        result    = evaluator.evaluate(test_loader, split="test")

        # Per-class for hard classes
        hard_classes = ["S", "M", "N"]
        hard_f1 = {
            c["class_name"]: c["f1"]
            for c in result.per_class
            if c["class_name"] in hard_classes
        }

        print(f"\n  {label}")
        print(f"    Test accuracy  : {result.accuracy*100:.2f}%")
        print(f"    Top-3 accuracy : {result.top3_accuracy*100:.2f}%")
        print(f"    Macro F1       : {result.macro_f1*100:.2f}%")
        print(f"    Hard classes   : ", end="")
        for cls, f1 in hard_f1.items():
            print(f"{cls}={f1:.4f}  ", end="")
        print()

    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    get_logger("signbridge", log_level=log_level)

    root = get_project_root()

    # ── Load configs ──────────────────────────────────────────────────
    dataset_cfg  = load_config("dataset")
    model_cfg    = load_config("model",    resolve_paths=False)
    training_cfg = load_config("training", resolve_paths=False)
    finetune_cfg = load_config("finetune", resolve_paths=False)

    if args.device:
        training_cfg["hardware"]["device"] = args.device

    if args.epochs_override:
        for stage in finetune_cfg["stages"].values():
            stage["epochs"] = args.epochs_override
        logger.info(f"Override: all stages → {args.epochs_override} epochs")

    # ── Device ───────────────────────────────────────────────────────
    device = get_device_from_config(training_cfg)

    # ── Base checkpoint ───────────────────────────────────────────────
    base_path = (
        Path(args.base_checkpoint)
        if args.base_checkpoint
        else root / "models" / "best" / "best_model.pth"
    )
    if not base_path.exists():
        logger.error(
            f"Base checkpoint not found: {base_path}\n"
            "Train first: python scripts/train.py"
        )
        sys.exit(1)

    # ── Print plan ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SignBridge — Progressive Unfreezing Fine-Tuning")
    print("=" * 62)
    print(f"  Base model    : {base_path.name}")
    print(f"  Device        : {device}")
    stages_to_run = (
        [args.stage] if args.stage
        else [k for k in finetune_cfg["stages"] if k not in args.skip_stages]
    )
    for s in stages_to_run:
        cfg = finetune_cfg["stages"][s]
        print(
            f"  {s}: {cfg.get('name',''):<30} "
            f"lr={cfg['lr']}  epochs={cfg['epochs']}"
        )
    print(f"  Hard mining   : {finetune_cfg['hard_mining'].get('hard_classes')}")
    print(f"  Aug copies    : {finetune_cfg['augmentation'].get('per_sample_copies')}")
    print("=" * 62 + "\n")

    input("  Press ENTER to start fine-tuning, or Ctrl+C to cancel...\n")

    # ── DataLoaders ───────────────────────────────────────────────────
    logger.info("Building fine-tuning DataLoaders...")
    train_loader, val_loader, test_loader, scaler = build_finetune_loaders(
        dataset_cfg=  dataset_cfg,
        training_cfg= training_cfg,
        finetune_cfg= finetune_cfg,
        force_rebuild=args.rebuild_data,
    )
    logger.info(
        f"DataLoaders ready | "
        f"train={len(train_loader)} batches | "
        f"val={len(val_loader)} batches"
    )

    # ── Load base model ───────────────────────────────────────────────
    logger.info(f"Loading base model from {base_path.name}...")
    model = build_model(model_cfg, dataset_cfg, device=device)
    model, base_epoch, base_metrics = load_checkpoint(
        model, base_path, device=device
    )
    logger.info(
        f"Base model loaded | "
        f"epoch={base_epoch} | "
        f"val_acc={base_metrics.get('val_accuracy', 'N/A')}"
    )
    print_model_summary(model)

    # ── Skip stages not in run list ───────────────────────────────────
    if args.skip_stages:
        for skip in args.skip_stages:
            finetune_cfg["stages"].pop(skip, None)
        logger.info(f"Skipped stages: {args.skip_stages}")

    if args.stage:
        single_cfg = {"stages": {args.stage: finetune_cfg["stages"][args.stage]}}
        finetune_cfg["stages"] = single_cfg["stages"]

    # ── Fine-tune ─────────────────────────────────────────────────────
    finetuner = FineTuner(
        model=        model,
        train_loader= train_loader,
        val_loader=   val_loader,
        finetune_cfg= finetune_cfg,
        device=       device,
    )
    history = finetuner.run_all_stages()

    # ── Evaluation comparison ─────────────────────────────────────────
    finetuned_path = root / "models" / "finetuned" / "finetuned_model.pth"
    if args.eval_after and finetuned_path.exists():
        logger.info("Evaluating fine-tuned model vs original...")
        compare_models(
            original_path=  base_path,
            finetuned_path= finetuned_path,
            test_loader=    test_loader,
            model_cfg=      model_cfg,
            dataset_cfg=    dataset_cfg,
            device=         device,
        )

    # ── ONNX export ───────────────────────────────────────────────────
    if args.export_after:
        logger.info("Exporting fine-tuned model to ONNX...")
        import subprocess
        subprocess.run([
            sys.executable, "scripts/export_onnx.py",
            "--base_checkpoint", str(finetuned_path),
        ], cwd=str(root))

    # ── Final summary ─────────────────────────────────────────────────
    best_val = max(history.get("val_accuracy", [0.0]))
    print("\n" + "=" * 62)
    print("  Fine-Tuning Complete")
    print("=" * 62)
    print(f"  Best fine-tuned val accuracy : {best_val:.4f} ({best_val*100:.2f}%)")
    print(f"  Fine-tuned model saved       : models/finetuned/finetuned_model.pth")
    print(f"  Fine-tune log                : logs/finetune/finetune_log.csv")
    print(f"\n  To use fine-tuned model in demo:")
    print(f"    python scripts/run_demo.py --checkpoint models/finetuned/finetuned_model.pth")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
