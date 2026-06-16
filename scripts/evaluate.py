"""
evaluate.py
-----------
CLI entry-point for the SignBridge evaluation pipeline.

What this script does
---------------------
1.  Loads dataset.yaml, model.yaml, training.yaml
2.  Detects best checkpoint at models/best/best_model.pth
3.  Loads test and val DataLoaders from data/splits/
4.  Runs full inference on both splits
5.  Computes accuracy, top-3 accuracy, F1, precision, recall per class
6.  Generates:
      reports/confusion_matrix.png
      reports/per_class_f1.png
      reports/evaluation_summary.json
      reports/evaluation_report.html
7.  Prints a rich terminal summary

Usage
-----
    python scripts/evaluate.py

    # Evaluate on val set only (faster)
    python scripts/evaluate.py --split val

    # Print per-class table to terminal
    python scripts/evaluate.py --verbose

    # Use a specific checkpoint instead of best_model.pth
    python scripts/evaluate.py --checkpoint models/checkpoints/epoch_080_val0.9200.pth
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.device import get_device_from_config
from src.evaluation.evaluator import (
    ModelEvaluator,
    build_evaluator_from_best,
    run_full_evaluation,
    EvaluationResult,
)
from src.evaluation.reporter import generate_all_reports

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py
  python scripts/evaluate.py --split val
  python scripts/evaluate.py --verbose
  python scripts/evaluate.py --checkpoint models/checkpoints/epoch_080_val0.9200.pth
        """,
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "both"],
        default="both",
        help="Which split to evaluate (default: both).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pth checkpoint (default: models/best/best_model.pth).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full per-class metrics table to terminal.",
    )
    parser.add_argument(
        "--no_report",
        action="store_true",
        help="Skip report generation (metrics only).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_prerequisites(dataset_cfg: dict) -> None:
    """Verify splits and best model exist before evaluation."""
    root       = get_project_root()
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    best_path  = root / "models" / "best" / "best_model.pth"

    if not (splits_dir / "X_test.npy").exists():
        logger.error(
            "data/splits/X_test.npy not found. "
            "Run  python scripts/preprocess.py  first."
        )
        sys.exit(1)


def check_best_model(checkpoint_override: str | None) -> Path:
    """Return the checkpoint path to load, or exit if not found."""
    root = get_project_root()
    if checkpoint_override:
        path = root / checkpoint_override
        if not path.exists():
            path = Path(checkpoint_override)
        if not path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_override}")
            sys.exit(1)
        return path

    best_path = root / "models" / "best" / "best_model.pth"
    if not best_path.exists():
        logger.error(
            "No best model found at models/best/best_model.pth.\n"
            "Train the model first:  python scripts/train.py"
        )
        sys.exit(1)
    return best_path


# ---------------------------------------------------------------------------
# Terminal summary printers
# ---------------------------------------------------------------------------

def print_result_summary(result: EvaluationResult, verbose: bool) -> None:
    """Print a formatted evaluation summary to the terminal."""
    split = result.split.upper()
    W     = 58

    print(f"\n{'='*W}")
    print(f"  Evaluation Results — {split} SET")
    print(f"{'='*W}")
    print(f"  Samples evaluated : {result.num_samples}")
    print(f"  Classes           : {len(result.class_names)}")
    print(f"{'-'*W}")
    print(f"  Top-1 Accuracy    : {result.accuracy*100:6.2f}%  "
          f"{'← ✓ TARGET MET' if result.target_met else '← ✗ below 90%'}")
    print(f"  Top-3 Accuracy    : {result.top3_accuracy*100:6.2f}%")
    print(f"  Macro F1          : {result.macro_f1*100:6.2f}%")
    print(f"  Macro Precision   : {result.macro_precision*100:6.2f}%")
    print(f"  Macro Recall      : {result.macro_recall*100:6.2f}%")
    print(f"  Weighted F1       : {result.weighted_f1*100:6.2f}%")
    print(f"{'-'*W}")

    # Worst classes
    worst = result.worst_classes(5)
    print(f"  5 Hardest Classes (F1):")
    for c in worst:
        bar = "█" * int(c["f1"] * 20)
        print(f"    {c['class_name']:<6} F1={c['f1']:.4f}  {bar}")

    # Top confused pairs
    if result.confused_pairs:
        print(f"\n  Top Confused Pairs:")
        for pair in result.confused_pairs[:5]:
            print(f"    {pair['true_class']:>4} → {pair['pred_class']:<4}  "
                  f"({pair['count']} times)")

    print(f"{'='*W}\n")

    # Verbose: full per-class table
    if verbose:
        print(f"  Full Per-Class Metrics ({split}):")
        print(f"  {'Class':<8} {'Prec':>7} {'Recall':>7} {'F1':>7} "
              f"{'Correct':>8} {'Total':>6}")
        print(f"  {'-'*50}")
        for cls in sorted(result.per_class, key=lambda x: x["f1"], reverse=True):
            flag = " ✗" if cls["f1"] < 0.85 else ""
            print(
                f"  {cls['class_name']:<8} "
                f"{cls['precision']:>7.4f} "
                f"{cls['recall']:>7.4f} "
                f"{cls['f1']:>7.4f}"
                f"{cls['correct']:>8}/{cls['support']:<6}"
                f"{flag}"
            )
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    get_logger("signbridge")

    # ── Load configs ──────────────────────────────────────────────────
    dataset_cfg  = load_config("dataset")
    model_cfg    = load_config("model",    resolve_paths=False)
    training_cfg = load_config("training", resolve_paths=False)

    if args.device:
        training_cfg["hardware"]["device"] = args.device

    # ── Pre-flight ────────────────────────────────────────────────────
    check_prerequisites(dataset_cfg)
    best_path = check_best_model(args.checkpoint)
    logger.info(f"Using checkpoint: {best_path}")

    # ── Device ───────────────────────────────────────────────────────
    device = get_device_from_config(training_cfg)

    # ── Build DataLoaders ─────────────────────────────────────────────
    from src.preprocessing.dataset_builder import build_dataloaders
    logger.info("Loading data splits...")
    _, val_loader, test_loader, _ = build_dataloaders(
        dataset_cfg=  dataset_cfg,
        training_cfg= training_cfg,
        force_rebuild=False,
    )

    # ── Load model ────────────────────────────────────────────────────
    if args.checkpoint:
        # Load specific checkpoint
        from src.models.model_factory import build_model, load_checkpoint
        model = build_model(model_cfg, dataset_cfg, device=device)
        model, _, ckpt_metrics = load_checkpoint(model, best_path, device=device)
        evaluator = ModelEvaluator(model=model, device=device)
    else:
        evaluator = build_evaluator_from_best(model_cfg, dataset_cfg, device)

    # ── Run evaluation ────────────────────────────────────────────────
    test_result = None
    val_result  = None

    if args.split in ("test", "both"):
        logger.info("Evaluating test set...")
        test_result = evaluator.evaluate(test_loader, split="test")
        print_result_summary(test_result, verbose=args.verbose)

    if args.split in ("val", "both"):
        logger.info("Evaluating val set...")
        val_result = evaluator.evaluate(val_loader, split="val")
        print_result_summary(val_result, verbose=args.verbose)

    # ── Fall back: generate dummy val/test if only one was requested ──
    if test_result is None:
        test_result = val_result
    if val_result is None:
        val_result = test_result

    # ── Generate reports ──────────────────────────────────────────────
    if not args.no_report:
        reports_dir = get_project_root() / "reports"
        paths = generate_all_reports(test_result, val_result, reports_dir)

        print(f"\n  Reports saved to:  {reports_dir}")
        for name, path in paths.items():
            print(f"    {name:<22} → {path.name}")

        print(f"\n  Open the report:  {paths.get('html_report', '')}")
        print()

    # ── Final verdict ─────────────────────────────────────────────────
    acc = test_result.accuracy if args.split != "val" else val_result.accuracy
    if acc >= 0.90:
        print(f"  ✓  Target accuracy (>90%) ACHIEVED — {acc*100:.2f}%")
        print(f"     Ready for Phase 7: Real-Time Inference\n")
    else:
        gap = 0.90 - acc
        print(f"  ⚠  Target not yet reached: {acc*100:.2f}%  (gap: {gap*100:.2f}%)")
        print(f"     Consider collecting more data or training longer.\n")


if __name__ == "__main__":
    main()
