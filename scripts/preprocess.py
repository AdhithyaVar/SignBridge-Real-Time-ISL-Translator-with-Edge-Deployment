"""
preprocess.py
-------------
CLI entry-point for the full SignBridge preprocessing pipeline.

What this script does
---------------------
1. Loads dataset.yaml and training.yaml
2. Scans data/raw/ and reports what data is available
3. Runs the full DatasetBuilder pipeline:
     a. Load all raw .npy sequences
     b. Wrist-relative normalization
     c. Stratified train / val / test split
     d. Fit SequenceScaler on training set
     e. Augment training set (×4 copies)
     f. Save all arrays + scaler to data/splits/
4. Prints a detailed summary of the final dataset

Usage
-----
    python scripts/preprocess.py
    python scripts/preprocess.py --force          # rebuild even if splits exist
    python scripts/preprocess.py --no_augment     # skip augmentation
    python scripts/preprocess.py --copies 6       # override augmentation copies
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.class_labels import CLASS_LABELS, NUM_CLASSES
from src.preprocessing.dataset_builder import DatasetBuilder

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/preprocess.py
  python scripts/preprocess.py --force
  python scripts/preprocess.py --no_augment
  python scripts/preprocess.py --copies 6
        """,
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild even if data/splits/ already exists.",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable augmentation (use raw training sequences only).",
    )
    parser.add_argument(
        "--copies", "-c",
        type=int,
        default=None,
        help="Number of augmented copies per sample (default from dataset.yaml).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight: check raw data availability
# ---------------------------------------------------------------------------

def check_raw_data(dataset_cfg: dict) -> dict[str, int]:
    """
    Scan data/raw/ and return a dict of {class_name: sequence_count}.
    Warns about missing or low-count classes.
    """
    raw_dir    = Path(dataset_cfg["paths"]["raw_data"])
    min_needed = int(dataset_cfg["recording"].get("num_sequences", 100))
    counts: dict[str, int] = {}

    print("\n" + "=" * 62)
    print("  Raw Data Inventory")
    print("=" * 62)

    for cls in CLASS_LABELS:
        cls_dir = raw_dir / cls
        if not cls_dir.exists():
            counts[cls] = 0
            status = "  ✗  MISSING"
        else:
            n = len(list(cls_dir.glob(f"{cls}_*.npy")))
            counts[cls] = n
            if n == 0:
                status = "  ✗  EMPTY"
            elif n < min_needed:
                status = f"  ⚠  {n:4d}  (below target {min_needed})"
            else:
                status = f"  ✓  {n:4d}"
        print(f"  {cls:<14} {status}")

    total   = sum(counts.values())
    missing = [c for c, n in counts.items() if n == 0]

    print("=" * 62)
    print(f"  Total sequences  : {total}")
    print(f"  Classes with data: {sum(1 for n in counts.values() if n > 0)}/{NUM_CLASSES}")
    if missing:
        print(f"  Missing classes  : {missing}")
    print("=" * 62 + "\n")

    if total == 0:
        logger.error(
            "No data found in data/raw/. "
            "Run  python scripts/collect_data.py  first."
        )
        sys.exit(1)

    return counts


# ---------------------------------------------------------------------------
# Post-processing summary
# ---------------------------------------------------------------------------

def print_dataset_summary(splits_dir: Path) -> None:
    """Print a rich summary of the built dataset."""
    info_path = splits_dir / "dataset_info.json"
    if not info_path.exists():
        logger.warning("dataset_info.json not found — skipping summary.")
        return

    with open(info_path) as fh:
        info = json.load(fh)

    print("\n" + "=" * 62)
    print("  Dataset Summary")
    print("=" * 62)
    print(f"  Classes          : {info['num_classes']}")
    print(f"  Sequence length  : {info['sequence_length']} frames")
    print(f"  Feature dim      : {info['feature_dim']} (2 hands × 21 lm × 3 coords)")
    print(f"  Aug copies       : {info['augmentation_copies']} per raw sample")
    print("-" * 62)
    print(f"  Train  size      : {info['train_size']:>7,}")
    print(f"  Val    size      : {info['val_size']:>7,}")
    print(f"  Test   size      : {info['test_size']:>7,}")
    total = info['train_size'] + info['val_size'] + info['test_size']
    print(f"  TOTAL            : {total:>7,}")
    print("-" * 62)

    # Class distribution in training set
    class_counts = info.get("class_counts_train", {})
    if class_counts:
        vals   = list(class_counts.values())
        mean_c = sum(vals) / len(vals)
        min_c  = min(vals)
        max_c  = max(vals)
        print(f"  Train class dist : min={min_c}  max={max_c}  mean={mean_c:.0f}")

    print("=" * 62 + "\n")

    # Load and verify shapes
    try:
        X_train = np.load(splits_dir / "X_train.npy")
        X_val   = np.load(splits_dir / "X_val.npy")
        X_test  = np.load(splits_dir / "X_test.npy")
        print(f"  X_train shape  : {X_train.shape}  dtype={X_train.dtype}")
        print(f"  X_val   shape  : {X_val.shape}  dtype={X_val.dtype}")
        print(f"  X_test  shape  : {X_test.shape}  dtype={X_test.dtype}")
        print(f"\n  Value range (train): [{X_train.min():.3f}, {X_train.max():.3f}]")
        print(f"  Mean / Std  (train): {X_train.mean():.4f} / {X_train.std():.4f}")
        print()
    except Exception as exc:
        logger.warning(f"Could not load arrays for shape check: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    get_logger("signbridge", log_level=log_level)

    print("\n" + "=" * 62)
    print("  SignBridge — Preprocessing Pipeline")
    print("=" * 62)

    # Load configs
    dataset_cfg  = load_config("dataset")
    training_cfg = load_config("training")

    # Apply CLI overrides
    if args.no_augment:
        dataset_cfg["augmentation"]["enabled"] = False
        logger.info("Override: augmentation disabled")

    if args.copies is not None:
        dataset_cfg["augmentation"]["per_sample_copies"] = args.copies
        logger.info(f"Override: per_sample_copies = {args.copies}")

    # Check raw data
    check_raw_data(dataset_cfg)

    # Confirm before proceeding
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    if splits_dir.exists() and any(splits_dir.glob("X_train.npy")):
        if not args.force:
            print("  data/splits/ already exists.")
            ans = input("  Rebuild from scratch? [y/N]: ").strip().lower()
            if ans != "y":
                print("  Skipped — using existing splits.\n")
                print_dataset_summary(splits_dir)
                return

    # Run pipeline
    logger.info("Starting preprocessing pipeline...")
    try:
        builder = DatasetBuilder(dataset_cfg, training_cfg)
        train_loader, val_loader, test_loader, scaler = builder.build(
            force_rebuild=True
        )
    except RuntimeError as exc:
        logger.error(f"Preprocessing failed: {exc}")
        sys.exit(1)

    # Print summary
    print_dataset_summary(splits_dir)

    print("  ✓  Preprocessing complete.")
    print("  Next step:  python scripts/train.py\n")


if __name__ == "__main__":
    main()
