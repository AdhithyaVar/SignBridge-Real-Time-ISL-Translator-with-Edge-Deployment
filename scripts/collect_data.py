"""
collect_data.py
---------------
Entry-point script for the SignBridge ISL dataset collection pipeline.

Multi-signer support
--------------------
Every recording session is tagged with a --signer_id.  Data is stored
under data/raw/<signer_id>/<class>/ so multiple signers can coexist.
This enables Leave-One-Signer-Out (LOSO) cross-validation — the only
honest generalisation metric for a sign language recognition system.

Target: collect data from at least 5 signers.

Usage
-----
    # First signer (default ID = signer_01)
    python scripts/collect_data.py

    # Additional signers (give each a unique ID)
    python scripts/collect_data.py --signer_id signer_02
    python scripts/collect_data.py --signer_id john
    python scripts/collect_data.py --signer_id priya

    # Collect a single class for one signer
    python scripts/collect_data.py --signer_id signer_02 --class_name A

    # Resume a signer's session from a specific class
    python scripts/collect_data.py --signer_id signer_02 --start_from M

    # Override sequences per class (50 is enough for additional signers)
    python scripts/collect_data.py --signer_id signer_02 --num_sequences 50

    # List all signers currently in the dataset
    python scripts/collect_data.py --list_signers

Controls (during recording window)
-----------------------------------
    Q  ->  Quit immediately
    R  ->  Reset current class session
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config_loader import load_config, ensure_dir
from src.utils.logger import get_logger
from src.utils.class_labels import CLASS_LABELS, NUM_CLASSES, requires_two_hands
from src.collection.collector import DataCollector

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — ISL Data Collection (multi-signer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Multi-signer workflow (run once per person):
  python scripts/collect_data.py --signer_id signer_01   # you
  python scripts/collect_data.py --signer_id signer_02   # friend 1
  python scripts/collect_data.py --signer_id signer_03   # friend 2
  python scripts/collect_data.py --signer_id signer_04   # friend 3
  python scripts/collect_data.py --signer_id signer_05   # friend 4

Then rebuild dataset with all signers combined:
  python scripts/preprocess.py --force

Run LOSO validation to get honest cross-signer accuracy:
  python scripts/loso_validate.py
        """,
    )
    parser.add_argument(
        "--signer_id", "-i",
        type=str,
        default="signer_01",
        help=(
            "Unique ID for this recording session. "
            "Data stored at data/raw/<signer_id>/<class>/. "
            "Use a different ID for every new person. "
            "(default: signer_01)"
        ),
    )
    parser.add_argument(
        "--class_name", "-c",
        type=str,
        default=None,
        help="Collect only this single class (e.g. 'A'). Default: all 26.",
    )
    parser.add_argument(
        "--start_from", "-s",
        type=str,
        default=None,
        help="Resume collection from this class (skips earlier classes).",
    )
    parser.add_argument(
        "--num_sequences", "-n",
        type=int,
        default=None,
        help=(
            "Sequences to record per class. "
            "Default from dataset.yaml (100). "
            "50 is sufficient for additional signers."
        ),
    )
    parser.add_argument(
        "--camera", "-k",
        type=int,
        default=None,
        help="Camera device index (default from dataset.yaml, usually 0).",
    )
    parser.add_argument(
        "--list_signers",
        action="store_true",
        help="List all signers currently in data/raw/ and exit.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight_check(dataset_cfg: dict, signer_id: str) -> None:
    """Validate environment before opening the webcam."""
    raw_dir = Path(dataset_cfg["paths"]["raw_data"]) / signer_id
    try:
        ensure_dir(raw_dir)
        test_file = raw_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
    except (PermissionError, OSError) as exc:
        logger.error(f"Cannot write to data/raw/{signer_id}/: {exc}")
        sys.exit(1)

    try:
        import cv2
        import mediapipe  # noqa: F401
        import numpy      # noqa: F401
    except ImportError as exc:
        logger.error(f"Missing dependency: {exc}")
        logger.error("Run:  pip install -r requirements.txt")
        sys.exit(1)

    logger.info(f"Pre-flight checks passed for signer '{signer_id}'.")


# ---------------------------------------------------------------------------
# Signer utilities
# ---------------------------------------------------------------------------

def list_signers(dataset_cfg: dict) -> None:
    """Print all signers currently present in data/raw/ and their progress."""
    raw_root = Path(dataset_cfg["paths"]["raw_data"])
    if not raw_root.exists():
        print("  data/raw/ does not exist yet. Run collect_data.py first.")
        return

    signers = sorted([
        d for d in raw_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not signers:
        print("  No signer directories found in data/raw/.")
        return

    num_seq = dataset_cfg["recording"].get("num_sequences", 100)
    print("\n" + "=" * 62)
    print("  Signers in data/raw/")
    print("=" * 62)
    print(f"  {'Signer':<18} {'Classes':>8} {'Sequences':>12} {'Complete':>10}")
    print("-" * 62)

    total_seqs = 0
    for signer_dir in signers:
        class_dirs  = [d for d in signer_dir.iterdir() if d.is_dir()]
        class_count = len(class_dirs)
        seq_count   = sum(
            len(list(d.glob("*.npy"))) for d in class_dirs
        )
        complete    = sum(
            1 for d in class_dirs
            if len(list(d.glob("*.npy"))) >= num_seq
        )
        total_seqs += seq_count
        print(
            f"  {signer_dir.name:<18} {class_count:>8} "
            f"{seq_count:>12} {complete:>8}/{NUM_CLASSES}"
        )
    print("-" * 62)
    print(f"  {'TOTAL':<18} {'':>8} {total_seqs:>12}")
    print("=" * 62)
    print(f"\n  For LOSO validation run:  python scripts/loso_validate.py\n")


# ---------------------------------------------------------------------------
# Summary printers
# ---------------------------------------------------------------------------

def print_collection_plan(
    dataset_cfg:   dict,
    target_classes:list[str],
    signer_id:     str,
) -> None:
    rec_cfg    = dataset_cfg.get("recording", {})
    num_seq    = rec_cfg.get("num_sequences", 100)
    seq_len    = rec_cfg.get("sequence_length", 30)
    camera_idx = rec_cfg.get("camera_index", 0)
    total_seqs = num_seq * len(target_classes)
    raw_dir    = Path(dataset_cfg["paths"]["raw_data"]) / signer_id

    print("\n" + "=" * 62)
    print("  SignBridge — ISL Data Collection")
    print("=" * 62)
    print(f"  Signer ID            : {signer_id}")
    print(f"  Classes to collect   : {len(target_classes)}")
    print(f"  Sequences per class  : {num_seq}")
    print(f"  Frames per sequence  : {seq_len}")
    print(f"  Total sequences      : {total_seqs}")
    print(f"  Camera index         : {camera_idx}")
    print(f"  Save directory       : {raw_dir}")
    print("-" * 62)
    print("  Classes:")
    for i, cls in enumerate(target_classes):
        hand_str = "2H" if requires_two_hands(cls) else "1H"
        print(f"    [{i+1:2d}] {cls:<14} ({hand_str})")
    print("=" * 62)
    print("  Controls:  Q = Quit   |   R = Reset current class")
    print("=" * 62 + "\n")


def print_final_summary(
    dataset_cfg:   dict,
    target_classes:list[str],
    signer_id:     str,
) -> None:
    raw_dir = Path(dataset_cfg["paths"]["raw_data"]) / signer_id
    num_seq = dataset_cfg["recording"].get("num_sequences", 100)
    print("\n" + "=" * 62)
    print(f"  Collection Complete — Signer: {signer_id}")
    print("=" * 62)
    total_saved = 0
    for cls in target_classes:
        class_dir   = raw_dir / cls
        saved_count = len(list(class_dir.glob("*.npy"))) if class_dir.exists() else 0
        total_saved += saved_count
        status = "\u2713" if saved_count >= num_seq else "\u2026"
        print(f"  {status} {cls:<16} : {saved_count:>4} sequences")
    print("-" * 62)
    print(f"  Total sequences saved : {total_saved}")
    print("=" * 62)

    # Show how many signers collected so far
    raw_root = Path(dataset_cfg["paths"]["raw_data"])
    existing_signers = [
        d.name for d in raw_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ] if raw_root.exists() else []
    print(f"\n  Signers collected: {len(existing_signers)}/5+ recommended")
    if len(existing_signers) < 5:
        remaining = 5 - len(existing_signers)
        print(f"  Need {remaining} more signer(s) for production-ready accuracy.")
        print(f"  Next signer command:")
        next_id = f"signer_{len(existing_signers)+1:02d}"
        print(f"    python scripts/collect_data.py --signer_id {next_id} "
              f"--num_sequences 50")
    else:
        print(f"  Sufficient signers collected.")
        print(f"  Run:  python scripts/loso_validate.py")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    get_logger("signbridge", log_level=log_level)

    dataset_cfg = load_config("dataset")

    # List signers and exit
    if args.list_signers:
        list_signers(dataset_cfg)
        return

    # Apply CLI overrides
    if args.num_sequences is not None:
        dataset_cfg["recording"]["num_sequences"] = args.num_sequences
        logger.info(f"Override: num_sequences = {args.num_sequences}")

    if args.camera is not None:
        dataset_cfg["recording"]["camera_index"] = args.camera
        logger.info(f"Override: camera_index = {args.camera}")

    # Inject signer_id into raw_data path
    signer_id = args.signer_id.strip().replace(" ", "_")
    raw_root  = Path(dataset_cfg["paths"]["raw_data"])
    dataset_cfg["paths"]["raw_data"] = raw_root / signer_id
    logger.info(f"Signer ID: {signer_id}")
    logger.info(f"Raw data path: {dataset_cfg['paths']['raw_data']}")

    # Pre-flight
    preflight_check(dataset_cfg, signer_id)

    # Determine target classes
    if args.class_name:
        if args.class_name not in CLASS_LABELS:
            logger.error(
                f"Unknown class '{args.class_name}'. "
                f"Valid: {CLASS_LABELS}"
            )
            sys.exit(1)
        target_classes = [args.class_name]
    else:
        target_classes = CLASS_LABELS[:]

    # Print plan
    print_collection_plan(dataset_cfg, target_classes, signer_id)
    input("  Press ENTER to start, or Ctrl+C to cancel...\n")

    # Run collection
    collector = DataCollector(dataset_cfg)
    try:
        if args.class_name:
            collector.collect_class(args.class_name)
        else:
            collector.collect_all(start_from=args.start_from)
    except KeyboardInterrupt:
        logger.warning("Collection interrupted by Ctrl+C.")
    except RuntimeError as exc:
        logger.error(f"Collection error: {exc}")
        sys.exit(1)

    print_final_summary(dataset_cfg, target_classes, signer_id)


if __name__ == "__main__":
    main()
