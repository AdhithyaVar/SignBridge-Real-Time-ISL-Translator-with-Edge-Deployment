"""
migrate_to_multisigner.py
--------------------------
One-time migration: move existing flat single-signer data from

    data/raw/<class>/*.npy          (old layout)

to the multi-signer layout:

    data/raw/<signer_id>/<class>/*.npy   (new layout)

This migration is REQUIRED before adding a second signer so that
dataset_builder.py can correctly detect the multi-signer structure.

Run once, then add more signers with:
    python scripts/collect_data.py --signer_id signer_02 --num_sequences 50

Usage
-----
    # Dry run — show what will be moved without touching anything
    python scripts/migrate_to_multisigner.py --dry_run

    # Migrate (assigns existing data to signer_01 by default)
    python scripts/migrate_to_multisigner.py

    # Assign a different ID to the existing data
    python scripts/migrate_to_multisigner.py --signer_id adhithya
"""

import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.class_labels import CLASS_LABELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — Migrate flat data to multi-signer layout"
    )
    parser.add_argument(
        "--signer_id", "-i",
        type=str,
        default="signer_01",
        help="Signer ID to assign to the existing data (default: signer_01).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be moved without touching any files.",
    )
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    raw_root = _PROJECT_ROOT / "data" / "raw"

    if not raw_root.exists():
        print("  data/raw/ does not exist. Nothing to migrate.\n")
        return

    # Detect flat class directories
    flat_class_dirs = [
        d for d in raw_root.iterdir()
        if d.is_dir()
        and d.name in CLASS_LABELS
        and not d.name.startswith(".")
    ]

    if not flat_class_dirs:
        print("  No flat class directories found in data/raw/.")
        print("  Data may already be in multi-signer layout.")
        print("  Run: python scripts/collect_data.py --list_signers\n")
        return

    # Check for existing signer dirs to avoid collisions
    existing_signer_dir = raw_root / args.signer_id
    if existing_signer_dir.exists():
        print(
            f"  ERROR: {existing_signer_dir} already exists.\n"
            f"  Choose a different --signer_id or remove the existing directory.\n"
        )
        sys.exit(1)

    # Count sequences
    total_seqs = sum(
        len(list(d.glob("*.npy"))) for d in flat_class_dirs
    )

    # Print migration plan
    print("\n" + "=" * 62)
    print("  SignBridge — Multi-Signer Migration")
    print("=" * 62)
    print(f"  Signer ID for existing data : {args.signer_id}")
    print(f"  Source                      : data/raw/<class>/")
    print(f"  Destination                 : data/raw/{args.signer_id}/<class>/")
    print(f"  Class directories found     : {len(flat_class_dirs)}")
    print(f"  Total sequences             : {total_seqs}")
    print("-" * 62)
    for cls_dir in sorted(flat_class_dirs, key=lambda d: d.name):
        n = len(list(cls_dir.glob("*.npy")))
        print(f"  {cls_dir.name:<14}  {n:>5} sequences")
    print("=" * 62)

    if args.dry_run:
        print("\n  DRY RUN — no files moved.")
        print(f"  Re-run without --dry_run to execute migration.\n")
        return

    # Confirm
    print()
    ans = input("  Proceed with migration? [y/N]: ").strip().lower()
    if ans != "y":
        print("  Migration cancelled.\n")
        return

    # Execute migration — move each class dir into signer_id/
    dest_signer = raw_root / args.signer_id
    dest_signer.mkdir(parents=True, exist_ok=True)
    moved = 0

    for cls_dir in flat_class_dirs:
        dest_cls = dest_signer / cls_dir.name
        try:
            shutil.move(str(cls_dir), str(dest_cls))
            n = len(list(dest_cls.glob("*.npy")))
            logger.info(f"  Moved: {cls_dir.name}/ → {args.signer_id}/{cls_dir.name}/  ({n} seqs)")
            moved += 1
        except Exception as exc:
            logger.error(f"  Failed to move {cls_dir}: {exc}")

    print("\n" + "=" * 62)
    print(f"  Migration complete — {moved}/{len(flat_class_dirs)} classes moved")
    print(f"  Data now at: data/raw/{args.signer_id}/")
    print("=" * 62)
    print(f"\n  Next steps:")
    print(f"  1. Verify migration:")
    print(f"     python scripts/collect_data.py --list_signers")
    print(f"  2. Add more signers (ask friends/family):")
    print(f"     python scripts/collect_data.py --signer_id signer_02 --num_sequences 50")
    print(f"  3. Rebuild dataset with all signers:")
    print(f"     python scripts/preprocess.py --force")
    print(f"  4. Retrain model:")
    print(f"     python scripts/train.py")
    print(f"  5. Run LOSO validation (2+ signers required):")
    print(f"     python scripts/loso_validate.py\n")


if __name__ == "__main__":
    main()
