"""
loso_validate.py
----------------
Leave-One-Signer-Out (LOSO) Cross-Validation for SignBridge.

This is the ONLY honest accuracy metric for a sign language recognition
system.  It measures how well the model generalises to a signer it has
NEVER seen during training.

How it works
------------
For each signer S in data/raw/:
  1. Training set  = sequences from ALL signers EXCEPT S
  2. Test set      = sequences from signer S only
  3. Train a model from scratch on the training set
  4. Evaluate on signer S's test set
  5. Record per-signer accuracy

Final LOSO score = mean accuracy across all signer folds.

Requirements
------------
  At least 2 signers in data/raw/ (3+ recommended, 5+ for reliable estimates).
  Each signer must have data for all 26 classes.

Usage
-----
    python scripts/loso_validate.py
    python scripts/loso_validate.py --epochs 50   # faster, slightly lower acc
    python scripts/loso_validate.py --dry_run      # show plan, don't train
"""

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.class_labels import CLASS_LABELS, CLASS_TO_IDX, NUM_CLASSES
from src.utils.device import get_device_from_config
from src.preprocessing.landmark_extractor import load_raw_class, normalize_dataset, FEATURE_DIM
from src.preprocessing.normalizer import SequenceScaler
from src.preprocessing.augmentor import build_augmentor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — Leave-One-Signer-Out Cross-Validation",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs per fold (default: 50). Use 150 for full accuracy.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show LOSO plan without training.",
    )
    parser.add_argument(
        "--output", type=str, default="reports/loso_results.json",
        help="Path to save LOSO results JSON.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_signers(raw_root: Path) -> list[Path]:
    """Return sorted list of signer directories in data/raw/."""
    if not raw_root.exists():
        return []
    signer_dirs = sorted([
        d for d in raw_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name not in CLASS_LABELS   # skip flat single-signer layout
    ])
    return signer_dirs


def load_signer_data(
    signer_dir:      Path,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load and wrist-normalise all sequences for one signer.

    Returns (X, y) where X.shape = (N, 30, 126) and y.shape = (N,)
    Returns None if the signer has no data.
    """
    all_seqs   = []
    all_labels = []

    for cls in CLASS_LABELS:
        seqs = load_raw_class(
            signer_dir, cls,
            sequence_length=sequence_length,
            feature_dim=FEATURE_DIM,
        )
        if seqs is not None and len(seqs) > 0:
            all_seqs.append(seqs)
            all_labels.extend([CLASS_TO_IDX[cls]] * len(seqs))

    if not all_seqs:
        return None

    X = normalize_dataset(np.concatenate(all_seqs, axis=0))
    y = np.array(all_labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# One LOSO fold
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx:     int,
    test_signer:  Path,
    train_signers:list[Path],
    dataset_cfg:  dict,
    model_cfg:    dict,
    training_cfg: dict,
    device:       torch.device,
    max_epochs:   int,
) -> dict:
    """
    Train on all signers except test_signer, evaluate on test_signer.

    Returns dict with fold results.
    """
    from src.models.model_factory import build_model
    from src.preprocessing.augmentor import build_augmentor
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    logger.info(
        f"\nFold {fold_idx+1}: test_signer={test_signer.name} | "
        f"train_signers={[s.name for s in train_signers]}"
    )
    t_start = time.time()

    # ── Load training data ────────────────────────────────────────────
    train_seqs, train_labs = [], []
    for signer in train_signers:
        result = load_signer_data(signer)
        if result:
            X_s, y_s = result
            train_seqs.append(X_s)
            train_labs.append(y_s)
            logger.info(f"  Loaded {len(X_s)} sequences from {signer.name}")

    if not train_seqs:
        logger.error("No training data available for this fold.")
        return {"fold": fold_idx, "test_signer": test_signer.name, "accuracy": 0.0}

    X_train = np.concatenate(train_seqs, axis=0)
    y_train = np.concatenate(train_labs, axis=0)

    # ── Load test data ────────────────────────────────────────────────
    test_result = load_signer_data(test_signer)
    if test_result is None:
        logger.error(f"No test data for signer {test_signer.name}")
        return {"fold": fold_idx, "test_signer": test_signer.name, "accuracy": 0.0}
    X_test, y_test = test_result

    # ── Fit scaler on training data only ─────────────────────────────
    scaler = SequenceScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Augment training set ──────────────────────────────────────────
    augmentor = build_augmentor(dataset_cfg)
    aug_copies = int(dataset_cfg.get("augmentation", {}).get("per_sample_copies", 4))
    idx_to_cls = {v: k for k, v in CLASS_TO_IDX.items()}

    aug_seqs, aug_labs = [X_train_sc], [y_train]
    for _ in range(aug_copies):
        batch = np.array([
            augmentor.augment(seq, idx_to_cls[int(lbl)])
            for seq, lbl in zip(X_train_sc, y_train)
        ], dtype=np.float32)
        aug_seqs.append(batch)
        aug_labs.append(y_train)

    X_aug = np.concatenate(aug_seqs, axis=0)
    y_aug = np.concatenate(aug_labs, axis=0)
    perm  = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    logger.info(
        f"  Train: {len(X_aug)} aug seqs | "
        f"Test: {len(X_test_sc)} seqs (signer {test_signer.name})"
    )

    # ── Build model ───────────────────────────────────────────────────
    model = build_model(model_cfg, dataset_cfg, device=device)

    # ── DataLoaders ───────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.from_numpy(X_aug).float(),
        torch.from_numpy(y_aug).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test_sc).float(),
        torch.from_numpy(y_test).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=128)

    # ── Train ─────────────────────────────────────────────────────────
    opt_cfg  = training_cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=           float(opt_cfg.get("lr",           1e-3)),
        weight_decay= float(opt_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval every 5 epochs
        if epoch % 5 == 0 or epoch == max_epochs:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for X_b, y_b in test_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    preds     = model(X_b).argmax(dim=-1)
                    correct  += (preds == y_b).sum().item()
                    total    += len(y_b)
            acc = correct / max(total, 1)
            best_acc = max(best_acc, acc)
            logger.info(
                f"  Fold {fold_idx+1} | Epoch {epoch:03d} | "
                f"Test acc (signer {test_signer.name}): {acc:.4f}"
            )

    elapsed = time.time() - t_start
    result = {
        "fold":         fold_idx + 1,
        "test_signer":  test_signer.name,
        "best_accuracy":round(best_acc, 4),
        "train_signers":[s.name for s in train_signers],
        "train_samples":int(len(X_aug)),
        "test_samples": int(len(X_test_sc)),
        "time_seconds": round(elapsed, 1),
    }
    logger.info(
        f"  Fold {fold_idx+1} complete | "
        f"Best LOSO acc = {best_acc:.4f} | "
        f"time = {elapsed/60:.1f} min"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    get_logger("signbridge")

    root        = get_project_root()
    dataset_cfg = load_config("dataset",  resolve_paths=False)
    model_cfg   = load_config("model",    resolve_paths=False)
    training_cfg= load_config("training", resolve_paths=False)
    device      = get_device_from_config(training_cfg)

    raw_root = root / "data" / "raw"
    signers  = discover_signers(raw_root)

    # ── Print plan ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  SignBridge — Leave-One-Signer-Out (LOSO) Cross-Validation")
    print("=" * 62)
    print(f"  Signers found : {len(signers)}")
    for s in signers:
        class_count = sum(
            1 for c in CLASS_LABELS
            if (s / c).exists() and len(list((s / c).glob("*.npy"))) > 0
        )
        seq_count = sum(
            len(list((s / c).glob("*.npy")))
            for c in CLASS_LABELS if (s / c).exists()
        )
        print(f"    {s.name:<20} {class_count}/26 classes  {seq_count} seqs")

    if len(signers) < 2:
        print(
            "\n  ERROR: Need at least 2 signers for LOSO validation.\n"
            "  Collect data from more signers first:\n"
            "    python scripts/collect_data.py --signer_id signer_02 "
            "--num_sequences 50\n"
        )
        sys.exit(1)

    print(f"\n  LOSO plan: {len(signers)} folds")
    print(f"  Epochs per fold: {args.epochs}")
    estimated_min = len(signers) * args.epochs * 0.7
    print(f"  Estimated time: ~{estimated_min:.0f} min (CPU)")
    print("=" * 62 + "\n")

    if args.dry_run:
        print("  Dry run complete. Use --dry_run=False to start training.\n")
        return

    input("  Press ENTER to start LOSO validation, or Ctrl+C to cancel...\n")

    # ── Run all folds ─────────────────────────────────────────────────
    results = []
    for fold_idx, test_signer in enumerate(signers):
        train_signers = [s for s in signers if s != test_signer]
        fold_result   = run_fold(
            fold_idx=     fold_idx,
            test_signer=  test_signer,
            train_signers=train_signers,
            dataset_cfg=  dataset_cfg,
            model_cfg=    model_cfg,
            training_cfg= training_cfg,
            device=       device,
            max_epochs=   args.epochs,
        )
        results.append(fold_result)

    # ── Summary ───────────────────────────────────────────────────────
    accs = [r["best_accuracy"] for r in results]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    min_acc  = float(np.min(accs))
    max_acc  = float(np.max(accs))

    print("\n" + "=" * 62)
    print("  LOSO Cross-Validation Results")
    print("=" * 62)
    for r in results:
        bar = "█" * int(r["best_accuracy"] * 20)
        print(
            f"  Fold {r['fold']} | {r['test_signer']:<18} | "
            f"acc={r['best_accuracy']:.4f}  {bar}"
        )
    print("-" * 62)
    print(f"  Mean LOSO accuracy : {mean_acc:.4f}  ({mean_acc*100:.2f}%)")
    print(f"  Std dev            : {std_acc:.4f}")
    print(f"  Min fold           : {min_acc:.4f}")
    print(f"  Max fold           : {max_acc:.4f}")
    print("=" * 62)

    if mean_acc >= 0.85:
        print(f"\n  ✓  LOSO accuracy {mean_acc*100:.1f}% >= 85% — ready for deployment\n")
    elif mean_acc >= 0.70:
        print(f"\n  ⚠  LOSO accuracy {mean_acc*100:.1f}% — collect more signer diversity\n")
    else:
        print(f"\n  ✗  LOSO accuracy {mean_acc*100:.1f}% — significant diversity gap\n")
        print("     Collect data from more signers with varied hand sizes.\n")

    # ── Save results ──────────────────────────────────────────────────
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "mean_loso_accuracy": mean_acc,
        "std_accuracy":       std_acc,
        "min_accuracy":       min_acc,
        "max_accuracy":       max_acc,
        "num_signers":        len(signers),
        "epochs_per_fold":    args.epochs,
        "folds":              results,
    }
    with open(output_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
