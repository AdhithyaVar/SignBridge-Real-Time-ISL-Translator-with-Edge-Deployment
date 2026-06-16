"""
dataset_builder.py
------------------
Builds the final train / val / test splits and PyTorch DataLoaders.

Pipeline
--------
1. Load all raw .npy sequences from data/raw/<class>/
2. Apply wrist-relative normalization (landmark_extractor)
3. Create stratified train / val / test split (sklearn)
4. Fit SequenceScaler on training set ONLY
5. Apply scaler to all three splits
6. Apply augmentation to training set only (N_aug = N_raw × per_sample_copies)
7. Save all arrays + scaler to data/splits/
8. Return PyTorch Dataset and DataLoader objects for each split

Saved artefacts (data/splits/)
-------------------------------
X_train.npy         (N_train_aug, T, 126)
y_train.npy         (N_train_aug,)
X_val.npy           (N_val, T, 126)
y_val.npy           (N_val,)
X_test.npy          (N_test, T, 126)
y_test.npy          (N_test,)
scaler_mean.npy     (126,)
scaler_std.npy      (126,)
dataset_info.json   summary statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.landmark_extractor import (
    load_raw_class,
    normalize_dataset,
    SEQUENCE_LEN,
    FEATURE_DIM,
)
from src.preprocessing.normalizer import SequenceScaler
from src.preprocessing.augmentor import Augmentor, build_augmentor
from src.utils.class_labels import CLASS_LABELS, CLASS_TO_IDX, NUM_CLASSES
from src.utils.config_loader import ensure_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ISLDataset(Dataset):
    """
    PyTorch Dataset for ISL landmark sequences.

    Parameters
    ----------
    sequences : np.ndarray, shape (N, T, 126)
    labels    : np.ndarray, shape (N,)  int64
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels:    np.ndarray,
    ) -> None:
        self.X = torch.from_numpy(sequences.astype(np.float32))   # (N, T, 126)
        self.y = torch.from_numpy(labels.astype(np.int64))         # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    @property
    def sequence_length(self) -> int:
        return self.X.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.X.shape[2]


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Orchestrates the full preprocessing pipeline from raw data to DataLoaders.

    Parameters
    ----------
    dataset_cfg  : dict — parsed dataset.yaml
    training_cfg : dict — parsed training.yaml (for batch size / num_workers)
    """

    def __init__(
        self,
        dataset_cfg:  dict,
        training_cfg: dict,
    ) -> None:
        self.dataset_cfg  = dataset_cfg
        self.training_cfg = training_cfg

        self.raw_dir       = Path(dataset_cfg["paths"]["raw_data"])
        self.splits_dir    = Path(dataset_cfg["paths"]["splits_dir"])
        self.rec_cfg       = dataset_cfg.get("recording",    {})
        self.split_cfg     = dataset_cfg.get("splits",       {})
        self.aug_cfg       = dataset_cfg.get("augmentation", {})

        self.sequence_length = int(self.rec_cfg.get("sequence_length", SEQUENCE_LEN))
        self.n_aug_copies    = int(self.aug_cfg.get("per_sample_copies", 4))

        self.train_ratio = float(self.split_cfg.get("train", 0.75))
        self.val_ratio   = float(self.split_cfg.get("val",   0.15))
        self.test_ratio  = float(self.split_cfg.get("test",  0.10))
        self.seed        = int(  self.split_cfg.get("random_seed", 42))

        self.batch_size   = int(training_cfg.get("training", {}).get("batch_size",   64))
        self.val_batch    = int(training_cfg.get("training", {}).get("val_batch_size",128))
        self.num_workers  = int(training_cfg.get("hardware", {}).get("num_workers",   4))
        self.pin_memory   = bool(training_cfg.get("hardware", {}).get("pin_memory", True))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def build(
        self,
        force_rebuild: bool = False,
    ) -> tuple[DataLoader, DataLoader, DataLoader, SequenceScaler]:
        """
        Run the full pipeline and return DataLoaders.

        Parameters
        ----------
        force_rebuild : bool
            If True, re-run even if splits already exist.

        Returns
        -------
        train_loader, val_loader, test_loader, scaler
        """
        splits_exist = (
            (self.splits_dir / "X_train.npy").exists()
            and (self.splits_dir / "X_val.npy").exists()
            and (self.splits_dir / "X_test.npy").exists()
            and (self.splits_dir / "scaler_mean.npy").exists()
        )

        if splits_exist and not force_rebuild:
            logger.info("Splits already exist — loading from disk.")
            return self._load_splits_from_disk()

        logger.info("Building dataset from raw sequences...")
        return self._run_full_pipeline()

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_full_pipeline(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader, SequenceScaler]:
        """Execute all preprocessing stages end to end."""

        # Stage 1: Load raw data
        all_seqs, all_labels = self._load_all_raw()
        logger.info(
            f"Loaded {len(all_seqs)} raw sequences | "
            f"{NUM_CLASSES} classes | "
            f"shape per seq: ({self.sequence_length}, {FEATURE_DIM})"
        )

        # Stage 2: Wrist-relative normalization
        logger.info("Applying wrist-relative normalization...")
        all_seqs = normalize_dataset(all_seqs)

        # Stage 3: Stratified split (on raw sequences before augmentation)
        logger.info(
            f"Splitting {len(all_seqs)} sequences | "
            f"train={self.train_ratio} val={self.val_ratio} test={self.test_ratio}"
        )
        X_train_raw, y_train_raw, X_val, y_val, X_test, y_test = \
            self._stratified_split(all_seqs, all_labels)

        logger.info(
            f"Raw split sizes — "
            f"train={len(X_train_raw)} | val={len(X_val)} | test={len(X_test)}"
        )

        # Stage 4: Fit scaler on training data only
        scaler = SequenceScaler()
        X_train_raw_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled       = scaler.transform(X_val)
        X_test_scaled      = scaler.transform(X_test)
        logger.info(f"Scaler fitted: {scaler}")

        # Stage 5: Augment training set
        augmentor = build_augmentor(self.dataset_cfg)
        aug_enabled = bool(self.aug_cfg.get("enabled", True))

        if aug_enabled:
            logger.info(
                f"Augmenting training set: "
                f"{len(X_train_raw_scaled)} raw → "
                f"×{self.n_aug_copies} copies + originals..."
            )
            X_train, y_train = self._augment_training(
                X_train_raw_scaled, y_train_raw,
                augmentor, self.n_aug_copies,
            )
            logger.info(
                f"Augmented training size: {len(X_train)} sequences"
            )
        else:
            X_train = X_train_raw_scaled
            y_train = y_train_raw
            logger.info("Augmentation disabled — using raw training sequences.")

        # Stage 6: Save artefacts
        ensure_dir(self.splits_dir)
        self._save_splits(
            X_train, y_train,
            X_val_scaled,   y_val,
            X_test_scaled,  y_test,
            scaler,
        )

        # Stage 7: Build DataLoaders
        return self._make_dataloaders(
            X_train, y_train,
            X_val_scaled,  y_val,
            X_test_scaled, y_test,
            scaler,
        )

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _load_all_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load all raw .npy files for all 26 classes.

        Supports both layouts:
          Single-signer: data/raw/<class>/*.npy
          Multi-signer:  data/raw/<signer_id>/<class>/*.npy
        """
        raw_root = Path(self.raw_dir)
        all_seqs:   list[np.ndarray] = []
        all_labels: list[int]        = []
        missing_classes              = []

        # Detect layout: multi-signer if subdirs contain CLASS_LABELS
        # Single-signer: raw_root/A/, raw_root/B/, ...
        # Multi-signer:  raw_root/signer_01/A/, raw_root/signer_02/A/, ...
        first_level = [d for d in raw_root.iterdir() if d.is_dir()] \
            if raw_root.exists() else []
        is_multi_signer = (
            len(first_level) > 0
            and first_level[0].name not in CLASS_LABELS
        )

        if is_multi_signer:
            signer_dirs = sorted(first_level)
            logger.info(
                f"Multi-signer layout detected | "
                f"signers={[d.name for d in signer_dirs]}"
            )
            # Load from every signer
            for signer_dir in signer_dirs:
                seqs, labels = self._load_signer(signer_dir)
                all_seqs.extend(seqs)
                all_labels.extend(labels)
        else:
            # Single-signer or flat layout
            logger.info("Single-signer layout detected.")
            for cls in CLASS_LABELS:
                seqs = load_raw_class(
                    raw_root, cls,
                    sequence_length=self.sequence_length,
                    feature_dim=FEATURE_DIM,
                )
                if seqs is None or len(seqs) == 0:
                    missing_classes.append(cls)
                    logger.warning(f"No data for class '{cls}' — skipping.")
                    continue
                label = CLASS_TO_IDX[cls]
                all_seqs.append(seqs)
                all_labels.extend([label] * len(seqs))
                logger.debug(f"  {cls}: {len(seqs)} sequences")

        if missing_classes:
            logger.warning(
                f"Missing data for {len(missing_classes)} class(es): "
                f"{missing_classes}"
            )
        if not all_seqs:
            raise RuntimeError(
                "No data found in data/raw/. "
                "Run  python scripts/collect_data.py  first."
            )

        X = np.concatenate(all_seqs, axis=0)     # (N, T, 126)
        y = np.array(all_labels, dtype=np.int64)  # (N,)
        return X, y

    def _load_signer(
        self,
        signer_dir: Path,
    ) -> tuple[list[np.ndarray], list[int]]:
        """Load all class sequences for one signer directory."""
        seqs:   list[np.ndarray] = []
        labels: list[int]        = []
        for cls in CLASS_LABELS:
            cls_seqs = load_raw_class(
                signer_dir, cls,
                sequence_length=self.sequence_length,
                feature_dim=FEATURE_DIM,
            )
            if cls_seqs is not None and len(cls_seqs) > 0:
                seqs.append(cls_seqs)
                labels.extend([CLASS_TO_IDX[cls]] * len(cls_seqs))
        return seqs, labels

    def _stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Create stratified train / val / test splits.
        Returns X_train, y_train, X_val, y_val, X_test, y_test.
        """
        # First split: separate test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_ratio,
            random_state=self.seed,
        )
        train_val_idx, test_idx = next(sss_test.split(X, y))

        X_trainval = X[train_val_idx]
        y_trainval = y[train_val_idx]
        X_test     = X[test_idx]
        y_test     = y[test_idx]

        # Second split: separate validation from train
        val_fraction = self.val_ratio / (self.train_ratio + self.val_ratio)
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=self.seed + 1,
        )
        train_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))

        return (
            X_trainval[train_idx], y_trainval[train_idx],
            X_trainval[val_idx],   y_trainval[val_idx],
            X_test,                y_test,
        )

    def _augment_training(
        self,
        X:         np.ndarray,
        y:         np.ndarray,
        augmentor: Augmentor,
        n_copies:  int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate n_copies augmented versions for each training sample
        and concatenate with the originals.
        """
        aug_seqs:   list[np.ndarray] = [X]   # start with originals
        aug_labels: list[np.ndarray] = [y]

        idx_to_cls = {v: k for k, v in CLASS_TO_IDX.items()}

        for copy_i in range(n_copies):
            batch = np.empty_like(X)
            for i, (seq, label) in enumerate(zip(X, y)):
                cls_name  = idx_to_cls[int(label)]
                batch[i]  = augmentor.augment(seq, cls_name)
            aug_seqs.append(batch)
            aug_labels.append(y.copy())
            logger.debug(f"  Augmentation copy {copy_i+1}/{n_copies} done")

        X_aug = np.concatenate(aug_seqs,   axis=0)
        y_aug = np.concatenate(aug_labels, axis=0)

        # Shuffle combined dataset
        rng   = np.random.default_rng(self.seed)
        perm  = rng.permutation(len(X_aug))
        return X_aug[perm], y_aug[perm]

    def _save_splits(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
        scaler:  SequenceScaler,
    ) -> None:
        """Save all split arrays and scaler to data/splits/."""
        d = self.splits_dir
        np.save(d / "X_train.npy", X_train.astype(np.float32))
        np.save(d / "y_train.npy", y_train.astype(np.int64))
        np.save(d / "X_val.npy",   X_val.astype(np.float32))
        np.save(d / "y_val.npy",   y_val.astype(np.int64))
        np.save(d / "X_test.npy",  X_test.astype(np.float32))
        np.save(d / "y_test.npy",  y_test.astype(np.int64))
        scaler.save(d)

        # Save human-readable summary
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts   = {CLASS_LABELS[int(k)]: int(v)
                          for k, v in zip(unique, counts)}
        info = {
            "num_classes":      NUM_CLASSES,
            "sequence_length":  int(X_train.shape[1]),
            "feature_dim":      int(X_train.shape[2]),
            "train_size":       int(len(X_train)),
            "val_size":         int(len(X_val)),
            "test_size":        int(len(X_test)),
            "augmentation_copies": self.n_aug_copies,
            "class_counts_train":  class_counts,
        }
        with open(d / "dataset_info.json", "w") as fh:
            json.dump(info, fh, indent=2)

        logger.info(
            f"Splits saved to {d} | "
            f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
        )

    def _load_splits_from_disk(
        self,
    ) -> tuple[DataLoader, DataLoader, DataLoader, SequenceScaler]:
        """Load pre-built splits from data/splits/."""
        d = self.splits_dir
        X_train = np.load(d / "X_train.npy")
        y_train = np.load(d / "y_train.npy")
        X_val   = np.load(d / "X_val.npy")
        y_val   = np.load(d / "y_val.npy")
        X_test  = np.load(d / "X_test.npy")
        y_test  = np.load(d / "y_test.npy")
        scaler  = SequenceScaler.load(d)

        logger.info(
            f"Loaded from disk | "
            f"train={len(X_train)} val={len(X_val)} test={len(X_test)}"
        )
        return self._make_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test, scaler
        )

    def _make_dataloaders(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
        X_test:  np.ndarray, y_test:  np.ndarray,
        scaler:  SequenceScaler,
    ) -> tuple[DataLoader, DataLoader, DataLoader, SequenceScaler]:
        """Wrap arrays in ISLDataset and return DataLoaders."""
        train_ds = ISLDataset(X_train, y_train)
        val_ds   = ISLDataset(X_val,   y_val)
        test_ds  = ISLDataset(X_test,  y_test)

        # Windows multiprocessing requires num_workers=0
        # On Linux/macOS, use configured num_workers for faster loading
        import sys as _sys
        safe_workers = 0 if _sys.platform == "win32" else self.num_workers

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=safe_workers,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.val_batch,
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.val_batch,
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=False,
        )

        logger.info(
            f"DataLoaders ready | "
            f"train_batches={len(train_loader)} | "
            f"val_batches={len(val_loader)} | "
            f"test_batches={len(test_loader)}"
        )
        return train_loader, val_loader, test_loader, scaler


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    dataset_cfg:   dict,
    training_cfg:  dict,
    force_rebuild: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, SequenceScaler]:
    """
    Convenience function — build all DataLoaders from configs.

    Parameters
    ----------
    dataset_cfg   : parsed dataset.yaml
    training_cfg  : parsed training.yaml
    force_rebuild : bool — ignore cached splits

    Returns
    -------
    train_loader, val_loader, test_loader, scaler
    """
    builder = DatasetBuilder(dataset_cfg, training_cfg)
    return builder.build(force_rebuild=force_rebuild)
