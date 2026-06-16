"""
normalizer.py
-------------
Dataset-level statistical normalization (Z-score / StandardScaler).

Why this is needed
------------------
After wrist-relative normalization the coordinate values are dimensionless
ratios.  However their distribution across the dataset is not zero-centred
and the variance differs between coordinates.  A StandardScaler fitted on
the training set brings all features to mean≈0, std≈1 which:
  * Helps gradient descent converge faster
  * Prevents any coordinate from dominating due to magnitude alone
  * Is the standard practice for LSTM / CNN inputs

Critical rule: fit ONLY on training data, then apply the same transform
to validation and test sets (prevents data leakage).

Saved artefact: data/splits/scaler_mean.npy  +  data/splits/scaler_std.npy
These are reloaded at inference time to normalise live webcam frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# SequenceScaler
# ---------------------------------------------------------------------------

class SequenceScaler:
    """
    Z-score scaler for (N, T, F) landmark sequence arrays.

    Statistics are computed over the N and T axes independently per
    feature dimension F — i.e. mean and std have shape (F,) = (126,).

    Parameters
    ----------
    eps : float
        Small constant added to std to prevent zero division.
    """

    def __init__(self, eps: float = _EPSILON) -> None:
        self.eps   = eps
        self.mean_ : np.ndarray | None = None   # shape (126,)
        self.std_  : np.ndarray | None = None   # shape (126,)
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SequenceScaler":
        """
        Compute mean and std from training data.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, 126)
            Training sequences AFTER wrist-relative normalization.

        Returns
        -------
        self
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D array (N, T, F), got shape {X.shape}.")

        # Reshape to (N*T, F) to compute statistics over all frames
        N, T, F = X.shape
        flat = X.reshape(-1, F)   # (N*T, F)

        self.mean_   = flat.mean(axis=0).astype(np.float32)   # (F,)
        self.std_    = flat.std(axis=0).astype(np.float32)    # (F,)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Z-score normalization using fitted statistics.

        Parameters
        ----------
        X : np.ndarray, shape (N, T, 126)  OR  (T, 126)  OR  (126,)

        Returns
        -------
        np.ndarray, same shape as input, float32.
        """
        self._check_fitted()
        X = X.astype(np.float32)
        std_safe = np.where(self.std_ < self.eps, 1.0, self.std_)

        # Works for any leading dimensions via broadcasting
        return (X - self.mean_) / std_safe

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the Z-score transform."""
        self._check_fitted()
        std_safe = np.where(self.std_ < self.eps, 1.0, self.std_)
        return X.astype(np.float32) * std_safe + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: Path) -> None:
        """
        Save mean and std arrays to disk.

        Parameters
        ----------
        save_dir : Path
            Directory where scaler_mean.npy and scaler_std.npy are written.
        """
        self._check_fitted()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "scaler_mean.npy", self.mean_)
        np.save(save_dir / "scaler_std.npy",  self.std_)

    @classmethod
    def load(cls, save_dir: Path) -> "SequenceScaler":
        """
        Load a previously saved scaler from disk.

        Parameters
        ----------
        save_dir : Path
            Directory containing scaler_mean.npy and scaler_std.npy.

        Returns
        -------
        SequenceScaler (fitted)

        Raises
        ------
        FileNotFoundError if scaler files are missing.
        """
        save_dir  = Path(save_dir)
        mean_path = save_dir / "scaler_mean.npy"
        std_path  = save_dir / "scaler_std.npy"

        for p in (mean_path, std_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Scaler file not found: {p}\n"
                    f"Run  python scripts/preprocess.py  first."
                )

        scaler         = cls()
        scaler.mean_   = np.load(mean_path).astype(np.float32)
        scaler.std_    = np.load(std_path).astype(np.float32)
        scaler._fitted = True
        return scaler

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "SequenceScaler is not fitted. "
                "Call .fit(X_train) before .transform()."
            )

    def __repr__(self) -> str:
        if self._fitted:
            return (
                f"SequenceScaler(fitted=True, "
                f"mean_range=[{self.mean_.min():.4f}, {self.mean_.max():.4f}], "
                f"std_range=[{self.std_.min():.4f}, {self.std_.max():.4f}])"
            )
        return "SequenceScaler(fitted=False)"
