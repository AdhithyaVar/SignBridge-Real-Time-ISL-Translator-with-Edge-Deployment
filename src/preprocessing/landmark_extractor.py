"""
landmark_extractor.py
---------------------
Wrist-relative normalization of raw MediaPipe landmark sequences.

Why this is needed
------------------
Raw MediaPipe coordinates are absolute pixel positions normalised to [0,1]
relative to the camera frame.  Two identical signs recorded at different
screen positions, distances, or hand sizes produce completely different
raw vectors — the model would memorise positions rather than shapes.

After wrist-relative normalization:
  * Translation invariant  — sign position on screen does not matter
  * Scale invariant        — hand size / camera distance does not matter
  * Only finger configuration and motion pattern remain

Algorithm (per frame, per hand)
--------------------------------
1. Extract wrist (landmark 0) as the origin.
2. Subtract wrist from all 21 landmarks → translation invariant.
3. Compute reference scale = Euclidean distance from wrist to
   middle-finger MCP (landmark 9).  This is proportional to hand size.
4. Divide all landmarks by scale → scale invariant.
5. If hand slot is all-zeros (absent hand), leave it as zeros.

Input  : (T, 126)  — raw 30-frame sequence
Output : (T, 126)  — normalised sequence, same shape
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQUENCE_LEN    = 30
FEATURE_DIM     = 126   # 2 hands × 21 landmarks × 3 coords
SINGLE_HAND_DIM = 63    # 21 × 3
NUM_LANDMARKS   = 21
NUM_COORDS      = 3

# MediaPipe landmark indices (per hand, 0-indexed)
WRIST_IDX          = 0    # landmark 0  — wrist
MIDDLE_MCP_IDX     = 9    # landmark 9  — middle finger MCP joint
MIDDLE_TIP_IDX     = 12   # landmark 12 — middle finger tip

# Hand slot offsets in the 126-dim vector
RIGHT_HAND_OFFSET = 0
LEFT_HAND_OFFSET  = 63

_EPSILON = 1e-8   # Prevent division by zero


# ---------------------------------------------------------------------------
# Core normalization functions
# ---------------------------------------------------------------------------

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply wrist-relative, scale-normalised transformation to one sequence.

    Parameters
    ----------
    sequence : np.ndarray, shape (T, 126)
        Raw landmark sequence from AutoRecorder.

    Returns
    -------
    np.ndarray, shape (T, 126)
        Normalised sequence.  Float32 dtype preserved.
    """
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Expected shape (T, {FEATURE_DIM}), got {sequence.shape}."
        )

    result = sequence.astype(np.float32).copy()

    for hand_off in (RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET):
        hand_slice = result[:, hand_off: hand_off + SINGLE_HAND_DIM]   # (T, 63)

        # Detect frames where hand is present (not all-zero)
        present = ~np.all(hand_slice == 0.0, axis=1)   # (T,) bool

        if not np.any(present):
            continue   # Entire hand slot is empty — leave as zeros

        # --- Step 1: Wrist-origin subtraction ---
        # Wrist coords for each frame: columns hand_off+0, hand_off+1, hand_off+2
        wrist = result[:, hand_off: hand_off + NUM_COORDS].copy()   # (T, 3)

        for lm in range(NUM_LANDMARKS):
            base = hand_off + lm * NUM_COORDS
            result[:, base: base + NUM_COORDS] -= wrist   # broadcast over frames

        # --- Step 2: Scale normalisation ---
        # Reference: distance wrist → middle-finger MCP (landmark 9)
        mcp_base = hand_off + MIDDLE_MCP_IDX * NUM_COORDS
        mcp_vec  = result[:, mcp_base: mcp_base + NUM_COORDS]   # (T, 3)
        scale    = np.linalg.norm(mcp_vec, axis=1, keepdims=True)   # (T, 1)
        scale    = np.where(scale < _EPSILON, 1.0, scale)            # safe divide

        for lm in range(NUM_LANDMARKS):
            base = hand_off + lm * NUM_COORDS
            result[:, base: base + NUM_COORDS] /= scale   # (T, 3) / (T, 1)

        # --- Step 3: Zero out absent frames ---
        result[:, hand_off: hand_off + SINGLE_HAND_DIM] *= present[:, np.newaxis]

    return result


def normalize_dataset(sequences: np.ndarray) -> np.ndarray:
    """
    Apply normalize_sequence to every sequence in a dataset array.

    Parameters
    ----------
    sequences : np.ndarray, shape (N, T, 126)

    Returns
    -------
    np.ndarray, shape (N, T, 126)
    """
    if sequences.ndim != 3 or sequences.shape[2] != FEATURE_DIM:
        raise ValueError(
            f"Expected shape (N, T, {FEATURE_DIM}), got {sequences.shape}."
        )

    out = np.empty_like(sequences, dtype=np.float32)
    for i, seq in enumerate(sequences):
        out[i] = normalize_sequence(seq)
    return out


# ---------------------------------------------------------------------------
# Loader: read raw .npy files for one class
# ---------------------------------------------------------------------------

def load_raw_class(
    raw_dir,
    class_name: str,
    sequence_length: int = SEQUENCE_LEN,
    feature_dim:     int = FEATURE_DIM,
) -> np.ndarray | None:
    """
    Load all .npy sequence files for one class from data/raw/<class_name>/.

    Parameters
    ----------
    raw_dir : Path
        Path to the data/raw/ directory.
    class_name : str
        ISL class label (e.g. 'A', 'hello').
    sequence_length : int
        Expected number of frames per sequence.
    feature_dim : int
        Expected feature dimension.

    Returns
    -------
    np.ndarray, shape (N, sequence_length, feature_dim)
        All valid sequences for this class, or None if directory is empty.
    """
    from pathlib import Path
    class_dir = Path(raw_dir) / class_name
    if not class_dir.exists():
        return None

    files = sorted(class_dir.glob(f"{class_name}_*.npy"))
    if not files:
        return None

    valid_seqs = []
    skipped    = 0

    for fp in files:
        try:
            seq = np.load(fp)
            if seq.shape == (sequence_length, feature_dim):
                valid_seqs.append(seq.astype(np.float32))
            else:
                skipped += 1
        except Exception:
            skipped += 1

    if skipped > 0:
        import warnings
        warnings.warn(
            f"[{class_name}] Skipped {skipped} files with unexpected shape."
        )

    if not valid_seqs:
        return None

    return np.stack(valid_seqs, axis=0)   # (N, T, 126)
