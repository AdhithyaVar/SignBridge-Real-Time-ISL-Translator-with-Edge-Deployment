"""
augmentor.py
------------
Geometric and temporal augmentation pipeline for ISL landmark sequences.

Why augmentation is critical here
----------------------------------
With 100 sequences per class × 26 classes = 2,600 raw training samples,
the model would severely overfit.  Each raw sample generates 4 augmented
copies → ~13,000 training samples.  Augmentation also teaches the model
to be invariant to:
  * Small wrist rotation          (rotation augmentation)
  * Recording distance variation  (scale augmentation)
  * Hand position on screen       (translation augmentation)
  * Sensor noise                  (Gaussian noise)
  * Signing speed variation       (time warp)
  * Left-handed signers           (mirror flip, selected classes only)

All augmentations operate on NORMALISED sequences (after wrist-relative
normalization) so they are coordinate-system aware.

Input / Output shape: (T, 126) float32 — single sequence.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SINGLE_HAND_DIM  = 63
RIGHT_HAND_OFFSET = 0
LEFT_HAND_OFFSET  = 63
NUM_LANDMARKS    = 21
NUM_COORDS       = 3
_EPSILON         = 1e-8


# ---------------------------------------------------------------------------
# Individual augmentation functions
# All accept (T, 126) float32 and return (T, 126) float32
# All are deterministic given their parameters — randomness is in the caller
# ---------------------------------------------------------------------------

def rotate(
    sequence: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """
    Rotate the x-y plane of both hand slots by angle_deg degrees.
    Z (depth) is unaffected — depth is not rotationally meaningful
    for 2-D display gestures.

    Parameters
    ----------
    sequence  : (T, 126) float32
    angle_deg : float — rotation angle in degrees (+/- 15 recommended)
    """
    result  = sequence.copy()
    cos_a   = np.cos(np.radians(angle_deg))
    sin_a   = np.sin(np.radians(angle_deg))

    for hand_off in (RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET):
        for lm in range(NUM_LANDMARKS):
            base = hand_off + lm * NUM_COORDS
            x = result[:, base].copy()
            y = result[:, base + 1].copy()
            result[:, base]     = cos_a * x - sin_a * y
            result[:, base + 1] = sin_a * x + cos_a * y

    return result


def scale(
    sequence: np.ndarray,
    factor: float,
) -> np.ndarray:
    """
    Uniformly scale all landmark coordinates by factor.
    Applied to both hand slots together so relative scale is preserved.

    Parameters
    ----------
    sequence : (T, 126) float32
    factor   : float — scale multiplier (0.85–1.15 recommended)
    """
    return sequence * factor


def translate(
    sequence: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Shift x and y coordinates of both hands by dx, dy.
    Applied in normalised coordinate space.

    Parameters
    ----------
    sequence : (T, 126) float32
    dx, dy   : float — shift in normalised units (±0.05 recommended)
    """
    result = sequence.copy()
    for hand_off in (RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET):
        for lm in range(NUM_LANDMARKS):
            base = hand_off + lm * NUM_COORDS
            result[:, base]     += dx
            result[:, base + 1] += dy
    return result


def add_gaussian_noise(
    sequence: np.ndarray,
    std: float = 0.005,
) -> np.ndarray:
    """
    Add independent Gaussian noise to every coordinate.
    Only applied to non-zero entries (preserves absent hand zeros).

    Parameters
    ----------
    sequence : (T, 126) float32
    std      : float — noise standard deviation
    """
    noise  = np.random.normal(0.0, std, size=sequence.shape).astype(np.float32)
    mask   = (sequence != 0.0).astype(np.float32)   # (T, 126)
    return sequence + noise * mask


def time_warp(
    sequence: np.ndarray,
    warp_factor: float,
) -> np.ndarray:
    """
    Temporally stretch or compress a sequence then resample back to
    the original length using cubic interpolation.

    Parameters
    ----------
    sequence    : (T, 126) float32
    warp_factor : float — 1.0 = no change; >1 = slower; <1 = faster
                  Recommended range: [0.8, 1.2]
    """
    T, F   = sequence.shape
    t_orig = np.linspace(0, 1, T)

    # New number of frames before resampling
    T_new  = max(3, int(T * warp_factor))
    t_new  = np.linspace(0, 1, T_new)

    # Interpolate to warped length
    interp_fn = interp1d(t_orig, sequence, axis=0, kind="linear",
                         fill_value="extrapolate")
    warped = interp_fn(t_new)   # (T_new, F)

    # Resample back to original length T
    t_back    = np.linspace(0, 1, T)
    t_warped  = np.linspace(0, 1, T_new)
    resample  = interp1d(t_warped, warped, axis=0, kind="linear",
                         fill_value="extrapolate")
    return resample(t_back).astype(np.float32)   # (T, F)


def mirror_flip(sequence: np.ndarray) -> np.ndarray:
    """
    Horizontally mirror the sequence by negating all x coordinates
    and swapping the right-hand and left-hand slots.

    Used only for classes where the mirror image is a valid sign of
    the same class (symmetric letters: C, I, L, O, U, V, B, M, N, W).

    Parameters
    ----------
    sequence : (T, 126) float32
    """
    result = sequence.copy()

    # Negate all x coordinates (every 3rd value starting at 0 and 63)
    for hand_off in (RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET):
        for lm in range(NUM_LANDMARKS):
            base = hand_off + lm * NUM_COORDS
            result[:, base] = -result[:, base]   # negate x

    # Swap right and left slots so handedness is consistent
    right_copy = result[:, RIGHT_HAND_OFFSET: RIGHT_HAND_OFFSET + SINGLE_HAND_DIM].copy()
    left_copy  = result[:, LEFT_HAND_OFFSET:  LEFT_HAND_OFFSET  + SINGLE_HAND_DIM].copy()
    result[:, RIGHT_HAND_OFFSET: RIGHT_HAND_OFFSET + SINGLE_HAND_DIM] = left_copy
    result[:, LEFT_HAND_OFFSET:  LEFT_HAND_OFFSET  + SINGLE_HAND_DIM] = right_copy

    return result


# ---------------------------------------------------------------------------
# Augmentor — orchestrates all augmentations
# ---------------------------------------------------------------------------

class Augmentor:
    """
    Applies a configurable pipeline of random augmentations to
    a single ISL landmark sequence.

    Parameters
    ----------
    rotation_range     : float  — max rotation angle in degrees (±)
    scale_range        : float  — max scale deviation from 1.0 (±)
    translation_range  : float  — max translation in normalised units (±)
    noise_std          : float  — Gaussian noise standard deviation
    time_warp_enabled  : bool
    time_warp_range    : float  — max warp factor deviation from 1.0 (±)
    mirror_enabled     : bool
    mirror_classes     : set[str] — classes where mirroring is safe
    random_seed        : int | None
    """

    def __init__(
        self,
        rotation_range:    float       = 15.0,
        scale_range:       float       = 0.15,
        translation_range: float       = 0.05,
        noise_std:         float       = 0.005,
        time_warp_enabled: bool        = True,
        time_warp_range:   float       = 0.20,
        mirror_enabled:    bool        = True,
        mirror_classes:    set[str]    = None,
        random_seed:       int | None  = None,
    ) -> None:
        self.rotation_range    = rotation_range
        self.scale_range       = scale_range
        self.translation_range = translation_range
        self.noise_std         = noise_std
        self.time_warp_enabled = time_warp_enabled
        self.time_warp_range   = time_warp_range
        self.mirror_enabled    = mirror_enabled
        self.mirror_classes    = mirror_classes or {
            "C", "I", "L", "O", "U", "V",
            "B", "M", "N", "W",
        }
        if random_seed is not None:
            np.random.seed(random_seed)

    def augment(
        self,
        sequence:   np.ndarray,
        class_name: str,
    ) -> np.ndarray:
        """
        Apply one random combination of augmentations to a sequence.

        Every augmentation is applied independently with its own random
        parameters.  This produces maximum diversity per call.

        Parameters
        ----------
        sequence   : np.ndarray, shape (T, 126) float32
        class_name : str — used to gate mirror augmentation

        Returns
        -------
        np.ndarray, shape (T, 126) float32
        """
        seq = sequence.copy()

        # 1. Rotation (always applied, random angle)
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        seq   = rotate(seq, angle)

        # 2. Scale (always applied)
        factor = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
        seq    = scale(seq, factor)

        # 3. Translation (always applied)
        dx = np.random.uniform(-self.translation_range, self.translation_range)
        dy = np.random.uniform(-self.translation_range, self.translation_range)
        seq = translate(seq, dx, dy)

        # 4. Gaussian noise (always applied)
        seq = add_gaussian_noise(seq, std=self.noise_std)

        # 5. Time warp (optional, ~70% probability)
        if self.time_warp_enabled and np.random.random() < 0.70:
            warp = np.random.uniform(
                1.0 - self.time_warp_range,
                1.0 + self.time_warp_range,
            )
            seq = time_warp(seq, warp)

        # 6. Mirror flip (optional, only for symmetric classes, ~50% probability)
        if (
            self.mirror_enabled
            and class_name in self.mirror_classes
            and np.random.random() < 0.50
        ):
            seq = mirror_flip(seq)

        return seq.astype(np.float32)

    def generate_copies(
        self,
        sequence:    np.ndarray,
        class_name:  str,
        n_copies:    int = 4,
    ) -> list[np.ndarray]:
        """
        Generate n_copies independently augmented versions of one sequence.

        Parameters
        ----------
        sequence   : (T, 126) float32
        class_name : str
        n_copies   : int

        Returns
        -------
        list of n_copies arrays, each shape (T, 126) float32
        """
        return [self.augment(sequence, class_name) for _ in range(n_copies)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_augmentor(dataset_cfg: dict) -> Augmentor:
    """
    Build an Augmentor from a parsed dataset.yaml config dict.
    """
    aug_cfg = dataset_cfg.get("augmentation", {})
    mirror_classes = set(aug_cfg.get("mirror_classes", []))

    return Augmentor(
        rotation_range=    float(aug_cfg.get("rotation_range",    15.0)),
        scale_range=       float(aug_cfg.get("scale_range",        0.15)),
        translation_range= float(aug_cfg.get("translation_range",  0.05)),
        noise_std=         float(aug_cfg.get("gaussian_noise_std", 0.005)),
        time_warp_enabled= bool( aug_cfg.get("time_warp",          True)),
        time_warp_range=   0.20,
        mirror_enabled=    bool( aug_cfg.get("mirror",             True)),
        mirror_classes=    mirror_classes,
    )
