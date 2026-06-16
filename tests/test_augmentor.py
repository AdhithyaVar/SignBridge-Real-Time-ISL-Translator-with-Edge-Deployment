"""
test_augmentor.py
-----------------
Unit tests for preprocessing pipeline.

Covers:
  - landmark_extractor.normalize_sequence
  - augmentor: rotate, scale, translate, noise, time_warp, mirror_flip
  - normalizer.SequenceScaler
  - Output shapes and dtypes
  - Numerical invariants

Run:
    pytest tests/test_augmentor.py -v
"""

import numpy as np
import pytest

from src.preprocessing.landmark_extractor import (
    normalize_sequence,
    normalize_dataset,
    FEATURE_DIM,
    SEQUENCE_LEN,
    NUM_LANDMARKS,
    NUM_COORDS,
)
from src.preprocessing.augmentor import (
    rotate,
    scale,
    translate,
    add_gaussian_noise,
    time_warp,
    mirror_flip,
    Augmentor,
    RIGHT_HAND_OFFSET,
    LEFT_HAND_OFFSET,
    SINGLE_HAND_DIM,
)
from src.preprocessing.normalizer import SequenceScaler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_seq() -> np.ndarray:
    """All-zero sequence — represents 0 hands detected."""
    return np.zeros((SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)


@pytest.fixture
def random_seq() -> np.ndarray:
    """Random sequence with realistic MediaPipe coordinate magnitudes."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.1, 0.9, size=(SEQUENCE_LEN, FEATURE_DIM)).astype(np.float32)


@pytest.fixture
def single_hand_seq() -> np.ndarray:
    """Sequence with right hand only (left slot zeroed)."""
    rng = np.random.default_rng(7)
    seq = np.zeros((SEQUENCE_LEN, FEATURE_DIM), dtype=np.float32)
    seq[:, RIGHT_HAND_OFFSET: RIGHT_HAND_OFFSET + SINGLE_HAND_DIM] = \
        rng.uniform(0.1, 0.9, (SEQUENCE_LEN, SINGLE_HAND_DIM)).astype(np.float32)
    return seq


# ---------------------------------------------------------------------------
# Tests: normalize_sequence
# ---------------------------------------------------------------------------

class TestNormalizeSequence:

    def test_output_shape(self, random_seq):
        out = normalize_sequence(random_seq)
        assert out.shape == random_seq.shape

    def test_output_dtype_float32(self, random_seq):
        out = normalize_sequence(random_seq)
        assert out.dtype == np.float32

    def test_zero_sequence_unchanged(self, zero_seq):
        out = normalize_sequence(zero_seq)
        np.testing.assert_array_equal(out, zero_seq)

    def test_wrist_is_origin_after_normalization(self, single_hand_seq):
        out = normalize_sequence(single_hand_seq)
        # Right hand wrist (landmark 0) should be ~0 after subtraction
        wrist_x = out[:, RIGHT_HAND_OFFSET + 0]
        wrist_y = out[:, RIGHT_HAND_OFFSET + 1]
        wrist_z = out[:, RIGHT_HAND_OFFSET + 2]
        np.testing.assert_allclose(wrist_x, 0.0, atol=1e-6)
        np.testing.assert_allclose(wrist_y, 0.0, atol=1e-6)
        np.testing.assert_allclose(wrist_z, 0.0, atol=1e-6)

    def test_left_slot_stays_zero_for_single_hand(self, single_hand_seq):
        out = normalize_sequence(single_hand_seq)
        left_slot = out[:, LEFT_HAND_OFFSET: LEFT_HAND_OFFSET + SINGLE_HAND_DIM]
        np.testing.assert_array_equal(left_slot, 0.0)

    def test_invalid_shape_raises(self):
        bad = np.zeros((30, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected shape"):
            normalize_sequence(bad)

    def test_normalize_dataset_shape(self, random_seq):
        batch = np.stack([random_seq] * 5, axis=0)   # (5, 30, 126)
        out   = normalize_dataset(batch)
        assert out.shape == (5, SEQUENCE_LEN, FEATURE_DIM)


# ---------------------------------------------------------------------------
# Tests: rotate
# ---------------------------------------------------------------------------

class TestRotate:

    def test_shape_preserved(self, random_seq):
        out = rotate(random_seq, 10.0)
        assert out.shape == random_seq.shape

    def test_zero_rotation_is_identity(self, random_seq):
        out = rotate(random_seq, 0.0)
        np.testing.assert_allclose(out, random_seq, atol=1e-6)

    def test_360_rotation_returns_original(self, random_seq):
        out = rotate(random_seq, 360.0)
        np.testing.assert_allclose(out, random_seq, atol=1e-5)

    def test_z_coord_unchanged(self, random_seq):
        out = rotate(random_seq, 45.0)
        # Z coordinate (every 3rd starting at index 2) should be unchanged
        for hand_off in (RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET):
            for lm in range(NUM_LANDMARKS):
                z_idx = hand_off + lm * NUM_COORDS + 2
                np.testing.assert_allclose(
                    out[:, z_idx], random_seq[:, z_idx], atol=1e-6
                )


# ---------------------------------------------------------------------------
# Tests: scale
# ---------------------------------------------------------------------------

class TestScale:

    def test_shape_preserved(self, random_seq):
        assert scale(random_seq, 1.1).shape == random_seq.shape

    def test_identity_scale(self, random_seq):
        np.testing.assert_allclose(scale(random_seq, 1.0), random_seq, atol=1e-7)

    def test_scale_changes_magnitude(self, random_seq):
        out = scale(random_seq, 2.0)
        np.testing.assert_allclose(out, random_seq * 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: translate
# ---------------------------------------------------------------------------

class TestTranslate:

    def test_shape_preserved(self, random_seq):
        assert translate(random_seq, 0.01, 0.01).shape == random_seq.shape

    def test_zero_translate_is_identity(self, random_seq):
        out = translate(random_seq, 0.0, 0.0)
        np.testing.assert_allclose(out, random_seq, atol=1e-7)

    def test_z_coord_unchanged(self, random_seq):
        out = translate(random_seq, 0.05, -0.03)
        for hand_off in (RIGHT_HAND_OFFSET, LEFT_HAND_OFFSET):
            for lm in range(NUM_LANDMARKS):
                z_idx = hand_off + lm * NUM_COORDS + 2
                np.testing.assert_allclose(
                    out[:, z_idx], random_seq[:, z_idx], atol=1e-7
                )


# ---------------------------------------------------------------------------
# Tests: add_gaussian_noise
# ---------------------------------------------------------------------------

class TestGaussianNoise:

    def test_shape_preserved(self, random_seq):
        assert add_gaussian_noise(random_seq).shape == random_seq.shape

    def test_zero_noise_is_identity(self, random_seq):
        np.random.seed(0)
        out = add_gaussian_noise(random_seq, std=0.0)
        np.testing.assert_allclose(out, random_seq, atol=1e-7)

    def test_zero_slots_stay_zero(self, single_hand_seq):
        np.random.seed(42)
        out = add_gaussian_noise(single_hand_seq, std=0.1)
        left = out[:, LEFT_HAND_OFFSET: LEFT_HAND_OFFSET + SINGLE_HAND_DIM]
        np.testing.assert_array_equal(left, 0.0)


# ---------------------------------------------------------------------------
# Tests: time_warp
# ---------------------------------------------------------------------------

class TestTimeWarp:

    def test_shape_preserved(self, random_seq):
        out = time_warp(random_seq, 1.2)
        assert out.shape == random_seq.shape

    def test_dtype_float32(self, random_seq):
        out = time_warp(random_seq, 0.9)
        assert out.dtype == np.float32

    def test_warp_factor_1_close_to_identity(self, random_seq):
        out = time_warp(random_seq, 1.0)
        np.testing.assert_allclose(out, random_seq, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: mirror_flip
# ---------------------------------------------------------------------------

class TestMirrorFlip:

    def test_shape_preserved(self, random_seq):
        assert mirror_flip(random_seq).shape == random_seq.shape

    def test_double_flip_is_identity(self, random_seq):
        out = mirror_flip(mirror_flip(random_seq))
        np.testing.assert_allclose(out, random_seq, atol=1e-6)

    def test_x_coords_negated(self, single_hand_seq):
        out = mirror_flip(single_hand_seq)
        # After flip, original right-hand x values should appear negated
        # in left-hand slot (slots are swapped)
        for lm in range(NUM_LANDMARKS):
            orig_x   = single_hand_seq[:, RIGHT_HAND_OFFSET + lm * NUM_COORDS]
            flipped_x = out[:, LEFT_HAND_OFFSET + lm * NUM_COORDS]
            np.testing.assert_allclose(flipped_x, -orig_x, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Augmentor
# ---------------------------------------------------------------------------

class TestAugmentor:

    def setup_method(self):
        self.aug = Augmentor(random_seed=42)

    def test_output_shape(self, random_seq):
        out = self.aug.augment(random_seq, "A")
        assert out.shape == random_seq.shape

    def test_output_dtype(self, random_seq):
        out = self.aug.augment(random_seq, "A")
        assert out.dtype == np.float32

    def test_augmented_differs_from_original(self, random_seq):
        out = self.aug.augment(random_seq, "A")
        assert not np.allclose(out, random_seq), "Augmented copy should differ from original"

    def test_generate_copies_count(self, random_seq):
        copies = self.aug.generate_copies(random_seq, "C", n_copies=4)
        assert len(copies) == 4

    def test_copies_are_distinct(self, random_seq):
        copies = self.aug.generate_copies(random_seq, "A", n_copies=4)
        for i in range(len(copies)):
            for j in range(i + 1, len(copies)):
                assert not np.allclose(copies[i], copies[j]), \
                    f"Copies {i} and {j} are identical — seed leak"


# ---------------------------------------------------------------------------
# Tests: SequenceScaler
# ---------------------------------------------------------------------------

class TestSequenceScaler:

    def _make_data(self, N=50) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.uniform(-1, 1, (N, SEQUENCE_LEN, FEATURE_DIM)).astype(np.float32)

    def test_fit_sets_statistics(self):
        X      = self._make_data()
        scaler = SequenceScaler()
        scaler.fit(X)
        assert scaler.mean_.shape == (FEATURE_DIM,)
        assert scaler.std_.shape  == (FEATURE_DIM,)
        assert scaler.is_fitted

    def test_transform_output_shape(self):
        X      = self._make_data()
        scaler = SequenceScaler()
        X_t    = scaler.fit_transform(X)
        assert X_t.shape == X.shape

    def test_transform_approx_zero_mean(self):
        X      = self._make_data(N=200)
        scaler = SequenceScaler()
        X_t    = scaler.fit_transform(X)
        assert abs(X_t.mean()) < 0.1

    def test_inverse_transform_roundtrip(self):
        X      = self._make_data()
        scaler = SequenceScaler()
        X_t    = scaler.fit_transform(X)
        X_back = scaler.inverse_transform(X_t)
        np.testing.assert_allclose(X_back, X, atol=1e-5)

    def test_unfitted_raises(self):
        scaler = SequenceScaler()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(np.zeros((5, SEQUENCE_LEN, FEATURE_DIM)))

    def test_save_and_load(self, tmp_path):
        X      = self._make_data()
        scaler = SequenceScaler()
        scaler.fit(X)
        scaler.save(tmp_path)
        loaded = SequenceScaler.load(tmp_path)
        np.testing.assert_array_equal(scaler.mean_, loaded.mean_)
        np.testing.assert_array_equal(scaler.std_,  loaded.std_)
