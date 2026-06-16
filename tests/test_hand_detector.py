"""
test_hand_detector.py
---------------------
Unit tests for src/collection/hand_detector.py

Run:
    pytest tests/test_hand_detector.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# We mock mediapipe before importing HandDetector so tests run without a camera
# ---------------------------------------------------------------------------
import sys

# Build minimal mediapipe mock
_mp_mock = MagicMock()
_mp_mock.solutions.hands.Hands.return_value = MagicMock()
_mp_mock.solutions.hands.HAND_CONNECTIONS = []
_mp_mock.solutions.drawing_utils.draw_landmarks = MagicMock()
_mp_mock.solutions.drawing_styles.get_default_hand_landmarks_style = MagicMock()
_mp_mock.solutions.drawing_styles.get_default_hand_connections_style = MagicMock()
sys.modules["mediapipe"] = _mp_mock
sys.modules["mediapipe.solutions"] = _mp_mock.solutions
sys.modules["mediapipe.solutions.hands"] = _mp_mock.solutions.hands
sys.modules["mediapipe.solutions.drawing_utils"] = _mp_mock.solutions.drawing_utils
sys.modules["mediapipe.solutions.drawing_styles"] = _mp_mock.solutions.drawing_styles

from src.collection.hand_detector import HandDetector, HandLandmarks, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_landmark_vector(value: float = 0.5) -> np.ndarray:
    """Return a 63-dim float32 vector with all values set to `value`."""
    return np.full(63, value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: HandDetector._landmarks_to_vector
# ---------------------------------------------------------------------------

class TestLandmarksToVector:

    def test_returns_correct_shape(self):
        lm_list = MagicMock()
        lm_list.landmark = [MagicMock(x=0.1, y=0.2, z=0.3) for _ in range(21)]
        vec = HandDetector._landmarks_to_vector(lm_list)
        assert vec.shape == (63,), f"Expected (63,), got {vec.shape}"

    def test_returns_float32(self):
        lm_list = MagicMock()
        lm_list.landmark = [MagicMock(x=0.0, y=0.0, z=0.0) for _ in range(21)]
        vec = HandDetector._landmarks_to_vector(lm_list)
        assert vec.dtype == np.float32

    def test_values_are_correct(self):
        lm_list = MagicMock()
        lm_list.landmark = [MagicMock(x=float(i), y=float(i)+0.1, z=float(i)+0.2)
                            for i in range(21)]
        vec = HandDetector._landmarks_to_vector(lm_list)
        assert vec[0] == pytest.approx(0.0)
        assert vec[1] == pytest.approx(0.1)
        assert vec[2] == pytest.approx(0.2)
        assert vec[60] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Tests: HandDetector._build_feature_vector
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:

    def test_no_hands_returns_zeros(self):
        vec = HandDetector._build_feature_vector([])
        assert vec.shape == (126,)
        assert np.all(vec == 0.0)

    def test_right_hand_only_fills_first_slot(self):
        right_vec = _make_landmark_vector(1.0)
        hand = HandLandmarks(
            vector=right_vec, handedness="Right",
            raw_landmarks=None, confidence=0.9
        )
        result = HandDetector._build_feature_vector([hand])
        assert result.shape == (126,)
        np.testing.assert_array_equal(result[:63],  right_vec)
        np.testing.assert_array_equal(result[63:], np.zeros(63))

    def test_left_hand_only_fills_second_slot(self):
        left_vec = _make_landmark_vector(2.0)
        hand = HandLandmarks(
            vector=left_vec, handedness="Left",
            raw_landmarks=None, confidence=0.9
        )
        result = HandDetector._build_feature_vector([hand])
        assert result.shape == (126,)
        np.testing.assert_array_equal(result[:63],  np.zeros(63))
        np.testing.assert_array_equal(result[63:], left_vec)

    def test_both_hands_fills_both_slots(self):
        right_vec = _make_landmark_vector(1.0)
        left_vec  = _make_landmark_vector(2.0)
        hands = [
            HandLandmarks(vector=right_vec, handedness="Right",
                          raw_landmarks=None, confidence=0.95),
            HandLandmarks(vector=left_vec,  handedness="Left",
                          raw_landmarks=None, confidence=0.90),
        ]
        result = HandDetector._build_feature_vector(hands)
        np.testing.assert_array_equal(result[:63],  right_vec)
        np.testing.assert_array_equal(result[63:], left_vec)

    def test_output_dtype_is_float32(self):
        result = HandDetector._build_feature_vector([])
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: HandDetector.is_correct_hand_count
# ---------------------------------------------------------------------------

class TestIsCorrectHandCount:

    def setup_method(self):
        # Create detector without actually calling MediaPipe constructor
        with patch("src.collection.hand_detector._mp_hands.Hands"):
            self.detector = HandDetector()

    def _detection(self, num_hands: int) -> DetectionResult:
        d = DetectionResult()
        d.num_hands = num_hands
        d.is_valid  = num_hands > 0
        return d

    # Single-hand sign (alphabet)
    def test_single_hand_sign_with_one_hand(self):
        assert self.detector.is_correct_hand_count(self._detection(1), "A") is True

    def test_single_hand_sign_with_two_hands(self):
        assert self.detector.is_correct_hand_count(self._detection(2), "A") is False

    def test_single_hand_sign_with_no_hands(self):
        assert self.detector.is_correct_hand_count(self._detection(0), "A") is False

    # Two-hand sign (word)
    def test_two_hand_sign_with_two_hands(self):
        assert self.detector.is_correct_hand_count(self._detection(2), "hello") is True

    def test_two_hand_sign_with_one_hand(self):
        assert self.detector.is_correct_hand_count(self._detection(1), "hello") is False

    def test_two_hand_sign_with_no_hands(self):
        assert self.detector.is_correct_hand_count(self._detection(0), "hello") is False
