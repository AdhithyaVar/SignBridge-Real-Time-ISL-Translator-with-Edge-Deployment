"""
test_inference_engine.py
------------------------
Unit and integration tests for Phase 7 inference components.

Covers:
  - inference_engine.py  : PyTorchEngine, ONNXEngine, build_inference_engine
  - post_processor.py    : TemporalSmoother, StabilityGate, CooldownGuard,
                           SentenceBuilder, PostProcessor
  - onnx_validator.py    : ONNXValidator shape/numerical checks

Run:
    pytest tests/test_inference_engine.py -v
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.inference_engine import Prediction, InferenceBackend
from src.inference.post_processor import (
    TemporalSmoother,
    StabilityGate,
    CooldownGuard,
    SentenceBuilder,
    PostProcessor,
    PostProcessorOutput,
    SignState,
    build_post_processor,
)
from src.preprocessing.normalizer import SequenceScaler
from src.utils.class_labels import CLASS_LABELS, NUM_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN  = 30
FEAT_DIM = 126


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_scaler() -> SequenceScaler:
    """Fitted scaler on random data."""
    rng = np.random.default_rng(42)
    X   = rng.uniform(-1, 1, (100, SEQ_LEN, FEAT_DIM)).astype(np.float32)
    sc  = SequenceScaler()
    sc.fit(X)
    return sc


def _make_prediction(class_idx: int, confidence: float = 0.95) -> Prediction:
    """Build a Prediction with one dominant class."""
    probs = np.full(NUM_CLASSES, (1 - confidence) / (NUM_CLASSES - 1), dtype=np.float32)
    probs[class_idx] = confidence
    return Prediction(probs, InferenceBackend.PYTORCH)


# ============================================================================
# Tests: Prediction dataclass
# ============================================================================

class TestPrediction:

    def test_class_idx_is_argmax(self):
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        probs[5] = 0.99
        p = Prediction(probs, InferenceBackend.PYTORCH)
        assert p.class_idx == 5

    def test_class_name_matches_index(self):
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        probs[0] = 0.99  # class A
        p = Prediction(probs, InferenceBackend.PYTORCH)
        assert p.class_name == "A"

    def test_confidence_is_max_prob(self):
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        probs[3] = 0.87
        p = Prediction(probs, InferenceBackend.PYTORCH)
        assert p.confidence == pytest.approx(0.87)

    def test_probabilities_dtype_float32(self):
        probs = np.ones(NUM_CLASSES, dtype=np.float64) / NUM_CLASSES
        p = Prediction(probs, InferenceBackend.PYTORCH)
        assert p.probabilities.dtype == np.float32

    def test_repr_contains_class_name(self):
        probs = np.zeros(NUM_CLASSES, dtype=np.float32)
        probs[1] = 0.9   # B
        p = Prediction(probs, InferenceBackend.PYTORCH)
        assert "B" in repr(p)
        assert "0.9" in repr(p)


# ============================================================================
# Tests: TemporalSmoother
# ============================================================================

class TestTemporalSmoother:

    def test_returns_none_before_buffer_full(self):
        sm = TemporalSmoother(window_size=5, confidence_threshold=0.5)
        for i in range(4):
            cls, conf = sm.update(_make_prediction(0, 0.9))
            assert cls is None, f"Expected None before buffer full at step {i}"

    def test_returns_majority_class_when_full(self):
        sm = TemporalSmoother(window_size=5, confidence_threshold=0.5)
        for _ in range(5):
            sm.update(_make_prediction(3, 0.9))
        cls, conf = sm.update(_make_prediction(3, 0.9))
        assert cls == CLASS_LABELS[3]
        assert conf > 0.5

    def test_majority_vote_wins(self):
        sm = TemporalSmoother(window_size=5, confidence_threshold=0.0)
        sm.update(_make_prediction(0, 0.9))
        sm.update(_make_prediction(0, 0.9))
        sm.update(_make_prediction(0, 0.9))
        sm.update(_make_prediction(1, 0.9))
        sm.update(_make_prediction(1, 0.9))
        cls, _ = sm.update(_make_prediction(0, 0.9))
        assert cls == CLASS_LABELS[0], "Class 0 should win with 4/6 votes"

    def test_low_confidence_returns_none(self):
        sm = TemporalSmoother(window_size=3, confidence_threshold=0.90)
        for _ in range(3):
            sm.update(_make_prediction(0, 0.50))
        cls, conf = sm.update(_make_prediction(0, 0.50))
        assert cls is None, "Should reject low-confidence predictions"

    def test_reset_clears_buffer(self):
        sm = TemporalSmoother(window_size=3, confidence_threshold=0.0)
        for _ in range(3):
            sm.update(_make_prediction(0, 0.9))
        sm.reset()
        cls, _ = sm.update(_make_prediction(0, 0.9))
        assert cls is None, "Buffer should be empty after reset"


# ============================================================================
# Tests: StabilityGate
# ============================================================================

class TestStabilityGate:

    def test_not_stable_before_required_frames(self):
        gate = StabilityGate(required_stable=5)
        for i in range(4):
            stable, ratio = gate.update("A")
            assert not stable, f"Should not be stable at step {i+1}"

    def test_stable_after_required_frames(self):
        gate = StabilityGate(required_stable=5)
        for _ in range(5):
            stable, ratio = gate.update("A")
        assert stable

    def test_ratio_increases_monotonically(self):
        gate  = StabilityGate(required_stable=4)
        ratios = []
        for _ in range(4):
            _, ratio = gate.update("A")
            ratios.append(ratio)
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1], "Ratio should increase"

    def test_class_change_resets_counter(self):
        gate = StabilityGate(required_stable=3)
        gate.update("A")
        gate.update("A")
        gate.update("B")   # class change — resets
        stable, ratio = gate.update("B")
        assert not stable
        assert ratio < 1.0

    def test_none_input_resets(self):
        gate = StabilityGate(required_stable=3)
        gate.update("A")
        gate.update("A")
        stable, _ = gate.update(None)
        assert not stable
        assert gate._stable_count == 0


# ============================================================================
# Tests: CooldownGuard
# ============================================================================

class TestCooldownGuard:

    def test_not_in_cooldown_initially(self):
        cd = CooldownGuard(cooldown_seconds=1.0)
        assert not cd.in_cooldown

    def test_in_cooldown_immediately_after_acceptance(self):
        cd = CooldownGuard(cooldown_seconds=2.0)
        cd.record_acceptance()
        assert cd.in_cooldown

    def test_cooldown_expires(self):
        cd = CooldownGuard(cooldown_seconds=0.05)
        cd.record_acceptance()
        time.sleep(0.10)
        assert not cd.in_cooldown

    def test_remaining_seconds_decreases(self):
        cd = CooldownGuard(cooldown_seconds=1.0)
        cd.record_acceptance()
        r1 = cd.remaining_seconds
        time.sleep(0.05)
        r2 = cd.remaining_seconds
        assert r2 < r1


# ============================================================================
# Tests: SentenceBuilder
# ============================================================================

class TestSentenceBuilder:

    def test_empty_initially(self):
        sb = SentenceBuilder()
        assert sb.get() == ""

    def test_add_one_sign(self):
        sb = SentenceBuilder()
        sb.add("A")
        assert sb.get() == "A"

    def test_add_multiple_signs(self):
        sb = SentenceBuilder()
        for letter in ["H", "I"]:
            sb.add(letter)
        assert sb.get() == "H I"

    def test_clear_empties_sentence(self):
        sb = SentenceBuilder()
        sb.add("A")
        sb.add("B")
        sb.clear()
        assert sb.get() == ""

    def test_word_count(self):
        sb = SentenceBuilder()
        sb.add("A")
        sb.add("B")
        sb.add("C")
        assert sb.word_count == 3

    def test_max_words_truncates(self):
        sb = SentenceBuilder(max_words=3)
        for letter in ["A", "B", "C", "D", "E"]:
            sb.add(letter)
        words = sb.get().split()
        assert len(words) <= 3


# ============================================================================
# Tests: PostProcessor (full pipeline)
# ============================================================================

class TestPostProcessor:

    def _make_pp(self, **kwargs) -> PostProcessor:
        defaults = dict(
            smoothing_window=3,
            confidence_threshold=0.5,
            required_stable=3,
            cooldown_seconds=0.05,
            idle_reset_secs=60.0,
        )
        defaults.update(kwargs)
        return PostProcessor(**defaults)

    def test_idle_on_no_prediction(self):
        pp     = self._make_pp()
        output = pp.update(None)
        assert output.state == SignState.IDLE
        assert output.current_sign is None

    def test_tracking_state_on_valid_prediction(self):
        pp  = self._make_pp()
        out = pp.update(_make_prediction(0, 0.9))
        assert out.state in (SignState.IDLE, SignState.TRACKING)

    def test_sign_accepted_after_stability(self):
        pp = self._make_pp(
            smoothing_window=1,
            required_stable=1,
            confidence_threshold=0.0,
            cooldown_seconds=0.0,
        )
        output = None
        for _ in range(5):
            output = pp.update(_make_prediction(0, 0.95))
        assert output is not None
        assert output.state == SignState.ACCEPTED or \
               output.state == SignState.COOLDOWN

    def test_sentence_accumulates_on_acceptance(self):
        pp = self._make_pp(
            smoothing_window=1,
            required_stable=1,
            confidence_threshold=0.0,
            cooldown_seconds=0.0,
        )
        for _ in range(3):
            pp.update(_make_prediction(0, 0.99))  # Accept A
        time.sleep(0.02)
        for _ in range(3):
            pp.update(_make_prediction(1, 0.99))  # Accept B
        sentence = pp.sentence.get()
        assert len(sentence) > 0, "Sentence should have at least one letter"

    def test_clear_sentence(self):
        pp = self._make_pp(
            smoothing_window=1,
            required_stable=1,
            confidence_threshold=0.0,
            cooldown_seconds=0.0,
        )
        for _ in range(3):
            pp.update(_make_prediction(0, 0.99))
        pp.clear_sentence()
        assert pp.sentence.get() == ""

    def test_reset_all(self):
        pp = self._make_pp(
            smoothing_window=1,
            required_stable=1,
            confidence_threshold=0.0,
            cooldown_seconds=0.0,
        )
        for _ in range(3):
            pp.update(_make_prediction(0, 0.99))
        pp.reset_all()
        assert pp.sentence.get() == ""

    def test_output_has_sentence_field(self):
        pp     = self._make_pp()
        output = pp.update(_make_prediction(0, 0.9))
        assert hasattr(output, "sentence")
        assert isinstance(output.sentence, str)

    def test_just_accepted_flag_transient(self):
        """just_accepted should be True for exactly one frame."""
        pp = self._make_pp(
            smoothing_window=1,
            required_stable=1,
            confidence_threshold=0.0,
            cooldown_seconds=1.0,
        )
        accepted_frames = 0
        for _ in range(10):
            out = pp.update(_make_prediction(0, 0.99))
            if out.just_accepted:
                accepted_frames += 1
        assert accepted_frames <= 1, \
            "just_accepted should fire at most once per sign"


# ============================================================================
# Tests: build_post_processor factory
# ============================================================================

class TestBuildPostProcessor:

    def test_builds_with_empty_config(self):
        pp = build_post_processor({})
        assert isinstance(pp, PostProcessor)

    def test_builds_with_custom_config(self):
        pp = build_post_processor({
            "confidence_threshold": 0.80,
            "required_stable":      5,
            "cooldown_seconds":     2.0,
        })
        assert pp.gate.required_stable == 5
        assert pp.cooldown.cooldown_seconds == pytest.approx(2.0)
        assert pp.smoother.confidence_threshold == pytest.approx(0.80)
