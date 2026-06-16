"""
post_processor.py
-----------------
Confidence filtering, temporal smoothing, and sentence building
for the SignBridge real-time inference pipeline.

Problem being solved
--------------------
The model runs inference on every 30-frame window.  Without post-
processing, the output stream would be:
  A A A A A A A B B B A A A B B B B B ...
(repeated predictions for each overlapping window, with noise between signs)

Post-processing turns this into:
  A  →  B  →  (space)

Pipeline
--------
1. TemporalSmoother   — rolling window over last K predictions,
                        returns the majority-vote class and mean confidence.
                        Filters out single-frame noise.

2. StabilityGate      — requires the smoothed prediction to hold for
                        M consecutive smoothed windows before accepting.
                        Prevents rapid sign changes from being logged.

3. CooldownGuard      — enforces a minimum gap between accepted signs
                        so a held sign isn't logged multiple times.

4. SentenceBuilder    — accumulates accepted signs into a phrase.
                        Resets on long pause. Provides the display string.

5. PostProcessor      — orchestrates all of the above in one .update() call.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from src.inference.inference_engine import Prediction
from src.utils.class_labels import IDX_TO_CLASS
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Post-processor output state
# ---------------------------------------------------------------------------

class SignState(Enum):
    IDLE        = auto()   # No sign / low confidence
    TRACKING    = auto()   # Sign being held, not yet stable
    STABLE      = auto()   # Sign stable, ready to accept
    ACCEPTED    = auto()   # Sign just accepted → triggers TTS
    COOLDOWN    = auto()   # Brief pause after acceptance


@dataclass
class PostProcessorOutput:
    """
    Output of PostProcessor.update() — consumed by live_demo.py each frame.

    Attributes
    ----------
    state           : SignState — current state of the pipeline
    current_sign    : str | None — smoothed current prediction (or None)
    confidence      : float      — smoothed confidence [0, 1]
    just_accepted   : bool       — True on the single frame where sign accepted
    accepted_sign   : str | None — the sign that was just accepted
    sentence        : str        — full sentence built so far
    stability_ratio : float      — fraction of stable hold frames (0–1)
    """
    state:           SignState
    current_sign:    str | None = None
    confidence:      float      = 0.0
    just_accepted:   bool       = False
    accepted_sign:   str | None = None
    sentence:        str        = ""
    stability_ratio: float      = 0.0


# ---------------------------------------------------------------------------
# 1. Temporal smoother
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """
    Rolling window of recent predictions. Returns the majority-vote class
    and mean confidence from the last K predictions.

    Parameters
    ----------
    window_size        : int   — number of predictions to smooth over (5)
    confidence_threshold: float — minimum mean confidence to consider valid (0.70)
    """

    def __init__(
        self,
        window_size:         int   = 5,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.window_size          = window_size
        self.confidence_threshold = confidence_threshold
        self._buffer: deque[Prediction] = deque(maxlen=window_size)

    def update(self, prediction: Prediction) -> tuple[str | None, float]:
        """
        Add a prediction and return (smoothed_class, smoothed_confidence).
        Returns (None, 0.0) if buffer is not full or confidence too low.
        """
        self._buffer.append(prediction)

        if len(self._buffer) < self.window_size:
            return None, 0.0

        # Majority vote over buffered class indices
        class_indices = [p.class_idx for p in self._buffer]
        majority_idx  = max(set(class_indices), key=class_indices.count)
        majority_name = IDX_TO_CLASS[majority_idx]

        # Mean confidence only for the majority class frames
        mean_conf = sum(
            p.confidence for p in self._buffer if p.class_idx == majority_idx
        ) / max(class_indices.count(majority_idx), 1)

        if mean_conf < self.confidence_threshold:
            return None, 0.0

        return majority_name, mean_conf

    def reset(self) -> None:
        self._buffer.clear()


# ---------------------------------------------------------------------------
# 2. Stability gate
# ---------------------------------------------------------------------------

class StabilityGate:
    """
    Requires the same class to appear in M consecutive smoothed outputs
    before it is considered stable and ready to accept.

    Parameters
    ----------
    required_stable : int — consecutive identical smoothed predictions (8)
    """

    def __init__(self, required_stable: int = 8) -> None:
        self.required_stable = required_stable
        self._current_class  = None
        self._stable_count   = 0

    def update(self, smoothed_class: str | None) -> tuple[bool, float]:
        """
        Parameters
        ----------
        smoothed_class : str | None from TemporalSmoother

        Returns
        -------
        (is_stable, stability_ratio)
        is_stable      : True when stable_count >= required_stable
        stability_ratio: count / required_stable  (for progress bar in UI)
        """
        if smoothed_class is None:
            self._current_class = None
            self._stable_count  = 0
            return False, 0.0

        if smoothed_class == self._current_class:
            self._stable_count = min(
                self._stable_count + 1, self.required_stable
            )
        else:
            self._current_class = smoothed_class
            self._stable_count  = 1

        ratio = self._stable_count / self.required_stable
        return self._stable_count >= self.required_stable, ratio

    def reset(self) -> None:
        self._current_class = None
        self._stable_count  = 0


# ---------------------------------------------------------------------------
# 3. Cooldown guard
# ---------------------------------------------------------------------------

class CooldownGuard:
    """
    Enforces a minimum gap between accepted sign events so that a held
    sign is not accepted multiple times.

    Parameters
    ----------
    cooldown_seconds : float — seconds to wait after each acceptance (1.5)
    """

    def __init__(self, cooldown_seconds: float = 1.5) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_accept_time: float = 0.0

    @property
    def in_cooldown(self) -> bool:
        return (time.time() - self._last_accept_time) < self.cooldown_seconds

    def record_acceptance(self) -> None:
        self._last_accept_time = time.time()

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.cooldown_seconds -
                   (time.time() - self._last_accept_time))


# ---------------------------------------------------------------------------
# 4. Sentence builder
# ---------------------------------------------------------------------------

class SentenceBuilder:
    """
    Accumulates accepted signs into a display phrase.

    Parameters
    ----------
    max_words        : int   — truncate sentence at this many words (20)
    idle_reset_secs  : float — reset sentence after this many seconds of silence
    """

    def __init__(
        self,
        max_words:       int   = 20,
        idle_reset_secs: float = 8.0,
    ) -> None:
        self.max_words       = max_words
        self.idle_reset_secs = idle_reset_secs
        self._words: list[str] = []
        self._last_accept_time = time.time()

    def add(self, sign: str) -> None:
        """Add a sign to the sentence."""
        self._last_accept_time = time.time()

        # Auto-reset on long silence (new sentence)
        if time.time() - self._last_accept_time > self.idle_reset_secs:
            self._words = []

        self._words.append(sign)
        if len(self._words) > self.max_words:
            self._words = self._words[-self.max_words:]

    def get(self) -> str:
        """Return the current sentence as a space-joined string."""
        # Auto-reset on long silence
        if (self._words and
                time.time() - self._last_accept_time > self.idle_reset_secs):
            self._words = []
        return " ".join(self._words)

    def clear(self) -> None:
        """Manually clear the sentence."""
        self._words = []
        self._last_accept_time = time.time()

    @property
    def word_count(self) -> int:
        return len(self._words)


# ---------------------------------------------------------------------------
# 5. PostProcessor — orchestrates all stages
# ---------------------------------------------------------------------------

class PostProcessor:
    """
    Single entry-point for the full post-processing pipeline.

    Parameters
    ----------
    smoothing_window   : int   — TemporalSmoother window size (5)
    confidence_threshold: float — minimum confidence to track (0.70)
    required_stable    : int   — StabilityGate frames needed (8)
    cooldown_seconds   : float — CooldownGuard pause after sign (1.5)
    idle_reset_secs    : float — SentenceBuilder reset after silence (8.0)
    """

    def __init__(
        self,
        smoothing_window:    int   = 5,
        confidence_threshold: float = 0.70,
        required_stable:     int   = 8,
        cooldown_seconds:    float = 1.5,
        idle_reset_secs:     float = 8.0,
    ) -> None:
        self.smoother  = TemporalSmoother(smoothing_window, confidence_threshold)
        self.gate      = StabilityGate(required_stable)
        self.cooldown  = CooldownGuard(cooldown_seconds)
        self.sentence  = SentenceBuilder(idle_reset_secs=idle_reset_secs)

        self._last_state: SignState = SignState.IDLE

    # ------------------------------------------------------------------
    # Main update — call once per inference result
    # ------------------------------------------------------------------

    def update(self, prediction: Prediction | None) -> PostProcessorOutput:
        """
        Process one inference result through the full pipeline.

        Parameters
        ----------
        prediction : Prediction from InferenceEngine, or None if no hand

        Returns
        -------
        PostProcessorOutput
        """
        # No hand detected or no prediction
        if prediction is None:
            self.smoother.reset()
            self.gate.reset()
            return PostProcessorOutput(
                state=        SignState.IDLE,
                current_sign= None,
                confidence=   0.0,
                sentence=     self.sentence.get(),
            )

        # Stage 1: Temporal smoothing
        smoothed_class, smoothed_conf = self.smoother.update(prediction)

        if smoothed_class is None:
            # Buffer not full or confidence too low
            self.gate.reset()
            return PostProcessorOutput(
                state=        SignState.TRACKING,
                current_sign= prediction.class_name,
                confidence=   prediction.confidence,
                sentence=     self.sentence.get(),
            )

        # Stage 2: Stability gate
        is_stable, stability_ratio = self.gate.update(smoothed_class)

        # Stage 3: Cooldown guard
        if self.cooldown.in_cooldown:
            return PostProcessorOutput(
                state=           SignState.COOLDOWN,
                current_sign=    smoothed_class,
                confidence=      smoothed_conf,
                stability_ratio= stability_ratio,
                sentence=        self.sentence.get(),
            )

        if not is_stable:
            return PostProcessorOutput(
                state=           SignState.TRACKING,
                current_sign=    smoothed_class,
                confidence=      smoothed_conf,
                stability_ratio= stability_ratio,
                sentence=        self.sentence.get(),
            )

        # Stage 4: Accept the sign
        self.sentence.add(smoothed_class)
        self.cooldown.record_acceptance()
        self.smoother.reset()
        self.gate.reset()

        logger.debug(f"Sign accepted: {smoothed_class} (conf={smoothed_conf:.3f})")

        return PostProcessorOutput(
            state=           SignState.ACCEPTED,
            current_sign=    smoothed_class,
            confidence=      smoothed_conf,
            just_accepted=   True,
            accepted_sign=   smoothed_class,
            sentence=        self.sentence.get(),
            stability_ratio= 1.0,
        )

    def clear_sentence(self) -> None:
        """Manually clear the sentence (bound to keyboard shortcut)."""
        self.sentence.clear()
        logger.debug("Sentence cleared by user.")

    def reset_all(self) -> None:
        """Full reset of all stages."""
        self.smoother.reset()
        self.gate.reset()
        self.sentence.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_post_processor(inference_cfg: dict | None = None) -> PostProcessor:
    """
    Build a PostProcessor from an optional config dict.
    Falls back to safe defaults if no config provided.
    """
    cfg = inference_cfg or {}
    return PostProcessor(
        smoothing_window=     int(  cfg.get("smoothing_window",    5)),
        confidence_threshold= float(cfg.get("confidence_threshold", 0.70)),
        required_stable=      int(  cfg.get("required_stable",      8)),
        cooldown_seconds=     float(cfg.get("cooldown_seconds",      1.5)),
        idle_reset_secs=      float(cfg.get("idle_reset_secs",       8.0)),
    )
