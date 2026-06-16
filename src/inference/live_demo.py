"""
live_demo.py
------------
Real-time ISL recognition demo using OpenCV + MediaPipe + SignBridge model.

Full pipeline per frame
-----------------------
Webcam → MediaPipe → 30-frame sliding buffer → InferenceEngine →
PostProcessor → OpenCV overlay + pyttsx3 TTS

UI Overlays
-----------
Top bar    : Status (IDLE / TRACKING / ACCEPTED / COOLDOWN), FPS, backend
Centre     : Live MediaPipe skeleton on hands
Bottom bar : Current prediction + confidence, stability bar, sentence

Keyboard controls
-----------------
  Q        — quit
  C        — clear sentence
  SPACE    — force clear + reset post-processor
  S        — toggle TTS on/off
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from src.collection.hand_detector import HandDetector, DetectionResult, build_hand_detector
from src.inference.inference_engine import _BaseEngine, Prediction
from src.inference.post_processor import PostProcessor, PostProcessorOutput, SignState
from src.inference.monitor import InferenceMonitor
from src.utils.class_labels import requires_two_hands
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# UI constants (BGR colours)
# ---------------------------------------------------------------------------
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX
_PAD        = 12

_C_WHITE    = (240, 240, 240)
_C_BLACK    = (20,  20,  20)
_C_GREEN    = (50,  200,  50)
_C_YELLOW   = (30,  210, 210)
_C_RED      = (60,   60, 220)
_C_CYAN     = (200, 200,   0)
_C_ORANGE   = (30,  140, 255)
_C_GRAY     = (140, 140, 140)
_C_PURPLE   = (200,  60, 200)

_STATE_STYLE: dict[SignState, tuple[str, tuple]] = {
    SignState.IDLE:     ("IDLE",     _C_GRAY),
    SignState.TRACKING: ("TRACKING", _C_YELLOW),
    SignState.STABLE:   ("STABLE",   _C_GREEN),
    SignState.ACCEPTED: ("ACCEPTED", _C_GREEN),
    SignState.COOLDOWN: ("COOLDOWN", _C_ORANGE),
}

# ---------------------------------------------------------------------------
# Threaded TTS engine
# ---------------------------------------------------------------------------

class _TTSEngine:
    """
    Non-blocking TTS using a background thread.
    Prevents pyttsx3.runAndWait() from freezing the OpenCV loop.
    """

    def __init__(self) -> None:
        self._enabled = True
        self._queue: deque[str] = deque(maxlen=3)
        self._lock   = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._queue.append(text)

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        logger.info(f"TTS {'enabled' if self._enabled else 'disabled'}")
        return self._enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _worker(self) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)    # words per minute
            engine.setProperty("volume", 0.9)
        except Exception as exc:
            logger.warning(f"TTS unavailable: {exc}")
            return

        while True:
            text = None
            with self._lock:
                if self._queue:
                    text = self._queue.popleft()
            if text:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass
            else:
                time.sleep(0.05)


# ---------------------------------------------------------------------------
# Sliding frame buffer
# ---------------------------------------------------------------------------

class _SlidingBuffer:
    """
    Maintains a rolling window of the last N landmark feature vectors.
    Returns the full sequence when the buffer is full.
    """

    def __init__(self, size: int = 30) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=size)
        self.size = size

    def push(self, vector: np.ndarray) -> None:
        """Add one (126,) frame to the buffer."""
        self._buf.append(vector.copy())

    def get_sequence(self) -> np.ndarray | None:
        """Return (size, 126) array if full, else None."""
        if len(self._buf) < self.size:
            return None
        return np.stack(list(self._buf), axis=0)   # (30, 126)

    @property
    def fill_ratio(self) -> float:
        return len(self._buf) / self.size

    def clear(self) -> None:
        self._buf.clear()


# ---------------------------------------------------------------------------
# Main demo class
# ---------------------------------------------------------------------------

class LiveDemo:
    """
    Real-time ISL recognition demo.

    Parameters
    ----------
    engine          : _BaseEngine — inference engine (PyTorch or ONNX)
    post_processor  : PostProcessor
    dataset_cfg     : dict — for HandDetector config
    camera_index    : int
    infer_every_n   : int  — run inference every N frames (1 = every frame)
    """

    def __init__(
        self,
        engine:         _BaseEngine,
        post_processor: PostProcessor,
        dataset_cfg:    dict,
        camera_index:   int = 0,
        infer_every_n:  int = 1,
    ) -> None:
        self.engine         = engine
        self.post_processor = post_processor
        self.dataset_cfg    = dataset_cfg
        self.camera_index   = camera_index
        self.infer_every_n  = infer_every_n

        self._buffer   = _SlidingBuffer(size=30)
        self._tts      = _TTSEngine()
        self._monitor  = InferenceMonitor()
        self._frame_n  = 0

        # FPS tracking
        self._fps_times: deque[float] = deque(maxlen=30)
        self._fps: float = 0.0

        # Last inference result (held until next inference)
        self._last_prediction: Prediction | None = None
        self._last_output: PostProcessorOutput | None = None

    # ------------------------------------------------------------------
    # Public run method
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open webcam and start the inference loop. Blocks until Q pressed."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam (index={self.camera_index}). "
                "Check camera is connected and not in use by another app."
            )

        # Request 720p
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"Webcam opened | index={self.camera_index} | "
            f"{actual_w}×{actual_h}"
        )

        window_name = "SignBridge — Real-Time ISL Recognition"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        with build_hand_detector(self.dataset_cfg) as detector:
            logger.info("Demo started — press Q to quit, C to clear, S to toggle TTS")
            self._main_loop(cap, detector, window_name)

        cap.release()
        cv2.destroyAllWindows()
        # Print and save session monitoring summary
        self._monitor.print_session_summary()
        self._monitor.save_session_log()
        logger.info("Demo ended.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _main_loop(
        self,
        cap:         cv2.VideoCapture,
        detector:    HandDetector,
        window_name: str,
    ) -> None:
        output = PostProcessorOutput(
            state=    SignState.IDLE,
            sentence= "",
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)   # mirror view
            t_now = time.time()

            # ── Hand detection ────────────────────────────────────────
            detection: DetectionResult = detector.process_frame(frame)
            self._monitor.record_frame(detected_hands=detection.num_hands)

            if detection.is_valid:
                self._buffer.push(detection.feature_vector)
            else:
                # No hands → reset buffer, inform post-processor
                self._buffer.clear()
                output = self.post_processor.update(None)

            # ── Inference (every N frames when buffer is full) ────────
            self._frame_n += 1
            if (
                detection.is_valid
                    and self._frame_n % self.infer_every_n == 0
            ):
                sequence = self._buffer.get_sequence()
                if sequence is not None:
                    try:
                        t_inf = time.time()
                        self._last_prediction = self.engine.predict(sequence)
                        inf_ms = (time.time() - t_inf) * 1000
                        self._monitor.record_frame(
                            detected_hands=detection.num_hands,
                            inference_ms=inf_ms,
                        )
                        self._monitor.record_prediction(
                            self._last_prediction.class_name,
                            self._last_prediction.confidence,
                        )
                        output = self.post_processor.update(self._last_prediction)
                    except Exception as exc:
                        logger.warning(f"Inference error: {exc}")

            # ── TTS on acceptance ─────────────────────────────────────
            if output.just_accepted and output.accepted_sign:
                self._tts.speak(output.accepted_sign)

            # ── FPS tracking ─────────────────────────────────────────
            self._fps_times.append(t_now)
            if len(self._fps_times) >= 2:
                elapsed = self._fps_times[-1] - self._fps_times[0]
                self._fps = (len(self._fps_times) - 1) / max(elapsed, 1e-6)

            # ── Draw UI ───────────────────────────────────────────────
            canvas = detection.annotated_frame.copy()
            self._draw_ui(canvas, output, detection)

            cv2.imshow(window_name, canvas)

            # ── Key handling ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                self.post_processor.clear_sentence()
                logger.info("Sentence cleared (C key).")
            elif key == ord(" "):
                self.post_processor.reset_all()
                self._buffer.clear()
                logger.info("Full reset (SPACE key).")
            elif key in (ord("s"), ord("S")):
                tts_state = self._tts.toggle()
                status = "ON" if tts_state else "OFF"
                logger.info(f"TTS toggled {status}")

    # ------------------------------------------------------------------
    # UI drawing
    # ------------------------------------------------------------------

    def _draw_ui(
        self,
        frame:     np.ndarray,
        output:    PostProcessorOutput,
        detection: DetectionResult,
    ) -> None:
        h, w = frame.shape[:2]
        self._draw_top_bar(frame, output, detection, w)
        self._draw_bottom_bar(frame, output, detection, h, w)
        if output.stability_ratio > 0:
            self._draw_stability_bar(frame, output.stability_ratio, h, w)

    def _draw_top_bar(
        self,
        frame:     np.ndarray,
        output:    PostProcessorOutput,
        detection: DetectionResult,
        w:         int,
    ) -> None:
        """Status bar at the top of the frame."""
        bar_h  = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), _C_BLACK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        state_text, state_color = _STATE_STYLE.get(
            output.state, ("UNKNOWN", _C_WHITE)
        )

        # State label
        cv2.putText(frame, state_text,
                    (_PAD, 32), _FONT_BOLD, 0.80, state_color, 2)

        # FPS + backend
        backend_name = getattr(
            getattr(self.engine, "backend", None), "name", "PyTorch"
        )
        fps_text = (f"FPS: {self._fps:.1f}  |  "
                    f"{backend_name}  |  "
                    f"TTS: {'ON' if self._tts.enabled else 'OFF'}  |  "
                    f"Hands: {getattr(detection, 'num_hands', 0)}")
        cv2.putText(frame, fps_text,
                    (w - 420, 32), _FONT, 0.55, _C_GRAY, 1)

        # Instructions
        cv2.putText(frame, "Q=quit  C=clear  SPACE=reset  S=TTS",
                    (_PAD, 48), _FONT, 0.42, _C_GRAY, 1)

    def _draw_bottom_bar(
        self,
        frame:     np.ndarray,
        output:    PostProcessorOutput,
        detection: DetectionResult,
        h:         int,
        w:         int,
    ) -> None:
        """Prediction + sentence bar at the bottom."""
        bar_h   = 90
        bar_top = h - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_top), (w, h), _C_BLACK, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

        # Current sign prediction (large)
        if output.current_sign:
            conf_pct = int(output.confidence * 100)
            sign_color = _C_GREEN if output.confidence >= 0.85 else _C_YELLOW
            cv2.putText(frame,
                        f"{output.current_sign}",
                        (_PAD, bar_top + 38),
                        _FONT_BOLD, 1.4, sign_color, 3)
            cv2.putText(frame,
                        f"{conf_pct}%",
                        (_PAD + 65, bar_top + 38),
                        _FONT, 0.75, sign_color, 2)

        # "ACCEPTED" flash
        if output.just_accepted:
            cv2.putText(frame, "✓ ACCEPTED",
                        (_PAD + 130, bar_top + 38),
                        _FONT_BOLD, 0.75, _C_GREEN, 2)

        # Sentence
        sentence = output.sentence or "..."
        # Truncate if too long for display
        max_chars = w // 13
        if len(sentence) > max_chars:
            sentence = "..." + sentence[-max_chars + 3:]
        cv2.putText(frame, sentence,
                    (_PAD, bar_top + 72),
                    _FONT, 0.70, _C_WHITE, 1)

        # Buffer fill indicator (small dots)
        fill = self._buffer.fill_ratio
        bar_w = 100
        cv2.rectangle(frame,
                      (w - bar_w - _PAD, bar_top + 8),
                      (w - _PAD, bar_top + 18),
                      (60, 60, 60), -1)
        cv2.rectangle(frame,
                      (w - bar_w - _PAD, bar_top + 8),
                      (w - _PAD - int((1 - fill) * bar_w), bar_top + 18),
                      _C_CYAN, -1)
        cv2.putText(frame, "buf",
                    (w - bar_w - _PAD - 28, bar_top + 17),
                    _FONT, 0.38, _C_GRAY, 1)

    def _draw_stability_bar(
        self,
        frame: np.ndarray,
        ratio: float,
        h:     int,
        w:     int,
    ) -> None:
        """Horizontal stability progress bar at h-95."""
        bar_y  = h - 95
        bar_w  = w - 2 * _PAD
        filled = int(bar_w * ratio)
        color  = _C_GREEN if ratio >= 1.0 else _C_YELLOW

        cv2.rectangle(frame, (_PAD, bar_y), (_PAD + bar_w, bar_y + 5),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (_PAD, bar_y), (_PAD + filled, bar_y + 5),
                      color, -1)
