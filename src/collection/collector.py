"""
collector.py
------------
Main data collection orchestrator for the SignBridge ISL dataset.

Responsibilities
----------------
* Open the webcam and display a live preview window
* Show a countdown before each new class recording session
* Overlay real-time status (class name, state, progress, FPS) on frames
* Coordinate HandDetector and AutoRecorder for each class
* Support collecting a single specified class or all 46 classes in sequence
* Graceful exit on 'Q' keypress at any time
* Persist every completed sequence immediately so data is never lost

Usage (via scripts/collect_data.py)
------------------------------------
    from src.collection.collector import DataCollector
    collector = DataCollector(dataset_cfg)
    collector.collect_all()          # All 46 classes
    collector.collect_class("A")     # Single class
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.collection.hand_detector import HandDetector, DetectionResult, build_hand_detector
from src.collection.auto_recorder import AutoRecorder, RecorderState, build_auto_recorder
from src.utils.class_labels import (
    CLASS_LABELS,
    requires_two_hands,
    display_label,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# OpenCV overlay constants
# ---------------------------------------------------------------------------
_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 0.7
_THICKNESS   = 2
_PAD         = 12   # pixels padding for text overlay

# Status colours (BGR)
_COLOR_GREEN  = (0,   200,  50)
_COLOR_YELLOW = (0,   200, 200)
_COLOR_RED    = (50,   50, 220)
_COLOR_WHITE  = (240, 240, 240)
_COLOR_GRAY   = (160, 160, 160)
_COLOR_CYAN   = (200, 200,   0)

# State → display text + colour mapping
_STATE_DISPLAY = {
    RecorderState.IDLE:      ("IDLE  — show hands",      _COLOR_GRAY),
    RecorderState.WAITING:   ("HOLD  — keep steady",     _COLOR_YELLOW),
    RecorderState.RECORDING: ("REC   — do not move",     _COLOR_RED),
    RecorderState.COOLDOWN:  ("SAVED — brief pause",     _COLOR_GREEN),
}


class DataCollector:
    """
    Orchestrates the full data collection session.

    Parameters
    ----------
    dataset_cfg : dict
        Parsed dataset.yaml configuration dictionary.
    """

    def __init__(self, dataset_cfg: dict) -> None:
        self.dataset_cfg   = dataset_cfg
        self.rec_cfg       = dataset_cfg.get("recording", {})
        self.camera_index  = int(self.rec_cfg.get("camera_index", 0))
        self.countdown_sec = int(self.rec_cfg.get("countdown_seconds", 3))
        self._detector: HandDetector | None = None
        self._cap:      cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def collect_all(self, start_from: str | None = None) -> None:
        """
        Collect sequences for all 46 ISL classes in order.

        Parameters
        ----------
        start_from : str | None
            If given, skip all classes before this one (resume support).
        """
        classes = CLASS_LABELS
        if start_from and start_from in classes:
            idx    = classes.index(start_from)
            classes = classes[idx:]
            logger.info(f"Resuming from class '{start_from}' (index {idx})")

        self._open_camera()
        with build_hand_detector(self.dataset_cfg) as detector:
            self._detector = detector
            for class_name in classes:
                if not self._collect_class_session(class_name):
                    logger.warning("Collection aborted by user.")
                    break
        self._release_camera()
        logger.info("All classes collection complete.")

    def collect_class(self, class_name: str) -> None:
        """
        Collect sequences for a single ISL class.

        Parameters
        ----------
        class_name : str
            Must be one of the 46 ISL class labels.
        """
        if class_name not in CLASS_LABELS:
            raise ValueError(
                f"Unknown class '{class_name}'. "
                f"Valid: {CLASS_LABELS}"
            )
        self._open_camera()
        with build_hand_detector(self.dataset_cfg) as detector:
            self._detector = detector
            self._collect_class_session(class_name)
        self._release_camera()

    # ------------------------------------------------------------------
    # Core session logic
    # ------------------------------------------------------------------

    def _collect_class_session(self, class_name: str) -> bool:
        """
        Run a full collection session for one ISL class.

        Returns
        -------
        bool
            True if session completed normally, False if user pressed Q.
        """
        recorder = build_auto_recorder(class_name, self.dataset_cfg)
        two_hand = requires_two_hands(class_name)

        logger.info(
            f"Starting class '{class_name}' | "
            f"{'2-hand' if two_hand else '1-hand'} sign | "
            f"target={recorder.num_sequences} sequences"
        )

        # Show countdown screen before starting
        if not self._show_countdown(class_name):
            return False   # Q pressed during countdown

        # Main frame loop
        fps_timer  = time.time()
        fps_value  = 0.0
        frame_cnt  = 0

        while not recorder.is_complete:
            ret, frame = self._cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam.")
                break

            # Flip horizontally for mirror-view UX
            frame = cv2.flip(frame, 1)

            # Run hand detection
            detection: DetectionResult = self._detector.process_frame(frame)
            hand_ok = self._detector.is_correct_hand_count(detection, class_name)

            # Update recorder state machine
            state = recorder.update(detection.feature_vector, hand_ok)

            # Overlay UI elements on the annotated frame
            annotated = detection.annotated_frame
            self._draw_status_bar(annotated, class_name, state, recorder, fps_value, two_hand)
            self._draw_hand_count_indicator(annotated, detection.num_hands, two_hand)

            cv2.imshow("SignBridge — Data Collection", annotated)

            # FPS calculation
            frame_cnt += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_value = frame_cnt / elapsed
                frame_cnt  = 0
                fps_timer  = time.time()

            # Key handler
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                logger.info("Q pressed — aborting collection.")
                return False
            if key == ord("r") or key == ord("R"):
                recorder.reset_session()
                logger.info(f"[{class_name}] Session reset by user (R key).")

        saved, total = recorder.progress
        logger.info(
            f"Class '{class_name}' complete — "
            f"{saved}/{total} sequences saved."
        )
        return True

    # ------------------------------------------------------------------
    # Countdown screen
    # ------------------------------------------------------------------

    def _show_countdown(self, class_name: str) -> bool:
        """
        Display a full-screen countdown before recording starts.
        Returns False if the user presses Q during countdown.
        """
        two_hand  = requires_two_hands(class_name)
        hand_str  = "BOTH hands" if two_hand else "ONE hand"
        disp_name = display_label(class_name)

        for remaining in range(self.countdown_sec, 0, -1):
            deadline = time.time() + 1.0
            while time.time() < deadline:
                ret, frame = self._cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                # Dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

                # Class name
                cv2.putText(frame, f"Next: {disp_name}", (w // 2 - 160, h // 2 - 80),
                            _FONT, 1.1, _COLOR_WHITE, 2)
                # Hand instruction
                cv2.putText(frame, f"Show {hand_str}", (w // 2 - 120, h // 2 - 40),
                            _FONT, 0.8, _COLOR_CYAN, _THICKNESS)
                # Countdown number
                cv2.putText(frame, str(remaining), (w // 2 - 30, h // 2 + 60),
                            _FONT, 4.0, _COLOR_GREEN, 4)
                # Instructions
                cv2.putText(frame, "Press Q to quit | R to reset",
                            (w // 2 - 170, h - 30),
                            _FONT, 0.6, _COLOR_GRAY, 1)

                cv2.imshow("SignBridge — Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    return False

        return True

    # ------------------------------------------------------------------
    # UI drawing helpers
    # ------------------------------------------------------------------

    def _draw_status_bar(
        self,
        frame:      np.ndarray,
        class_name: str,
        state:      RecorderState,
        recorder:   AutoRecorder,
        fps:        float,
        two_hand:   bool,
    ) -> None:
        """Draw a status bar at the top of the frame."""
        h, w = frame.shape[:2]
        bar_h = 70

        # Semi-transparent background strip
        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)

        state_text, state_color = _STATE_DISPLAY.get(
            state, ("UNKNOWN", _COLOR_WHITE)
        )
        saved, total = recorder.progress

        # Row 1: class name + state
        cv2.putText(frame, f"{display_label(class_name)}", (_PAD, 25),
                    _FONT, _FONT_SCALE, _COLOR_WHITE, _THICKNESS)
        cv2.putText(frame, state_text, (w // 2 - 100, 25),
                    _FONT, _FONT_SCALE, state_color, _THICKNESS)

        # Row 2: progress + FPS
        progress_pct = int((saved / total) * 100) if total > 0 else 0
        cv2.putText(frame,
                    f"Seqs: {saved}/{total}  ({progress_pct}%)",
                    (_PAD, 55), _FONT, 0.60, _COLOR_GREEN, 1)
        cv2.putText(frame,
                    f"FPS: {fps:.1f}  |  {'2-hand' if two_hand else '1-hand'}",
                    (w - 220, 55), _FONT, 0.60, _COLOR_GRAY, 1)

        # Progress bar
        bar_y  = bar_h - 6
        bar_w  = w - 2 * _PAD
        filled = int(bar_w * saved / total) if total > 0 else 0
        cv2.rectangle(frame, (_PAD, bar_y), (_PAD + bar_w, bar_y + 4),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (_PAD, bar_y), (_PAD + filled, bar_y + 4),
                      _COLOR_GREEN, -1)

    def _draw_hand_count_indicator(
        self,
        frame:       np.ndarray,
        num_detected:int,
        two_hand:    bool,
    ) -> None:
        """
        Draw a small indicator showing how many hands are detected and whether
        the count matches the requirement for the current class.
        """
        h, w   = frame.shape[:2]
        needed = 2 if two_hand else 1
        ok     = num_detected == needed
        color  = _COLOR_GREEN if ok else _COLOR_RED
        text   = f"Hands: {num_detected}/{needed}"
        cv2.putText(frame, text, (_PAD, h - _PAD),
                    _FONT, 0.65, color, _THICKNESS)

    # ------------------------------------------------------------------
    # Camera lifecycle
    # ------------------------------------------------------------------

    def _open_camera(self) -> None:
        """Open the webcam capture device."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam (index={self.camera_index}). "
                "Check that the camera is connected and not in use."
            )
        # Request a higher resolution if available
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            f"Webcam opened | index={self.camera_index} | "
            f"resolution={actual_w}×{actual_h}"
        )

    def _release_camera(self) -> None:
        """Release webcam and destroy all OpenCV windows."""
        if self._cap:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
        logger.info("Webcam released.")
