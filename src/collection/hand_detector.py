"""
hand_detector.py
----------------
MediaPipe Hands wrapper using the Tasks API (mediapipe >= 0.10.x).

MediaPipe 0.10.x removed mp.solutions entirely.
This module uses mp.tasks.vision.HandLandmarker — the current stable API.

The hand_landmarker.task model file is automatically downloaded on first run
and cached at models/mediapipe/hand_landmarker.task.

Public interface is identical to the previous version so all callers
(collector.py, auto_recorder.py) work without any changes.
"""

from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from src.utils.logger import get_logger
from src.utils.class_labels import requires_two_hands

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MediaPipe Tasks API — available in all mediapipe >= 0.10.x installs
# ---------------------------------------------------------------------------
_BaseOptions        = mp.tasks.BaseOptions
_HandLandmarker     = mp.tasks.vision.HandLandmarker
_HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
_RunningMode        = mp.tasks.vision.RunningMode

# ---------------------------------------------------------------------------
# Model file paths and download URL
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent   # PROJECT/
_MODEL_DIR    = _PROJECT_ROOT / "models" / "mediapipe"
_MODEL_PATH   = _MODEL_DIR / "hand_landmarker.task"
_MODEL_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# ---------------------------------------------------------------------------
# Hand skeleton connections for OpenCV drawing (21 landmarks)
# Defined manually — no longer available via mp.solutions
# ---------------------------------------------------------------------------
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# Drawing colours (BGR)
_LM_COLOR   = (0,   255,   0)   # Green dots for landmarks
_CONN_COLOR = (200, 200, 200)   # White lines for connections
_LM_RADIUS  = 5
_CONN_THICK = 2


# ---------------------------------------------------------------------------
# Model auto-downloader
# ---------------------------------------------------------------------------

def _ensure_model(model_path: Path = _MODEL_PATH) -> Path:
    """
    Ensure the hand_landmarker.task model file exists.
    Downloads it automatically on first run (~30 MB).

    Parameters
    ----------
    model_path : Path
        Where to save/find the model file.

    Returns
    -------
    Path
        Path to the verified model file.

    Raises
    ------
    RuntimeError
        If the download fails and no cached model exists.
    """
    if model_path.exists():
        logger.debug(f"Model found at {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("Hand landmarker model not found — downloading now.")
    logger.info(f"Source : {_MODEL_URL}")
    logger.info(f"Target : {model_path}")
    logger.info("Size   : ~30 MB  (one-time download)")
    logger.info("=" * 60)

    try:
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, int(block_num * block_size * 100 / total_size))
                print(f"\r  Downloading... {pct:3d}%", end="", flush=True)

        urllib.request.urlretrieve(_MODEL_URL, model_path, reporthook=_progress)
        print()   # newline after progress bar
        logger.info("Model downloaded and cached successfully.")
    except Exception as exc:
        raise RuntimeError(
            f"\n[HandDetector] Failed to download MediaPipe model.\n"
            f"URL   : {_MODEL_URL}\n"
            f"Error : {exc}\n\n"
            f"Manual fix:\n"
            f"  1. Open a browser and download:\n"
            f"     {_MODEL_URL}\n"
            f"  2. Save the file to:\n"
            f"     {model_path}\n"
            f"  3. Re-run the script.\n"
        ) from exc

    return model_path


# ---------------------------------------------------------------------------
# Data structures  (same public interface as before)
# ---------------------------------------------------------------------------

@dataclass
class HandLandmarks:
    """
    Landmarks for a single detected hand.

    Attributes
    ----------
    vector : np.ndarray, shape (63,)
        Flattened [x, y, z] for 21 keypoints.
    handedness : str
        'Left' or 'Right'.
    raw_landmarks : list
        Raw mediapipe NormalizedLandmark list.
    confidence : float
        Handedness classification score.
    """
    vector:        np.ndarray
    handedness:    str
    raw_landmarks: list
    confidence:    float = 1.0


@dataclass
class DetectionResult:
    """
    Result of processing one video frame.

    Attributes
    ----------
    hands : list[HandLandmarks]
        Detected hands (0, 1, or 2 entries).
    num_hands : int
    feature_vector : np.ndarray, shape (126,)
        Right-hand (0:63) + Left-hand (63:126). Zero-padded if absent.
    annotated_frame : np.ndarray
        BGR frame with skeleton drawn on it.
    is_valid : bool
        True if at least one hand was detected.
    """
    hands:            list[HandLandmarks] = field(default_factory=list)
    num_hands:        int                 = 0
    feature_vector:   np.ndarray          = field(
        default_factory=lambda: np.zeros(126, dtype=np.float32)
    )
    annotated_frame:  np.ndarray | None   = None
    is_valid:         bool                = False


# ---------------------------------------------------------------------------
# HandDetector
# ---------------------------------------------------------------------------

class HandDetector:
    """
    Wraps MediaPipe Tasks HandLandmarker for the SignBridge pipeline.

    Parameters
    ----------
    min_detection_confidence : float
    min_tracking_confidence  : float
    max_num_hands            : int   (1 or 2)
    draw_landmarks           : bool
    model_path               : Path | None
        Override model path (default: models/mediapipe/hand_landmarker.task)
    """

    LANDMARKS_PER_HAND: int = 21
    COORDS_PER_LANDMARK: int = 3
    SINGLE_HAND_DIM: int = 63     # 21 × 3
    TWO_HAND_DIM:    int = 126    # 21 × 3 × 2

    def __init__(
        self,
        min_detection_confidence: float       = 0.75,
        min_tracking_confidence:  float       = 0.75,
        max_num_hands:            int         = 2,
        draw_landmarks:           bool        = True,
        model_path:               Path | None = None,
    ) -> None:
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.max_num_hands            = max_num_hands
        self.draw_landmarks           = draw_landmarks

        # Ensure model is available (download if missing)
        resolved_model = _ensure_model(model_path or _MODEL_PATH)

        # Build HandLandmarker with VIDEO running mode for real-time tracking
        options = _HandLandmarkerOpts(
            base_options=_BaseOptions(
                model_asset_path=str(resolved_model)
            ),
            running_mode=_RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker  = _HandLandmarker.create_from_options(options)
        self._start_ms    = int(time.time() * 1000)

        logger.info(
            f"HandDetector ready (Tasks API) | "
            f"max_hands={max_num_hands} | "
            f"det_conf={min_detection_confidence} | "
            f"track_conf={min_tracking_confidence}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, bgr_frame: np.ndarray) -> DetectionResult:
        """
        Process a single BGR video frame.

        Parameters
        ----------
        bgr_frame : np.ndarray
            BGR frame from cv2.VideoCapture.read().

        Returns
        -------
        DetectionResult
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Wrap in MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Monotonically increasing timestamp required for VIDEO mode
        timestamp_ms = int(time.time() * 1000) - self._start_ms
        if timestamp_ms < 0:
            timestamp_ms = 0

        # Run detection
        mp_result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        annotated = bgr_frame.copy()
        result    = DetectionResult(annotated_frame=annotated)

        if not mp_result.hand_landmarks:
            return result

        hands: list[HandLandmarks] = []

        for lm_list, handedness_list in zip(
            mp_result.hand_landmarks,
            mp_result.handedness,
        ):
            vector     = self._landmarks_to_vector(lm_list)
            handedness = handedness_list[0].category_name   # 'Left' | 'Right'
            confidence = handedness_list[0].score

            hands.append(HandLandmarks(
                vector=vector,
                handedness=handedness,
                raw_landmarks=lm_list,
                confidence=confidence,
            ))

            if self.draw_landmarks:
                self._draw_hand(annotated, lm_list, bgr_frame.shape)

        feature_vector = self._build_feature_vector(hands)

        result.hands          = hands
        result.num_hands      = len(hands)
        result.feature_vector = feature_vector
        result.is_valid       = len(hands) > 0

        return result

    def is_correct_hand_count(
        self,
        detection:    DetectionResult,
        target_class: str,
    ) -> bool:
        """Return True if detected hand count matches sign requirements."""
        if requires_two_hands(target_class):
            return detection.num_hands == 2
        return detection.num_hands == 1

    def release(self) -> None:
        """Release the HandLandmarker resources."""
        try:
            self._landmarker.close()
        except Exception:
            pass
        logger.debug("HandDetector released.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _landmarks_to_vector(lm_list) -> np.ndarray:
        """
        Convert a MediaPipe landmark list to a flat (63,) float32 array.
        Works with both Tasks API NormalizedLandmark objects (which have
        .x, .y, .z attributes directly).
        """
        coords = []
        for lm in lm_list:
            coords.extend([lm.x, lm.y, lm.z])
        return np.array(coords, dtype=np.float32)   # shape: (63,)

    @staticmethod
    def _build_feature_vector(hands: list[HandLandmarks]) -> np.ndarray:
        """
        Build a unified 126-dim vector.
        Slot 0 ( 0:63)  → Right hand  (zeros if absent)
        Slot 1 (63:126) → Left  hand  (zeros if absent)
        """
        right_vec = np.zeros(63, dtype=np.float32)
        left_vec  = np.zeros(63, dtype=np.float32)

        for hand in hands:
            if hand.handedness == "Right":
                right_vec = hand.vector
            else:
                left_vec  = hand.vector

        return np.concatenate([right_vec, left_vec], axis=0)   # (126,)

    @staticmethod
    def _draw_hand(
        frame:    np.ndarray,
        lm_list,
        shape:    tuple,
    ) -> None:
        """
        Draw hand skeleton on frame using OpenCV.
        (Replaces mp.solutions.drawing_utils which no longer exists.)
        """
        h, w = shape[:2]

        # Convert normalised coords to pixel coords
        pts: list[tuple[int, int]] = [
            (int(lm.x * w), int(lm.y * h)) for lm in lm_list
        ]

        # Draw connections first (so dots appear on top)
        for start_idx, end_idx in _HAND_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(frame, pts[start_idx], pts[end_idx],
                         _CONN_COLOR, _CONN_THICK)

        # Draw landmark dots
        for pt in pts:
            cv2.circle(frame, pt, _LM_RADIUS, _LM_COLOR, -1)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_hand_detector(dataset_cfg: dict) -> HandDetector:
    """
    Build a HandDetector from a parsed dataset.yaml config dict.
    """
    hd_cfg = dataset_cfg.get("hand_detection", {})
    return HandDetector(
        min_detection_confidence=float(hd_cfg.get("min_detection_confidence", 0.75)),
        min_tracking_confidence= float(hd_cfg.get("min_tracking_confidence",  0.75)),
        max_num_hands=           int(  hd_cfg.get("max_num_hands", 2)),
        draw_landmarks=True,
    )
