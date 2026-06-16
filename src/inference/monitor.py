"""
monitor.py
----------
Production monitoring for the SignBridge real-time inference pipeline.

Tracks per-session metrics and writes structured logs for:
  - Prediction confidence distribution
  - Sign acceptance rate and frequency
  - Inference latency (ms per call)
  - Hand detection rate (frames with hands / total frames)
  - Confusion events (low-confidence, rejected predictions)
  - Session summary on exit

Usage
-----
    from src.inference.monitor import InferenceMonitor
    monitor = InferenceMonitor()
    monitor.record_frame(detected_hands=2, inference_ms=2.8)
    monitor.record_prediction(class_name="A", confidence=0.94)
    monitor.record_acceptance(class_name="A")
    monitor.save_session_log()  # call on demo exit
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceMonitor:
    """
    Lightweight, zero-overhead production monitor for SignBridge.

    Designed to run in the main inference loop without blocking.
    All operations are O(1) or O(window_size).

    Parameters
    ----------
    log_dir        : Path — where session logs are saved
    window_size    : int  — rolling window for live stats (default 100 frames)
    confidence_warn: float — log warning when confidence drops below this
    """

    def __init__(
        self,
        log_dir:         Path | str = None,
        window_size:     int        = 100,
        confidence_warn: float      = 0.60,
    ) -> None:
        from src.utils.config_loader import get_project_root
        self.log_dir = Path(log_dir or (get_project_root() / "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.window_size      = window_size
        self.confidence_warn  = confidence_warn
        self._session_start   = time.time()

        # Rolling windows for live display
        self._latencies:    deque[float] = deque(maxlen=window_size)
        self._confidences:  deque[float] = deque(maxlen=window_size)
        self._hand_flags:   deque[int]   = deque(maxlen=window_size)  # 1=detected, 0=not

        # Session-level counters
        self._total_frames:       int   = 0
        self._frames_with_hands:  int   = 0
        self._inference_calls:    int   = 0
        self._total_acceptances:  int   = 0
        self._low_conf_warnings:  int   = 0

        # Per-class acceptance counts
        self._class_counts: dict[str, int] = defaultdict(int)

        # Prediction history (capped to last 1000)
        self._prediction_log: list[dict] = []
        self._max_log_size = 1000

    # ------------------------------------------------------------------
    # Recording methods (call from inference loop)
    # ------------------------------------------------------------------

    def record_frame(
        self,
        detected_hands: int,
        inference_ms:   float | None = None,
    ) -> None:
        """
        Call once per webcam frame.

        Parameters
        ----------
        detected_hands : int   — number of hands detected (0, 1, or 2)
        inference_ms   : float — inference latency in milliseconds (optional)
        """
        self._total_frames += 1
        has_hand = 1 if detected_hands > 0 else 0
        self._hand_flags.append(has_hand)
        self._frames_with_hands += has_hand

        if inference_ms is not None:
            self._latencies.append(inference_ms)
            self._inference_calls += 1

    def record_prediction(
        self,
        class_name:  str,
        confidence:  float,
    ) -> None:
        """
        Call each time the inference engine produces a prediction.

        Parameters
        ----------
        class_name : str   — predicted class label
        confidence : float — softmax confidence [0, 1]
        """
        self._confidences.append(confidence)

        if confidence < self.confidence_warn:
            self._low_conf_warnings += 1
            logger.debug(
                f"[Monitor] Low confidence: {class_name} @ {confidence:.3f}"
            )

        if len(self._prediction_log) < self._max_log_size:
            self._prediction_log.append({
                "t":    round(time.time() - self._session_start, 2),
                "cls":  class_name,
                "conf": round(confidence, 4),
            })

    def record_acceptance(self, class_name: str) -> None:
        """Call when PostProcessor accepts a sign."""
        self._total_acceptances += 1
        self._class_counts[class_name] += 1
        logger.debug(
            f"[Monitor] Accepted: {class_name} "
            f"(total={self._total_acceptances})"
        )

    # ------------------------------------------------------------------
    # Live stats (call from UI draw method)
    # ------------------------------------------------------------------

    @property
    def live_fps(self) -> float:
        """Estimated FPS from inference latency rolling window."""
        if not self._latencies:
            return 0.0
        mean_ms = np.mean(self._latencies)
        return round(1000.0 / max(mean_ms, 0.1), 1)

    @property
    def live_mean_confidence(self) -> float:
        """Rolling mean confidence over last window_size predictions."""
        if not self._confidences:
            return 0.0
        return round(float(np.mean(self._confidences)), 3)

    @property
    def hand_detection_rate(self) -> float:
        """Fraction of frames in rolling window where hands were detected."""
        if not self._hand_flags:
            return 0.0
        return round(float(np.mean(self._hand_flags)), 3)

    @property
    def session_duration_s(self) -> float:
        return round(time.time() - self._session_start, 1)

    def live_stats_str(self) -> str:
        """One-line status string suitable for UI overlay or logging."""
        return (
            f"FPS:{self.live_fps:.0f} | "
            f"Conf:{self.live_mean_confidence:.2f} | "
            f"Hands:{self.hand_detection_rate:.0%} | "
            f"Signs:{self._total_acceptances}"
        )

    # ------------------------------------------------------------------
    # Session summary
    # ------------------------------------------------------------------

    def get_session_summary(self) -> dict:
        """Return a dict summarising the full inference session."""
        duration = self.session_duration_s
        all_lats = list(self._latencies)
        all_confs = self._confidences

        return {
            "session_duration_s":    duration,
            "total_frames":          self._total_frames,
            "frames_with_hands":     self._frames_with_hands,
            "hand_detection_rate":   round(
                self._frames_with_hands / max(self._total_frames, 1), 3
            ),
            "inference_calls":       self._inference_calls,
            "total_acceptances":     self._total_acceptances,
            "acceptance_rate_per_min": round(
                self._total_acceptances / max(duration / 60, 1e-6), 1
            ),
            "low_confidence_warnings": self._low_conf_warnings,
            "mean_latency_ms":       round(float(np.mean(all_lats)), 2)
                                     if all_lats else None,
            "p95_latency_ms":        round(float(np.percentile(all_lats, 95)), 2)
                                     if all_lats else None,
            "mean_confidence":       round(float(np.mean(list(all_confs))), 3)
                                     if all_confs else None,
            "class_acceptance_counts": dict(
                sorted(self._class_counts.items(), key=lambda x: -x[1])
            ),
        }

    def save_session_log(self) -> Path:
        """
        Write full session log to logs/session_<timestamp>.json.

        Returns Path to written file.
        """
        ts       = time.strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"session_{ts}.json"

        payload = {
            "summary":          self.get_session_summary(),
            "prediction_sample": self._prediction_log[-200:],  # last 200
        }

        with open(log_path, "w") as fh:
            json.dump(payload, fh, indent=2)

        logger.info(f"Session log saved → {log_path}")
        return log_path

    def print_session_summary(self) -> None:
        """Print a formatted session summary to stdout."""
        s = self.get_session_summary()
        print("\n" + "=" * 52)
        print("  SignBridge — Session Summary")
        print("=" * 52)
        print(f"  Duration          : {s['session_duration_s']:.0f}s")
        print(f"  Total frames      : {s['total_frames']}")
        print(f"  Hand detection    : {s['hand_detection_rate']:.0%}")
        print(f"  Signs accepted    : {s['total_acceptances']}")
        print(f"  Signs / minute    : {s['acceptance_rate_per_min']:.1f}")
        if s["mean_latency_ms"]:
            print(f"  Mean latency      : {s['mean_latency_ms']:.2f} ms")
            print(f"  P95 latency       : {s['p95_latency_ms']:.2f} ms")
        if s["mean_confidence"]:
            print(f"  Mean confidence   : {s['mean_confidence']:.3f}")
        print(f"  Low conf warnings : {s['low_confidence_warnings']}")
        if s["class_acceptance_counts"]:
            top5 = list(s["class_acceptance_counts"].items())[:5]
            print(f"  Top signs used    : {top5}")
        print("=" * 52 + "\n")
