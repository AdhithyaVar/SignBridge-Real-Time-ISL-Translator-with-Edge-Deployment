"""
auto_recorder.py
----------------
Trigger-based automatic sample recording logic.

Responsibilities
----------------
* Track a rolling buffer of the last N frames
* Decide WHEN to start and stop a recording sequence automatically:
    - Start : hand(s) stable and correct count for N consecutive frames
    - Stop  : sequence_length frames collected
* Enforce a cooldown period between consecutive auto-recordings
* Save completed sequences as .npy files to the raw data directory
* Return a status enum for each frame so the caller can update the UI

State machine
-------------
    IDLE  →  WAITING (correct hands detected)
          →  IDLE    (wrong hand count or no hands)
WAITING  →  RECORDING (held for hold_frames consecutive frames)
          →  IDLE      (hands lost or wrong count)
RECORDING →  IDLE (sequence_length frames collected → save)
           →  IDLE (hands lost mid-sequence → discard)
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum, auto
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Recorder states
# ---------------------------------------------------------------------------

class RecorderState(Enum):
    IDLE      = auto()   # Waiting for correct hands to appear
    WAITING   = auto()   # Correct hands detected; counting hold_frames
    RECORDING = auto()   # Actively collecting frames into current sequence
    COOLDOWN  = auto()   # Just saved a sequence; brief pause before next


# ---------------------------------------------------------------------------
# AutoRecorder
# ---------------------------------------------------------------------------

class AutoRecorder:
    """
    Manages the automatic recording lifecycle for one sign class session.

    Parameters
    ----------
    save_dir : Path
        Directory where recorded .npy files will be saved.
    class_name : str
        Name of the current ISL class being collected.
    sequence_length : int
        Number of frames per recorded sequence.
    num_sequences : int
        Target number of sequences to record for this session.
    hold_frames : int
        Consecutive frames with correct hands required before recording starts.
    cooldown_seconds : float
        Seconds to wait after saving before the next recording can start.
    """

    def __init__(
        self,
        save_dir:        Path,
        class_name:      str,
        sequence_length: int   = 30,
        num_sequences:   int   = 100,
        hold_frames:     int   = 10,
        cooldown_seconds:float = 1.5,
    ) -> None:
        self.save_dir         = Path(save_dir)
        self.class_name       = class_name
        self.sequence_length  = sequence_length
        self.num_sequences    = num_sequences
        self.hold_frames      = hold_frames
        self.cooldown_seconds = cooldown_seconds

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Determine next sequence index (resume if directory has existing files)
        existing = sorted(self.save_dir.glob(f"{class_name}_*.npy"))
        self._next_idx = len(existing)

        # Rolling buffers
        self._hold_counter:  int            = 0
        self._frame_buffer:  list[np.ndarray] = []   # frames collected so far
        self._cooldown_until: float         = 0.0   # epoch time

        self._state = RecorderState.IDLE

        logger.info(
            f"AutoRecorder ready | class={class_name} | "
            f"target={num_sequences} seqs | "
            f"resume_from={self._next_idx} existing sequences"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def sequences_saved(self) -> int:
        """Number of sequences successfully saved this session (+ pre-existing)."""
        return self._next_idx

    @property
    def is_complete(self) -> bool:
        """True when the target number of sequences has been reached."""
        return self._next_idx >= self.num_sequences

    @property
    def progress(self) -> tuple[int, int]:
        """Return (sequences_saved, num_sequences) for progress display."""
        return (self._next_idx, self.num_sequences)

    def update(
        self,
        feature_vector: np.ndarray,
        hand_count_ok: bool,
    ) -> RecorderState:
        """
        Process one frame.  Call once per frame with the 126-dim feature
        vector and a boolean indicating whether the hand count is correct.

        Parameters
        ----------
        feature_vector : np.ndarray, shape (126,)
            Output of HandDetector._build_feature_vector for this frame.
        hand_count_ok : bool
            True if the number of detected hands matches the target class.

        Returns
        -------
        RecorderState
            Current state after processing this frame.
        """
        now = time.time()

        # ------------------------------------------------------------------
        # COOLDOWN: skip until cooldown period expires
        # ------------------------------------------------------------------
        if self._state == RecorderState.COOLDOWN:
            if now >= self._cooldown_until:
                self._state = RecorderState.IDLE
            else:
                return self._state

        # ------------------------------------------------------------------
        # IDLE: wait for correct hand configuration
        # ------------------------------------------------------------------
        if self._state == RecorderState.IDLE:
            if hand_count_ok:
                self._hold_counter = 1
                self._state = RecorderState.WAITING
            return self._state

        # ------------------------------------------------------------------
        # WAITING: count consecutive frames with correct hands
        # ------------------------------------------------------------------
        if self._state == RecorderState.WAITING:
            if hand_count_ok:
                self._hold_counter += 1
                if self._hold_counter >= self.hold_frames:
                    # Transition to recording
                    self._frame_buffer = []
                    self._hold_counter = 0
                    self._state = RecorderState.RECORDING
                    logger.debug(
                        f"[{self.class_name}] Recording started "
                        f"(seq {self._next_idx + 1}/{self.num_sequences})"
                    )
            else:
                # Lost hands — reset
                self._hold_counter = 0
                self._state = RecorderState.IDLE
            return self._state

        # ------------------------------------------------------------------
        # RECORDING: accumulate frames until sequence_length is reached
        # ------------------------------------------------------------------
        if self._state == RecorderState.RECORDING:
            if not hand_count_ok:
                # Hands lost mid-sequence — discard and restart
                logger.debug(
                    f"[{self.class_name}] Sequence discarded (hands lost at "
                    f"frame {len(self._frame_buffer)}/{self.sequence_length})"
                )
                self._frame_buffer = []
                self._state = RecorderState.IDLE
                return self._state

            self._frame_buffer.append(feature_vector.copy())

            if len(self._frame_buffer) == self.sequence_length:
                self._save_sequence()
                self._frame_buffer = []
                self._cooldown_until = now + self.cooldown_seconds
                self._state = RecorderState.COOLDOWN

        return self._state

    def reset_session(self) -> None:
        """Reset in-progress recording state without altering saved files."""
        self._frame_buffer  = []
        self._hold_counter  = 0
        self._state         = RecorderState.IDLE
        logger.info(f"[{self.class_name}] Session reset.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_sequence(self) -> None:
        """
        Save the completed sequence buffer as a .npy file.

        File naming: <class_name>_<zero_padded_index>.npy
        Array shape: (sequence_length, 126)  dtype float32
        """
        sequence = np.array(self._frame_buffer, dtype=np.float32)
        # shape: (30, 126)

        filename = f"{self.class_name}_{self._next_idx:04d}.npy"
        filepath = self.save_dir / filename
        np.save(filepath, sequence)
        self._next_idx += 1

        logger.debug(
            f"[{self.class_name}] Saved {filename} | "
            f"shape={sequence.shape} | "
            f"progress={self._next_idx}/{self.num_sequences}"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_auto_recorder(
    class_name:   str,
    dataset_cfg:  dict,
) -> AutoRecorder:
    """
    Build an AutoRecorder from a parsed dataset.yaml config dict.

    Parameters
    ----------
    class_name : str
        The ISL class currently being collected.
    dataset_cfg : dict
        Parsed dataset.yaml config.

    Returns
    -------
    AutoRecorder
    """
    rec_cfg  = dataset_cfg.get("recording", {})
    raw_dir  = dataset_cfg["paths"]["raw_data"] / class_name

    return AutoRecorder(
        save_dir=        raw_dir,
        class_name=      class_name,
        sequence_length= int(rec_cfg.get("sequence_length", 30)),
        num_sequences=   int(rec_cfg.get("num_sequences",   100)),
        hold_frames=     int(rec_cfg.get("hold_frames",     10)),
        cooldown_seconds=float(rec_cfg.get("countdown_seconds", 1.5)),
    )
