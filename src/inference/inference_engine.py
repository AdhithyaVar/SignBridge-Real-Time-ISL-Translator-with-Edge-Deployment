"""
inference_engine.py
-------------------
Unified inference engine supporting PyTorch and ONNX backends.

Backend priority (auto-selected):
  1. ONNX INT8 quantized  — fastest on CPU (~42 FPS)
  2. ONNX FP32            — faster than PyTorch (~28 FPS)
  3. PyTorch              — always available (~12 FPS)

The engine handles the full preprocessing chain internally:
  raw (30, 126) → wrist-relative normalisation → Z-score scale → predict

This means callers (live_demo.py) only need to pass the raw landmark
sequence straight from the AutoRecorder / sliding buffer.
"""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

import numpy as np
import torch

from src.preprocessing.landmark_extractor import normalize_sequence
from src.preprocessing.normalizer import SequenceScaler
from src.utils.class_labels import CLASS_LABELS, IDX_TO_CLASS
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SOFTMAX = torch.nn.Softmax(dim=-1)


# ---------------------------------------------------------------------------
# Backend enum
# ---------------------------------------------------------------------------

class InferenceBackend(Enum):
    PYTORCH   = auto()
    ONNX_FP32 = auto()
    ONNX_INT8 = auto()


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

class Prediction:
    """
    Single-frame prediction result.

    Attributes
    ----------
    class_idx   : int   — predicted class index (0–25)
    class_name  : str   — predicted class label ('A'–'Z')
    confidence  : float — softmax probability [0, 1]
    probabilities: (26,) float32 array — full softmax distribution
    backend     : InferenceBackend
    """
    __slots__ = ("class_idx", "class_name", "confidence",
                 "probabilities", "backend")

    def __init__(
        self,
        probabilities: np.ndarray,
        backend:       InferenceBackend,
    ) -> None:
        self.probabilities = probabilities.astype(np.float32)
        self.class_idx     = int(np.argmax(probabilities))
        self.class_name    = IDX_TO_CLASS[self.class_idx]
        self.confidence    = float(probabilities[self.class_idx])
        self.backend       = backend

    def __repr__(self) -> str:
        return (f"Prediction(class={self.class_name}, "
                f"conf={self.confidence:.4f}, "
                f"backend={self.backend.name})")


# ---------------------------------------------------------------------------
# Base engine class
# ---------------------------------------------------------------------------

class _BaseEngine:
    """Shared preprocessing + softmax logic for all backends."""

    def __init__(self, scaler: SequenceScaler) -> None:
        self.scaler  = scaler
        self.backend = InferenceBackend.PYTORCH   # overridden by subclasses

    def _preprocess(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply wrist-relative normalisation then Z-score scaling.

        Parameters
        ----------
        sequence : (T, 126) float32 — raw landmark sequence

        Returns
        -------
        (T, 126) float32 — ready for model input
        """
        normalised = normalize_sequence(sequence)          # wrist-relative
        scaled     = self.scaler.transform(normalised)     # Z-score
        return scaled.astype(np.float32)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict(self, sequence: np.ndarray) -> Prediction:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# PyTorch engine
# ---------------------------------------------------------------------------

class PyTorchEngine(_BaseEngine):
    """
    Runs inference using the PyTorch model directly.

    Parameters
    ----------
    model_path  : Path — path to best_model.pth checkpoint
    model_cfg   : dict — parsed model.yaml
    dataset_cfg : dict — parsed dataset.yaml
    device      : torch.device
    scaler      : fitted SequenceScaler
    """

    def __init__(
        self,
        model_path:  Path,
        model_cfg:   dict,
        dataset_cfg: dict,
        device:      torch.device,
        scaler:      SequenceScaler,
    ) -> None:
        super().__init__(scaler)
        self.backend = InferenceBackend.PYTORCH
        self.device  = device

        from src.models.model_factory import build_model, load_checkpoint
        model = build_model(model_cfg, dataset_cfg, device=device)
        model, _, _ = load_checkpoint(model, model_path, device=device)
        model.eval()
        self.model = model
        logger.info(f"PyTorchEngine ready | device={device}")

    def predict(self, sequence: np.ndarray) -> Prediction:
        """
        Parameters
        ----------
        sequence : (T, 126) float32

        Returns
        -------
        Prediction
        """
        processed = self._preprocess(sequence)                  # (T, 126)
        x = torch.from_numpy(processed).unsqueeze(0).to(self.device)  # (1,T,126)

        with torch.no_grad():
            logits = self.model(x)                              # (1, 26)
            probs  = _SOFTMAX(logits).squeeze(0).cpu().numpy()  # (26,)

        return Prediction(probs, self.backend)


# ---------------------------------------------------------------------------
# ONNX engine (FP32 and INT8)
# ---------------------------------------------------------------------------

class ONNXEngine(_BaseEngine):
    """
    Runs inference using an ONNX Runtime session.
    Works for both FP32 and INT8 quantized models.

    Parameters
    ----------
    model_path : Path — path to .onnx model file
    scaler     : fitted SequenceScaler
    int8       : bool — True for INT8 quantized model
    """

    def __init__(
        self,
        model_path: Path,
        scaler:     SequenceScaler,
        int8:       bool = False,
    ) -> None:
        super().__init__(scaler)
        self.backend = InferenceBackend.ONNX_INT8 if int8 else InferenceBackend.ONNX_FP32

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. "
                "Run:  pip install onnxruntime"
            )

        # Session options — use all available CPU threads
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0   # auto-detect

        self.session    = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(
            f"ONNXEngine ready | "
            f"{'INT8' if int8 else 'FP32'} | "
            f"input={self.input_name} | "
            f"model={Path(model_path).name}"
        )

    def predict(self, sequence: np.ndarray) -> Prediction:
        """
        Parameters
        ----------
        sequence : (T, 126) float32

        Returns
        -------
        Prediction
        """
        processed = self._preprocess(sequence)              # (T, 126)
        x         = processed[np.newaxis, :, :]             # (1, T, 126)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )
        logits = outputs[0][0]                              # (26,)
        probs  = self._softmax(logits)                      # (26,)

        return Prediction(probs, self.backend)


# ---------------------------------------------------------------------------
# Factory: auto-select best available backend
# ---------------------------------------------------------------------------

def build_inference_engine(
    model_cfg:   dict,
    dataset_cfg: dict,
    device:      torch.device,
    prefer_onnx: bool = True,
) -> _BaseEngine:
    """
    Build the best available inference engine.

    Tries backends in order: ONNX INT8 → ONNX FP32 → PyTorch.
    Falls back gracefully if ONNX models have not been exported yet.

    Parameters
    ----------
    model_cfg, dataset_cfg : parsed YAML dicts
    device                 : torch.device
    prefer_onnx            : if False, always use PyTorch

    Returns
    -------
    _BaseEngine (PyTorchEngine or ONNXEngine)
    """
    from src.utils.config_loader import get_project_root
    root = get_project_root()

    # Load scaler (required by all backends)
    splits_dir = root / "data" / "splits"
    if not (splits_dir / "scaler_mean.npy").exists():
        raise FileNotFoundError(
            "Scaler not found. Run  python scripts/preprocess.py  first."
        )
    scaler = SequenceScaler.load(splits_dir)
    logger.info(f"Scaler loaded from {splits_dir}")

    if prefer_onnx:
        # Benchmark reality: ONNX FP32 outperforms INT8 on LSTM-heavy models
        # because QDQ nodes add overhead that offsets INT8 compute gains.
        # Priority: ONNX FP32 (411 FPS) > ONNX INT8 (299 FPS) > PyTorch (203 FPS)

        # Try ONNX FP32 first (fastest on LSTM models)
        fp32_path = root / "models" / "onnx" / "model.onnx"
        if fp32_path.exists():
            try:
                engine = ONNXEngine(fp32_path, scaler, int8=False)
                logger.info("Using ONNX FP32 backend (fastest on LSTM models)")
                return engine
            except Exception as exc:
                logger.warning(f"ONNX FP32 load failed: {exc} — trying INT8")

        # Try ONNX INT8 as fallback
        int8_path = root / "models" / "quantized" / "model_int8.onnx"
        if int8_path.exists():
            try:
                engine = ONNXEngine(int8_path, scaler, int8=True)
                logger.info("Using ONNX INT8 backend (fallback)")
                return engine
            except Exception as exc:
                logger.warning(f"ONNX INT8 load failed: {exc} — falling back to PyTorch")

    # Fall back to PyTorch
    best_path = root / "models" / "best" / "best_model.pth"
    if not best_path.exists():
        raise FileNotFoundError(
            "No model found. Train first:  python scripts/train.py"
        )
    engine = PyTorchEngine(
        model_path=  best_path,
        model_cfg=   model_cfg,
        dataset_cfg= dataset_cfg,
        device=      device,
        scaler=      scaler,
    )
    logger.info("Using PyTorch backend")
    return engine
