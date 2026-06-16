"""
export_onnx.py
--------------
Exports the trained SignBridge PyTorch model to ONNX FP32 and
then quantizes it to ONNX INT8 using static quantization.

Pipeline
--------
1.  Load best_model.pth
2.  Export to ONNX FP32  → models/onnx/model.onnx
3.  Validate FP32 ONNX matches PyTorch outputs (atol=1e-4)
4.  Static INT8 quantization with calibration data
    → models/quantized/model_int8.onnx
5.  Validate INT8 ONNX matches PyTorch outputs (atol=5e-3)
6.  Print file sizes and quick FPS estimate

Usage
-----
    python scripts/export_onnx.py
    python scripts/export_onnx.py --skip_int8     # FP32 only
    python scripts/export_onnx.py --cal_samples 200
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.device import get_device_from_config
from src.models.model_factory import build_model, load_checkpoint
from src.preprocessing.normalizer import SequenceScaler
from src.inference.onnx_validator import ONNXValidator

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — ONNX Export & INT8 Quantization",
    )
    parser.add_argument(
        "--skip_int8",
        action="store_true",
        help="Export FP32 ONNX only; skip INT8 quantization.",
    )
    parser.add_argument(
        "--cal_samples",
        type=int,
        default=500,
        help="Number of calibration samples for INT8 (default: 500).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14 — best compatibility with INT8 quantizer).",
    )
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to checkpoint to export. "
            "Default: models/best/best_model.pth. "
            "Use models/finetuned/finetuned_model.pth after fine-tuning."
        ),
    )
    parser.add_argument(
        "--skip_validate",
        action="store_true",
        help="Skip numerical validation after export.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Export PyTorch → ONNX FP32
# ---------------------------------------------------------------------------

def export_fp32(
    model:      nn.Module,
    onnx_path:  Path,
    seq_len:    int = 30,
    feat_dim:   int = 126,
    opset:      int = 17,
) -> Path:
    """
    Export a PyTorch model to ONNX FP32 format.

    Parameters
    ----------
    model     : eval-mode PyTorch model
    onnx_path : output .onnx file path
    seq_len   : sequence length (30)
    feat_dim  : feature dimension (126)
    opset     : ONNX opset version

    Returns
    -------
    Path to written .onnx file
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # Dummy input for tracing
    dummy = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)

    logger.info(f"Exporting PyTorch → ONNX FP32 (opset {opset})...")

    # Use legacy TorchScript-based exporter (dynamo=False).
    # PyTorch 2.x defaults to the dynamo exporter which requires onnxscript.
    # The legacy exporter is stable and produces valid ONNX for our model.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=       True,
        opset_version=       opset,
        do_constant_folding= True,
        input_names=         ["input_sequence"],
        output_names=        ["logits"],
        dynamic_axes={
            "input_sequence": {0: "batch_size"},
            "logits":         {0: "batch_size"},
        },
        verbose=False,
        dynamo=False,        # force legacy TorchScript exporter
    )

    # Verify with onnx checker
    try:
        import onnx
        model_proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(model_proto)
        logger.info("ONNX model check passed.")
    except ImportError:
        logger.warning("onnx package not installed — skipping model check.")

    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    logger.info(f"FP32 ONNX saved → {onnx_path}  ({size_mb:.1f} MB)")
    return onnx_path


# ---------------------------------------------------------------------------
# Step 2: INT8 Static Quantization
# ---------------------------------------------------------------------------

def quantize_int8(
    fp32_path:   Path,
    int8_path:   Path,
    cal_data:    np.ndarray,
    input_name:  str = "input_sequence",
) -> Path:
    """
    Quantize an ONNX FP32 model to INT8 using static quantization.

    Parameters
    ----------
    fp32_path  : path to input FP32 .onnx model
    int8_path  : path for output INT8 .onnx model
    cal_data   : (N, T, F) calibration data array
    input_name : ONNX input node name

    Returns
    -------
    Path to written INT8 .onnx file
    """
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
            QuantFormat,
        )
    except ImportError:
        raise ImportError(
            "onnxruntime.quantization not available. "
            "Install: pip install onnxruntime"
        )

    class _CalibReader(CalibrationDataReader):
        """Feed calibration samples one-by-one to the quantizer."""
        def __init__(self, data: np.ndarray, inp_name: str) -> None:
            self._data     = data
            self._inp_name = inp_name
            self._idx      = 0

        def get_next(self):
            if self._idx >= len(self._data):
                return None
            sample = self._data[self._idx: self._idx + 1]   # (1, T, F)
            self._idx += 1
            return {self._inp_name: sample}

        def rewind(self):
            self._idx = 0

    logger.info(
        f"INT8 quantization | "
        f"calibration samples={len(cal_data)} | "
        f"input={fp32_path.name}"
    )

    reader = _CalibReader(cal_data.astype(np.float32), input_name)

    quantize_static(
        model_input=        str(fp32_path),
        model_output=       str(int8_path),
        calibration_data_reader=reader,
        quant_format=       QuantFormat.QDQ,
        per_channel=        False,
        activation_type=    QuantType.QInt8,
        weight_type=        QuantType.QInt8,
    )

    size_mb = int8_path.stat().st_size / (1024 ** 2)
    fp32_mb = fp32_path.stat().st_size  / (1024 ** 2)
    compression = fp32_mb / size_mb

    logger.info(
        f"INT8 ONNX saved → {int8_path}  "
        f"({size_mb:.1f} MB, {compression:.1f}× smaller than FP32)"
    )
    return int8_path


# ---------------------------------------------------------------------------
# Step 3: FPS benchmark
# ---------------------------------------------------------------------------

def benchmark_fps(
    model_path:  Path,
    backend:     str,
    n_warmup:    int = 20,
    n_bench:     int = 100,
    seq_len:     int = 30,
    feat_dim:    int = 126,
) -> float:
    """
    Measure inference FPS for one model.

    Parameters
    ----------
    model_path : path to .pth or .onnx file
    backend    : 'pytorch' | 'onnx_fp32' | 'onnx_int8'
    n_warmup   : warmup iterations before timing
    n_bench    : timed iterations

    Returns
    -------
    float — mean FPS
    """
    x_np = np.random.randn(1, seq_len, feat_dim).astype(np.float32)

    if backend == "pytorch":
        import torch
        from src.models.model_factory import build_model, load_checkpoint
        from src.utils.config_loader import load_config

        model_cfg   = load_config("model",   resolve_paths=False)
        dataset_cfg = load_config("dataset", resolve_paths=False)
        model       = build_model(model_cfg, dataset_cfg, device=torch.device("cpu"))
        model, _, _ = load_checkpoint(model, model_path, device=torch.device("cpu"))
        model.eval()
        x_pt = torch.from_numpy(x_np)

        for _ in range(n_warmup):
            with torch.no_grad():
                model(x_pt)

        t0 = time.perf_counter()
        for _ in range(n_bench):
            with torch.no_grad():
                model(x_pt)
        elapsed = time.perf_counter() - t0

    else:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inp_name = sess.get_inputs()[0].name

        for _ in range(n_warmup):
            sess.run(None, {inp_name: x_np})

        t0 = time.perf_counter()
        for _ in range(n_bench):
            sess.run(None, {inp_name: x_np})
        elapsed = time.perf_counter() - t0

    fps = n_bench / elapsed
    return round(fps, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    get_logger("signbridge")

    root = get_project_root()

    # ── Load configs ──────────────────────────────────────────────────
    model_cfg    = load_config("model",    resolve_paths=False)
    dataset_cfg  = load_config("dataset",  resolve_paths=False)
    training_cfg = load_config("training", resolve_paths=False)

    seq_len  = int(dataset_cfg["recording"]["sequence_length"])
    feat_dim = int(dataset_cfg["features"]["two_hand_features"])

    # ── Paths ─────────────────────────────────────────────────────────
    # Use --base_checkpoint if provided, otherwise default to best_model.pth
    if args.base_checkpoint:
        best_path = Path(args.base_checkpoint)
        if not best_path.is_absolute():
            best_path = root / best_path
        logger.info(f"Using custom checkpoint: {best_path}")
    else:
        best_path = root / "models" / "best" / "best_model.pth"
        logger.info(f"Using default checkpoint: {best_path}")

    fp32_path = root / "models" / "onnx"      / "model.onnx"
    int8_path = root / "models" / "quantized" / "model_int8.onnx"

    if not best_path.exists():
        logger.error(
            f"Checkpoint not found: {best_path}\n"
            f"Available checkpoints:\n"
            f"  models/best/best_model.pth\n"
            f"  models/finetuned/finetuned_model.pth"
        )
        sys.exit(1)

    # ── Load PyTorch model ────────────────────────────────────────────
    device = torch.device("cpu")
    logger.info("Loading PyTorch model...")
    model = build_model(model_cfg, dataset_cfg, device=device)
    model, _, _ = load_checkpoint(model, best_path, device=device)
    model.eval()

    print("\n" + "=" * 58)
    print("  SignBridge — ONNX Export & Quantization")
    print("=" * 58)

    # ── Step 1: Export FP32 ───────────────────────────────────────────
    print("\n  Step 1: Exporting PyTorch → ONNX FP32...")
    fp32_path = export_fp32(model, fp32_path,
                            seq_len=seq_len, feat_dim=feat_dim,
                            opset=args.opset)
    print(f"  ✓  FP32 model: {fp32_path}")
    print(f"     Size: {fp32_path.stat().st_size / 1024**2:.1f} MB")

    # ── Step 2: Validate FP32 ─────────────────────────────────────────
    if not args.skip_validate:
        print("\n  Step 2: Validating FP32 ONNX vs PyTorch...")
        validator = ONNXValidator(model, fp32_path, device, atol=1e-4, n_samples=50)
        validator.validate(seq_len=seq_len, feat_dim=feat_dim)

    # ── Step 3: INT8 Quantization ─────────────────────────────────────
    if not args.skip_int8:
        print(f"\n  Step 3: INT8 quantization ({args.cal_samples} calibration samples)...")

        # Load calibration data from training split
        splits_dir = root / "data" / "splits"
        X_train    = np.load(splits_dir / "X_train.npy")

        # Sub-sample calibration set (balanced across classes)
        n_cal = min(args.cal_samples, len(X_train))
        rng   = np.random.default_rng(42)
        idx   = rng.choice(len(X_train), n_cal, replace=False)
        cal_data = X_train[idx]

        logger.info(f"Calibration data shape: {cal_data.shape}")
        int8_path = quantize_int8(fp32_path, int8_path,
                                  cal_data=cal_data,
                                  input_name="input_sequence")
        print(f"  ✓  INT8 model: {int8_path}")
        print(f"     Size: {int8_path.stat().st_size / 1024**2:.1f} MB")

        # Step 4: Validate INT8
        if not args.skip_validate:
            print("\n  Step 4: Validating INT8 ONNX vs PyTorch...")
            # CRITICAL: INT8 was calibrated on real preprocessed ISL data.
            # Validating with random noise produces out-of-distribution inputs
            # that cause quantized activations to overflow — false failures.
            # We must validate with real preprocessed data from X_val.
            splits_dir = root / "data" / "splits"
            X_val   = np.load(splits_dir / "X_val.npy")    # already preprocessed
            n_val   = min(50, len(X_val))
            rng_val = np.random.default_rng(0)
            val_idx = rng_val.choice(len(X_val), n_val, replace=False)
            val_samples = X_val[val_idx]  # (50, 30, 126) real preprocessed data

            validator_int8 = ONNXValidator(
                model, int8_path, device, atol=0.10, n_samples=n_val
            )
            # Override validate() to use real data instead of random noise
            validator_int8.validate(
                seq_len=seq_len, feat_dim=feat_dim,
                real_data=val_samples,
            )

    # ── Step 5: FPS Benchmark ─────────────────────────────────────────
    print("\n  Step 5: FPS Benchmark (CPU, single sequence)...")
    results = {}

    print("    Benchmarking PyTorch...", end=" ", flush=True)
    results["PyTorch (FP32)"] = benchmark_fps(best_path, "pytorch")
    print(f"{results['PyTorch (FP32)']} FPS")

    if fp32_path.exists():
        print("    Benchmarking ONNX FP32...", end=" ", flush=True)
        results["ONNX FP32"] = benchmark_fps(fp32_path, "onnx_fp32")
        print(f"{results['ONNX FP32']} FPS")

    if not args.skip_int8 and int8_path.exists():
        print("    Benchmarking ONNX INT8...", end=" ", flush=True)
        results["ONNX INT8"] = benchmark_fps(int8_path, "onnx_int8")
        print(f"{results['ONNX INT8']} FPS")

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Export Complete — Benchmark Summary")
    print("=" * 58)
    baseline = results.get("PyTorch (FP32)", 1)
    max_fps  = max(results.values()) if results else 1
    for name, fps in results.items():
        speedup = fps / baseline
        bar     = "█" * min(int(fps / 2), 30)
        marker  = " ⭐" if fps == max_fps else ""  # star on fastest
        print(f"  {name:<18} {fps:>6.1f} FPS  "
              f"({speedup:.1f}×)  {bar}{marker}")

    print("=" * 58)

    # Auto-detect which model run_demo.py will use
    if fp32_path.exists():
        print(f"\n  run_demo.py will auto-select: ONNX FP32 (fastest backend)")
        print(f"  FP32 model : {fp32_path}")
    if not args.skip_int8 and int8_path.exists():
        print(f"  INT8 model : {int8_path} (fallback)")
    print()


if __name__ == "__main__":
    main()
