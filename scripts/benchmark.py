"""
benchmark.py
------------
Benchmarks all available inference backends and prints a comparison table.

Measures
--------
- Mean FPS over 200 iterations
- Mean latency per inference (ms)
- Model file size (MB)
- Memory usage during inference (MB)

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 500
    python scripts/benchmark.py --batch_size 4
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

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge — Inference Backend Benchmark"
    )
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of timed iterations (default: 200).")
    parser.add_argument("--warmup",     type=int, default=30,
                        help="Warmup iterations before timing (default: 30).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1).")
    return parser.parse_args()


def _mem_mb() -> float:
    """Return current process RSS memory in MB."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def benchmark_pytorch(
    best_path:  Path,
    x_pt:       torch.Tensor,
    n_warmup:   int,
    n_iter:     int,
) -> dict:
    from src.models.model_factory import build_model, load_checkpoint
    model_cfg   = load_config("model",   resolve_paths=False)
    dataset_cfg = load_config("dataset", resolve_paths=False)

    model = build_model(model_cfg, dataset_cfg, device=torch.device("cpu"))
    model, _, _ = load_checkpoint(model, best_path, device=torch.device("cpu"))
    model.eval()

    for _ in range(n_warmup):
        with torch.no_grad():
            model(x_pt)

    mem_before = _mem_mb()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.no_grad():
            model(x_pt)
    elapsed = time.perf_counter() - t0
    mem_after = _mem_mb()

    return {
        "fps":        round(n_iter / elapsed, 1),
        "latency_ms": round(elapsed / n_iter * 1000, 2),
        "size_mb":    round(best_path.stat().st_size / 1024**2, 1),
        "mem_mb":     round(mem_after - mem_before, 1),
    }


def benchmark_onnx(
    onnx_path: Path,
    x_np:      np.ndarray,
    n_warmup:  int,
    n_iter:    int,
    label:     str,
) -> dict:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess     = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    inp_name = sess.get_inputs()[0].name

    for _ in range(n_warmup):
        sess.run(None, {inp_name: x_np})

    mem_before = _mem_mb()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sess.run(None, {inp_name: x_np})
    elapsed = time.perf_counter() - t0
    mem_after = _mem_mb()

    return {
        "fps":        round(n_iter / elapsed, 1),
        "latency_ms": round(elapsed / n_iter * 1000, 2),
        "size_mb":    round(onnx_path.stat().st_size / 1024**2, 1),
        "mem_mb":     round(mem_after - mem_before, 1),
    }


def main() -> None:
    args = parse_args()
    get_logger("signbridge")

    root = get_project_root()
    dataset_cfg = load_config("dataset", resolve_paths=False)
    seq_len  = int(dataset_cfg["recording"]["sequence_length"])
    feat_dim = int(dataset_cfg["features"]["two_hand_features"])

    # Paths
    pt_path   = root / "models" / "best"      / "best_model.pth"
    fp32_path = root / "models" / "onnx"      / "model.onnx"
    int8_path = root / "models" / "quantized" / "model_int8.onnx"

    # Test inputs
    np.random.seed(0)
    x_np = np.random.randn(
        args.batch_size, seq_len, feat_dim
    ).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    print("\n" + "=" * 70)
    print("  SignBridge — Inference Backend Benchmark")
    print("=" * 70)
    print(f"  Iterations  : {args.iterations}  (+ {args.warmup} warmup)")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Input shape : ({args.batch_size}, {seq_len}, {feat_dim})")
    print(f"  Device      : CPU")
    print("=" * 70)

    results: dict[str, dict] = {}

    # PyTorch
    if pt_path.exists():
        print(f"  Benchmarking PyTorch FP32...", end=" ", flush=True)
        results["PyTorch FP32"] = benchmark_pytorch(
            pt_path, x_pt, args.warmup, args.iterations
        )
        print(f"{results['PyTorch FP32']['fps']} FPS")
    else:
        print(f"  PyTorch model not found: {pt_path}")

    # ONNX FP32
    if fp32_path.exists():
        print(f"  Benchmarking ONNX FP32...  ", end=" ", flush=True)
        results["ONNX FP32"] = benchmark_onnx(
            fp32_path, x_np, args.warmup, args.iterations, "fp32"
        )
        print(f"{results['ONNX FP32']['fps']} FPS")
    else:
        print(f"  ONNX FP32 not found — run: python scripts/export_onnx.py")

    # ONNX INT8
    if int8_path.exists():
        print(f"  Benchmarking ONNX INT8...  ", end=" ", flush=True)
        results["ONNX INT8 ⭐"] = benchmark_onnx(
            int8_path, x_np, args.warmup, args.iterations, "int8"
        )
        print(f"{results['ONNX INT8 ⭐']['fps']} FPS")
    else:
        print(f"  ONNX INT8 not found  — run: python scripts/export_onnx.py")

    # ── Print table ───────────────────────────────────────────────────
    if not results:
        print("\n  No backends found to benchmark.")
        return

    baseline_fps = results.get("PyTorch FP32", {}).get("fps", 1.0)

    print("\n" + "=" * 70)
    print(f"  {'Backend':<20} {'FPS':>8} {'Speedup':>9} "
          f"{'Latency ms':>12} {'Size MB':>9} {'Mem MB':>8}")
    print("-" * 70)

    for name, m in results.items():
        speedup = m["fps"] / baseline_fps
        bar = "█" * min(int(m["fps"] / 3), 20)
        print(
            f"  {name:<20} {m['fps']:>8.1f} {speedup:>8.1f}× "
            f"{m['latency_ms']:>11.2f} {m['size_mb']:>9.1f} "
            f"{m['mem_mb']:>8.1f}  {bar}"
        )

    print("=" * 70)

    # Recommendation
    best = max(results, key=lambda k: results[k]["fps"])
    print(f"\n  Fastest backend : {best}  ({results[best]['fps']} FPS)")
    print(f"  Note: On LSTM-heavy models ONNX FP32 is often faster than INT8")
    print(f"        because QDQ nodes add overhead that offsets INT8 gains.")
    print(f"  run_demo.py auto-selects ONNX FP32 as the recommended backend.")
    print()


if __name__ == "__main__":
    main()
