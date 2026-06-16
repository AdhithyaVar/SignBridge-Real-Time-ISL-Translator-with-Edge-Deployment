"""
run_demo.py
-----------
CLI entry-point for the SignBridge real-time ISL recognition demo.

What this script does
---------------------
1.  Loads dataset.yaml, model.yaml
2.  Auto-selects best available inference backend:
      ONNX FP32 (fastest) -> ONNX INT8 -> PyTorch (always available)
3.  Optionally loads a specific checkpoint (--checkpoint)
4.  Builds PostProcessor with configurable thresholds
5.  Opens webcam and runs real-time recognition loop
6.  Displays MediaPipe skeleton, predictions, confidence, sentence
7.  Speaks accepted signs via TTS (pyttsx3)
8.  Saves session monitoring log on exit

Usage
-----
    # Standard demo (auto backend selection)
    python scripts/run_demo.py

    # Use fine-tuned model
    python scripts/run_demo.py --checkpoint models/finetuned/finetuned_model.pth

    # Force PyTorch backend (skip ONNX)
    python scripts/run_demo.py --no_onnx

    # Use a different camera
    python scripts/run_demo.py --camera 1

    # Adjust confidence threshold (lower = more sensitive)
    python scripts/run_demo.py --confidence 0.60

    # Run inference every 2 frames (faster on slow CPUs)
    python scripts/run_demo.py --infer_every 2

    # Disable TTS at startup
    python scripts/run_demo.py --no_tts

Keyboard controls (during demo)
---------------------------------
    Q      - quit
    C      - clear the sentence
    SPACE  - full reset (sentence + post-processor + buffer)
    S      - toggle TTS on/off
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config_loader import load_config, get_project_root
from src.utils.logger import get_logger
from src.utils.device import get_device_from_config
from src.inference.inference_engine import build_inference_engine
from src.inference.post_processor import build_post_processor
from src.inference.live_demo import LiveDemo

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SignBridge -- Real-Time ISL Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_demo.py
  python scripts/run_demo.py --checkpoint models/finetuned/finetuned_model.pth
  python scripts/run_demo.py --no_onnx
  python scripts/run_demo.py --camera 1
  python scripts/run_demo.py --confidence 0.65 --infer_every 2
  python scripts/run_demo.py --no_tts
        """,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a specific model checkpoint. "
            "When set, forces PyTorch backend with that checkpoint. "
            "Examples: models/best/best_model.pth, "
            "models/finetuned/finetuned_model.pth"
        ),
    )
    parser.add_argument(
        "--camera", "-k",
        type=int,
        default=None,
        help="Webcam device index (default from dataset.yaml, usually 0).",
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.70,
        help="Minimum confidence threshold for sign acceptance (default: 0.70).",
    )
    parser.add_argument(
        "--stable_frames",
        type=int,
        default=8,
        help="Consecutive stable predictions before accepting a sign (default: 8).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="Seconds between sign acceptances (default: 1.5).",
    )
    parser.add_argument(
        "--infer_every",
        type=int,
        default=1,
        help="Run inference every N frames (default: 1, reduce for slow CPUs).",
    )
    parser.add_argument(
        "--no_onnx",
        action="store_true",
        help="Force PyTorch backend (do not attempt ONNX).",
    )
    parser.add_argument(
        "--no_tts",
        action="store_true",
        help="Disable text-to-speech output at startup.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_prerequisites() -> None:
    """Verify all required artefacts exist before opening the webcam."""
    root = _PROJECT_ROOT
    required = {
        "Best model":  root / "models" / "best" / "best_model.pth",
        "Scaler mean": root / "data" / "splits" / "scaler_mean.npy",
        "Scaler std":  root / "data" / "splits" / "scaler_std.npy",
    }
    missing = []
    for name, path in required.items():
        if not path.exists():
            missing.append(f"  {name}: {path}")

    if missing:
        print("\n  Missing required files:")
        for m in missing:
            print(m)
        print("\n  Run the pipeline in order:")
        print("    1.  python scripts/collect_data.py")
        print("    2.  python scripts/preprocess.py")
        print("    3.  python scripts/train.py")
        print("    4.  python scripts/run_demo.py\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Demo startup banner
# ---------------------------------------------------------------------------

def print_banner(
    backend_name: str,
    camera_idx:   int,
    checkpoint:   str | None,
    args,
) -> None:
    print("\n" + "=" * 58)
    print("  SignBridge -- Real-Time ISL Recognition")
    print("=" * 58)
    print(f"  Inference backend : {backend_name}")
    if checkpoint:
        print(f"  Checkpoint        : {Path(checkpoint).name}")
    print(f"  Camera index      : {camera_idx}")
    print(f"  Confidence thresh : {args.confidence}")
    print(f"  Stable frames     : {args.stable_frames}")
    print(f"  Cooldown (s)      : {args.cooldown}")
    print(f"  Infer every N     : {args.infer_every}")
    print(f"  TTS               : {'OFF (--no_tts)' if args.no_tts else 'ON'}")
    print("-" * 58)
    print("  Controls:")
    print("    Q      -- quit")
    print("    C      -- clear sentence")
    print("    SPACE  -- full reset")
    print("    S      -- toggle TTS on/off")
    print("=" * 58 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    import logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    get_logger("signbridge", log_level=log_level)

    # Pre-flight
    check_prerequisites()

    # Load configs
    dataset_cfg  = load_config("dataset")
    model_cfg    = load_config("model",    resolve_paths=False)
    training_cfg = load_config("training", resolve_paths=False)

    # Camera index
    camera_index = args.camera
    if camera_index is None:
        camera_index = int(
            dataset_cfg.get("recording", {}).get("camera_index", 0)
        )

    # Device
    device = get_device_from_config(training_cfg)

    # Build inference engine
    logger.info("Loading inference engine...")

    if args.checkpoint:
        # Custom checkpoint -> always use PyTorch backend
        # ONNX models may not match the custom checkpoint weights
        from src.inference.inference_engine import PyTorchEngine
        from src.preprocessing.normalizer import SequenceScaler

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _PROJECT_ROOT / ckpt_path

        if not ckpt_path.exists():
            logger.error(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Available checkpoints:\n"
                f"  models/best/best_model.pth\n"
                f"  models/finetuned/finetuned_model.pth"
            )
            sys.exit(1)

        splits_dir = get_project_root() / "data" / "splits"
        scaler     = SequenceScaler.load(splits_dir)
        engine     = PyTorchEngine(
            model_path=  ckpt_path,
            model_cfg=   model_cfg,
            dataset_cfg= dataset_cfg,
            device=      device,
            scaler=      scaler,
        )
        logger.info(f"Custom checkpoint: {ckpt_path.name} (PyTorch backend)")
        backend_name = "PYTORCH (custom)"

    else:
        # Auto backend selection: ONNX FP32 > ONNX INT8 > PyTorch
        try:
            engine = build_inference_engine(
                model_cfg=   model_cfg,
                dataset_cfg= dataset_cfg,
                device=      device,
                prefer_onnx= not args.no_onnx,
            )
        except FileNotFoundError as exc:
            logger.error(str(exc))
            sys.exit(1)

        backend_name = getattr(
            getattr(engine, "backend", None), "name", "PYTORCH"
        )

    # Build post-processor
    post_processor = build_post_processor({
        "confidence_threshold": args.confidence,
        "required_stable":      args.stable_frames,
        "cooldown_seconds":     args.cooldown,
        "smoothing_window":     5,
        "idle_reset_secs":      8.0,
    })

    # Print banner
    print_banner(backend_name, camera_index, args.checkpoint, args)

    # Build and run demo
    demo = LiveDemo(
        engine=         engine,
        post_processor= post_processor,
        dataset_cfg=    dataset_cfg,
        camera_index=   camera_index,
        infer_every_n=  args.infer_every,
    )

    # Disable TTS at startup if requested
    if args.no_tts:
        demo._tts.toggle()

    try:
        demo.run()
    except RuntimeError as exc:
        logger.error(f"Demo error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Demo interrupted by Ctrl+C.")


if __name__ == "__main__":
    main()
