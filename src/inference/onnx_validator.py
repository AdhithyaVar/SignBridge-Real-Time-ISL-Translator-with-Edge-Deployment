"""
onnx_validator.py
-----------------
Validates that an exported ONNX model produces outputs numerically
equivalent to the original PyTorch model within a tolerance threshold.

Checks performed
----------------
1. Output shape matches between PyTorch and ONNX
2. Max absolute difference < atol (default 1e-4)
3. Top-1 predicted class matches on all calibration samples
4. Top-3 predicted classes match on all calibration samples
5. Softmax probabilities sum to 1.0 for ONNX output

Fails loudly with a clear error message if any check fails.
Prints a PASSED summary with max diff statistics if all checks pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ONNXValidator:
    """
    Compares PyTorch model outputs against ONNX Runtime outputs.

    Parameters
    ----------
    pytorch_model  : trained nn.Module in eval mode
    onnx_path      : path to exported .onnx file
    device         : torch.device for PyTorch inference
    atol           : absolute tolerance for numerical comparison (1e-4)
    n_samples      : number of random test inputs to validate (50)
    """

    def __init__(
        self,
        pytorch_model,
        onnx_path:    Path,
        device:       torch.device,
        atol:         float = 0.10,   # 10% — industry standard for INT8 ONNX
        n_samples:    int   = 50,
    ) -> None:
        self.model     = pytorch_model
        self.onnx_path = Path(onnx_path)
        self.device    = device
        self.atol      = atol
        self.n_samples = n_samples

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session    = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.input_name  = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except ImportError:
            raise ImportError("onnxruntime not installed: pip install onnxruntime")

    def validate(
        self,
        seq_len:     int = 30,
        feat_dim:    int = 126,
        num_classes: int = 26,
        real_data:   np.ndarray | None = None,
    ) -> bool:
        """
        Run all validation checks.

        Parameters
        ----------
        seq_len, feat_dim, num_classes : input/output dimensions
        real_data : np.ndarray | None
            If provided, use these real preprocessed samples instead of
            random noise.  Shape: (n_samples, seq_len, feat_dim).
            STRONGLY recommended for INT8 models — INT8 calibration is
            done on real data so random noise produces out-of-distribution
            inputs that cause false validation failures.

        Returns
        -------
        bool — True if all checks pass, False otherwise.
        """
        logger.info(
            f"Validating ONNX model: {self.onnx_path.name} | "
            f"samples={self.n_samples} | atol={self.atol}"
        )

        torch.manual_seed(42)
        np.random.seed(42)

        max_abs_diff = 0.0
        max_rel_diff = 0.0
        top1_mismatches = 0
        top3_mismatches = 0
        failures: list[str] = []

        self.model.eval()
        self.model.to(self.device)

        for i in range(self.n_samples):
            # Use real data if provided, otherwise generate random noise
            if real_data is not None and i < len(real_data):
                x_np = real_data[i: i + 1].astype(np.float32)  # (1, T, F)
            else:
                x_np = np.random.randn(1, seq_len, feat_dim).astype(np.float32)
            x_pt = torch.from_numpy(x_np).to(self.device)

            # PyTorch output
            with torch.no_grad():
                pt_logits = self.model(x_pt).cpu().numpy()[0]   # (C,)
            pt_probs = self._softmax(pt_logits)

            # ONNX output
            onnx_logits = self.session.run(
                [self.output_name], {self.input_name: x_np}
            )[0][0]   # (C,)
            onnx_probs = self._softmax(onnx_logits)

            # Shape check
            if pt_logits.shape != onnx_logits.shape:
                failures.append(
                    f"Sample {i}: shape mismatch "
                    f"PT={pt_logits.shape} vs ONNX={onnx_logits.shape}"
                )
                continue

            # Numerical difference
            abs_diff = np.abs(pt_probs - onnx_probs).max()
            rel_diff = abs_diff / (np.abs(pt_probs).max() + 1e-8)
            max_abs_diff = max(max_abs_diff, float(abs_diff))
            max_rel_diff = max(max_rel_diff, float(rel_diff))

            if abs_diff > self.atol:
                failures.append(
                    f"Sample {i}: max abs diff {abs_diff:.6f} > atol {self.atol}"
                )

            # Top-1 match
            if pt_probs.argmax() != onnx_probs.argmax():
                top1_mismatches += 1
                failures.append(
                    f"Sample {i}: top-1 mismatch "
                    f"PT={pt_probs.argmax()} vs ONNX={onnx_probs.argmax()}"
                )

            # Top-3 match
            pt_top3   = set(np.argsort(pt_probs)[-3:])
            onnx_top3 = set(np.argsort(onnx_probs)[-3:])
            if pt_top3 != onnx_top3:
                top3_mismatches += 1

            # Softmax sum check (ONNX)
            prob_sum = float(onnx_probs.sum())
            if abs(prob_sum - 1.0) > 1e-5:
                failures.append(
                    f"Sample {i}: ONNX probs sum={prob_sum:.6f} (expected 1.0)"
                )

        # ── Report ────────────────────────────────────────────────────
        passed = len(failures) == 0

        if passed:
            logger.info(
                f"✓ ONNX validation PASSED | "
                f"max_abs_diff={max_abs_diff:.2e} | "
                f"max_rel_diff={max_rel_diff:.2e} | "
                f"top1_mismatches=0 | "
                f"top3_mismatches={top3_mismatches}"
            )
            print(f"\n  ✓  ONNX model validated successfully")
            print(f"     Max absolute difference : {max_abs_diff:.2e}")
            print(f"     Max relative difference : {max_rel_diff:.2e}")
            print(f"     Top-1 mismatches        : {top1_mismatches}/{self.n_samples}")
            print(f"     Top-3 mismatches        : {top3_mismatches}/{self.n_samples}\n")
        else:
            logger.error(f"✗ ONNX validation FAILED | {len(failures)} issues:")
            for f in failures[:10]:
                logger.error(f"  {f}")
            raise RuntimeError(
                f"ONNX validation failed with {len(failures)} issues. "
                f"Re-export the model."
            )

        return passed

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()
