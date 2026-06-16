"""
evaluator.py
------------
Full test-set evaluation using the best trained model.

Loads the best checkpoint from models/best/best_model.pth,
runs inference over the entire test set, and returns a rich
EvaluationResult containing every metric needed for reporting.

Also provides a quick validation-set evaluator used during
training for live accuracy tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import classification_summary, find_confused_pairs
from src.utils.class_labels import CLASS_LABELS, NUM_CLASSES
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Complete evaluation output for one dataset split.

    Attributes
    ----------
    split          : 'test' | 'val'
    accuracy       : top-1 accuracy  [0, 1]
    top3_accuracy  : top-3 accuracy  [0, 1]
    macro_f1       : macro-averaged F1 score
    macro_precision: macro-averaged precision
    macro_recall   : macro-averaged recall
    weighted_f1    : weighted-averaged F1 score
    per_class      : list of per-class metric dicts
    cm             : (C, C) confusion matrix (counts)
    confused_pairs : top-10 most confused class pairs
    y_true         : (N,) ground-truth labels
    y_pred         : (N,) predicted labels
    y_scores       : (N, C) softmax probabilities
    num_samples    : total samples evaluated
    """
    split:           str
    accuracy:        float
    top3_accuracy:   float
    macro_f1:        float
    macro_precision: float
    macro_recall:    float
    weighted_f1:     float
    per_class:       list[dict]
    cm:              np.ndarray
    confused_pairs:  list[dict]
    y_true:          np.ndarray
    y_pred:          np.ndarray
    y_scores:        np.ndarray
    num_samples:     int
    class_names:     list[str] = field(default_factory=lambda: CLASS_LABELS)

    @property
    def target_met(self) -> bool:
        """True if val accuracy >= 90%."""
        return self.accuracy >= 0.90

    def summary_str(self) -> str:
        """One-line summary for logging."""
        return (
            f"[{self.split.upper()}] "
            f"acc={self.accuracy:.4f} "
            f"top3={self.top3_accuracy:.4f} "
            f"macro_f1={self.macro_f1:.4f} "
            f"n={self.num_samples}"
        )

    def worst_classes(self, n: int = 5) -> list[dict]:
        """Return the N classes with lowest F1 score."""
        return sorted(self.per_class, key=lambda x: x["f1"])[:n]

    def best_classes(self, n: int = 5) -> list[dict]:
        """Return the N classes with highest F1 score."""
        return sorted(self.per_class, key=lambda x: x["f1"], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Runs inference over a DataLoader and computes all metrics.

    Parameters
    ----------
    model       : trained SignBridgeCNNLSTM (or any nn.Module)
    device      : torch.device
    class_names : list of class name strings
    """

    def __init__(
        self,
        model:       nn.Module,
        device:      torch.device,
        class_names: list[str] | None = None,
    ) -> None:
        self.model       = model.to(device)
        self.device      = device
        self.class_names = class_names or CLASS_LABELS

    def evaluate(
        self,
        loader: DataLoader,
        split:  str = "test",
    ) -> EvaluationResult:
        """
        Run full evaluation over a DataLoader.

        Parameters
        ----------
        loader : DataLoader yielding (X, y) batches
        split  : label for this evaluation ('test' | 'val')

        Returns
        -------
        EvaluationResult
        """
        self.model.eval()

        all_true:   list[np.ndarray] = []
        all_pred:   list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        pbar = tqdm(
            loader,
            desc=f"Evaluating [{split}]",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )

        with torch.no_grad():
            for X, y in pbar:
                X = X.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.model(X)                          # (batch, C)
                probs  = torch.softmax(logits, dim=-1)          # (batch, C)
                preds  = logits.argmax(dim=-1)                  # (batch,)

                all_true.append(  y.cpu().numpy())
                all_pred.append(  preds.cpu().numpy())
                all_scores.append(probs.cpu().numpy())

        y_true   = np.concatenate(all_true,   axis=0)   # (N,)
        y_pred   = np.concatenate(all_pred,   axis=0)   # (N,)
        y_scores = np.concatenate(all_scores, axis=0)   # (N, C)

        # Compute all metrics
        summary = classification_summary(
            y_true, y_pred, y_scores,
            class_names=self.class_names,
        )
        confused = find_confused_pairs(
            summary["cm"],
            self.class_names,
            top_n=10,
        )

        result = EvaluationResult(
            split=           split,
            accuracy=        summary["accuracy"],
            top3_accuracy=   summary["top3_accuracy"] or 0.0,
            macro_f1=        summary["macro_f1"],
            macro_precision= summary["macro_precision"],
            macro_recall=    summary["macro_recall"],
            weighted_f1=     summary["weighted_f1"],
            per_class=       summary["per_class"],
            cm=              summary["cm"],
            confused_pairs=  confused,
            y_true=          y_true,
            y_pred=          y_pred,
            y_scores=        y_scores,
            num_samples=     summary["num_samples"],
            class_names=     self.class_names,
        )

        logger.info(result.summary_str())
        return result


# ---------------------------------------------------------------------------
# Convenience loader: build evaluator from configs + best checkpoint
# ---------------------------------------------------------------------------

def build_evaluator_from_best(
    model_cfg:   dict,
    dataset_cfg: dict,
    device:      torch.device,
) -> ModelEvaluator:
    """
    Build a ModelEvaluator loaded with the best checkpoint weights.

    Parameters
    ----------
    model_cfg, dataset_cfg : parsed YAML dicts
    device                 : target device

    Returns
    -------
    ModelEvaluator with best model loaded and set to eval mode.
    """
    from src.models.model_factory import load_best_model
    from src.utils.config_loader import get_project_root

    best_dir  = get_project_root() / "models" / "best"
    model, metrics = load_best_model(
        model_cfg=   model_cfg,
        dataset_cfg= dataset_cfg,
        best_dir=    best_dir,
        device=      device,
    )
    logger.info(
        f"Loaded best model | "
        f"val_accuracy={metrics.get('val_accuracy', 'N/A')}"
    )
    return ModelEvaluator(model=model, device=device)


def run_full_evaluation(
    model_cfg:   dict,
    dataset_cfg: dict,
    training_cfg:dict,
    device:      torch.device,
) -> tuple[EvaluationResult, EvaluationResult]:
    """
    Run evaluation on both test and val splits using the best model.

    Parameters
    ----------
    model_cfg, dataset_cfg, training_cfg : parsed YAML dicts
    device                               : target device

    Returns
    -------
    (test_result, val_result)
    """
    from src.preprocessing.dataset_builder import build_dataloaders

    logger.info("Loading dataset splits for evaluation...")
    _, val_loader, test_loader, _ = build_dataloaders(
        dataset_cfg=  dataset_cfg,
        training_cfg= training_cfg,
        force_rebuild=False,
    )

    evaluator = build_evaluator_from_best(model_cfg, dataset_cfg, device)

    logger.info("Running test-set evaluation...")
    test_result = evaluator.evaluate(test_loader, split="test")

    logger.info("Running val-set evaluation...")
    val_result  = evaluator.evaluate(val_loader,  split="val")

    return test_result, val_result
