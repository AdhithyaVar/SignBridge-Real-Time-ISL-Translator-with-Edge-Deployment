"""
metrics.py
----------
All evaluation metrics for the SignBridge ISL classifier.

Provides
--------
compute_accuracy            — top-1 accuracy
compute_topk_accuracy       — top-k accuracy (k=3)
compute_per_class_metrics   — precision, recall, F1, support per class
compute_confusion_matrix    — (N_classes, N_classes) count matrix
classification_summary      — single dict with all metrics combined

All functions accept plain numpy arrays (no PyTorch tensors) so they
are framework-agnostic and can be used in any context.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from src.utils.class_labels import CLASS_LABELS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Top-1 accuracy.

    Parameters
    ----------
    y_true : (N,) int array of ground-truth class indices
    y_pred : (N,) int array of predicted class indices

    Returns
    -------
    float in [0, 1]
    """
    return float(accuracy_score(y_true, y_pred))


def compute_topk_accuracy(
    y_true:   np.ndarray,
    y_scores: np.ndarray,
    k:        int = 3,
) -> float:
    """
    Top-k accuracy: fraction of samples where the true label
    appears in the top-k predicted classes.

    Parameters
    ----------
    y_true   : (N,)     int array of ground-truth class indices
    y_scores : (N, C)   float array of class logits or probabilities
    k        : int      number of top predictions to consider

    Returns
    -------
    float in [0, 1]
    """
    N   = len(y_true)
    k   = min(k, y_scores.shape[1])
    # Indices of top-k classes per sample (descending score)
    topk = np.argsort(y_scores, axis=1)[:, -k:]   # (N, k)

    correct = sum(
        int(y_true[i]) in topk[i].tolist()
        for i in range(N)
    )
    return correct / max(N, 1)


def compute_per_class_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    class_names:  list[str] | None = None,
) -> list[dict]:
    """
    Compute precision, recall, F1, and support for every class.

    Parameters
    ----------
    y_true      : (N,) int array
    y_pred      : (N,) int array
    class_names : list of class name strings (default: CLASS_LABELS)

    Returns
    -------
    list of dicts, one per class, each with keys:
        class_name, class_idx, precision, recall, f1, support,
        correct, total
    """
    class_names = class_names or CLASS_LABELS
    labels      = list(range(len(class_names)))

    precision = precision_score(y_true, y_pred, labels=labels,
                                average=None, zero_division=0)
    recall    = recall_score(   y_true, y_pred, labels=labels,
                                average=None, zero_division=0)
    f1        = f1_score(       y_true, y_pred, labels=labels,
                                average=None, zero_division=0)

    # Per-class support and correct counts
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    results = []
    for idx, name in enumerate(class_names):
        if idx >= len(precision):
            continue
        support = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        results.append({
            "class_name": name,
            "class_idx":  idx,
            "precision":  float(precision[idx]),
            "recall":     float(recall[idx]),
            "f1":         float(f1[idx]),
            "support":    support,
            "correct":    correct,
        })
    return results


def compute_confusion_matrix(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list[str] | None = None,
) -> np.ndarray:
    """
    Compute the (C, C) confusion matrix.

    Parameters
    ----------
    y_true      : (N,) int
    y_pred      : (N,) int
    class_names : list of C class name strings

    Returns
    -------
    np.ndarray, shape (C, C), dtype int
    Row = true class, Column = predicted class.
    """
    class_names = class_names or CLASS_LABELS
    labels      = list(range(len(class_names)))
    return confusion_matrix(y_true, y_pred, labels=labels)


def classification_summary(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    y_scores:     np.ndarray | None = None,
    class_names:  list[str] | None  = None,
) -> dict:
    """
    Compute all evaluation metrics and return as a single dict.

    Parameters
    ----------
    y_true      : (N,) int ground-truth labels
    y_pred      : (N,) int predicted labels
    y_scores    : (N, C) float logits / probabilities (optional, for top-k)
    class_names : list of class names (default: CLASS_LABELS)

    Returns
    -------
    dict with keys:
        accuracy, top3_accuracy,
        macro_f1, macro_precision, macro_recall,
        weighted_f1,
        per_class  : list[dict]
        cm         : np.ndarray (C, C)
    """
    class_names = class_names or CLASS_LABELS
    labels      = list(range(len(class_names)))

    top1 = compute_accuracy(y_true, y_pred)
    top3 = (compute_topk_accuracy(y_true, y_scores, k=3)
            if y_scores is not None else None)

    macro_f1  = float(f1_score(y_true, y_pred, labels=labels,
                               average="macro",    zero_division=0))
    macro_p   = float(precision_score(y_true, y_pred, labels=labels,
                                      average="macro",    zero_division=0))
    macro_r   = float(recall_score(y_true, y_pred, labels=labels,
                                   average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels,
                                 average="weighted", zero_division=0))

    per_class = compute_per_class_metrics(y_true, y_pred, class_names)
    cm        = compute_confusion_matrix( y_true, y_pred, class_names)

    summary = {
        "accuracy":          top1,
        "top3_accuracy":     top3,
        "macro_f1":          macro_f1,
        "macro_precision":   macro_p,
        "macro_recall":      macro_r,
        "weighted_f1":       weighted_f1,
        "per_class":         per_class,
        "cm":                cm,
        "num_samples":       int(len(y_true)),
        "num_classes":       len(class_names),
    }
    return summary


def find_confused_pairs(
    cm:          np.ndarray,
    class_names: list[str],
    top_n:       int = 10,
) -> list[dict]:
    """
    Find the most commonly confused class pairs (off-diagonal CM entries).

    Parameters
    ----------
    cm          : (C, C) confusion matrix
    class_names : list of C class names
    top_n       : number of top confused pairs to return

    Returns
    -------
    list of dicts with keys: true_class, pred_class, count
    Sorted by count descending.
    """
    C = len(class_names)
    pairs = []
    for i in range(C):
        for j in range(C):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count":      int(cm[i, j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_n]
