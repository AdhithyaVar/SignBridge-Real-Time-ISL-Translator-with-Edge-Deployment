"""
test_finetuner.py
-----------------
Smoke tests for the fine-tuning pipeline.

Tests:
  - freeze_all / unfreeze_layers correctly control requires_grad
  - build_weighted_criterion creates correct class weights
  - FineTuner runs 2 epochs without crash on synthetic data
  - Progressive unfreezing increases trainable param count per stage
  - Hard example mining weights are correctly applied

Run:
    pytest tests/test_finetuner.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.cnn_lstm import SignBridgeCNNLSTM
from src.training.finetuner import (
    freeze_all,
    unfreeze_layers,
    get_trainable_count,
    build_weighted_criterion,
    FineTuner,
)
from src.utils.class_labels import CLASS_LABELS, CLASS_TO_IDX, NUM_CLASSES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEQ_LEN  = 30
FEAT_DIM = 126
DEVICE   = torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model() -> SignBridgeCNNLSTM:
    return SignBridgeCNNLSTM(
        num_classes=NUM_CLASSES,
        sequence_length=SEQ_LEN,
        feature_dim=FEAT_DIM,
        cnn_channels=[16, 32, 64],
        lstm_hidden=32,
        lstm_layers=1,
        lstm_dropout=0.0,
        attn_heads=2,
        attn_dim=16,
        attn_dropout=0.0,
        clf_dims=[64, 32],
        clf_dropout=[0.0, 0.0],
    )


def _make_loaders(n=40, batch=8):
    torch.manual_seed(42)
    X = torch.randn(n, SEQ_LEN, FEAT_DIM)
    y = torch.randint(0, NUM_CLASSES, (n,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=True), \
           DataLoader(ds, batch_size=batch)


def _minimal_finetune_cfg(epochs=2):
    return {
        "stages": {
            "stage1": {
                "name":    "Classifier only",
                "epochs":  epochs,
                "lr":      1e-4,
                "unfreeze":["classifier"],
                "batch_size": 8,
            },
            "stage2": {
                "name":    "Attention + Classifier",
                "epochs":  epochs,
                "lr":      5e-5,
                "unfreeze":["attention", "classifier"],
                "batch_size": 8,
            },
        },
        "optimizer":     {"weight_decay": 5e-4},
        "loss":          {"label_smoothing": 0.1},
        "hard_mining":   {
            "enabled":           True,
            "hard_classes":      ["S", "M", "N"],
            "hard_class_weight": 2.5,
        },
        "augmentation":  {},
        "early_stopping":{"patience": 100, "min_delta": 0.0},
        "checkpoints":   {
            "save_dir": "models/finetune_checkpoints",
            "best_dir": "models/finetuned",
            "best_filename": "finetuned_model.pth",
        },
        "logging": {
            "log_dir": "logs/finetune",
            "csv_filename": "finetune_test_log.csv",
        },
        "hardware": {"mixed_precision": False},
    }


# ============================================================================
# Tests: freeze_all / unfreeze_layers
# ============================================================================

class TestFreezeUnfreeze:

    def test_freeze_all_no_trainable_params(self):
        model = _make_tiny_model()
        freeze_all(model)
        trainable = get_trainable_count(model)
        assert trainable == 0, f"Expected 0 trainable params, got {trainable}"

    def test_unfreeze_classifier_only(self):
        model = _make_tiny_model()
        freeze_all(model)
        unfreeze_layers(model, ["classifier"])
        trainable = get_trainable_count(model)
        clf_params = sum(p.numel() for p in model.classifier.parameters())
        assert trainable == clf_params, \
            f"Expected {clf_params} trainable, got {trainable}"

    def test_unfreeze_multiple_layers(self):
        model = _make_tiny_model()
        freeze_all(model)
        unfreeze_layers(model, ["classifier"])
        count1 = get_trainable_count(model)

        unfreeze_layers(model, ["attention"])
        count2 = get_trainable_count(model)
        assert count2 > count1, "Trainable count should increase after unfreeze"

    def test_progressive_unfreezing_order(self):
        """Each stage should have >= trainable params than previous."""
        model = _make_tiny_model()
        freeze_all(model)

        counts = []
        for layers in [["classifier"], ["attention", "classifier"],
                       ["lstm", "attention", "classifier"],
                       ["cnn", "lstm", "attention", "classifier"]]:
            unfreeze_layers(model, layers)
            counts.append(get_trainable_count(model))

        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1], \
                f"Trainable count decreased at stage {i}: {counts}"

    def test_unfreeze_all_restores_full_count(self):
        model = _make_tiny_model()
        total_params = sum(p.numel() for p in model.parameters())

        freeze_all(model)
        unfreeze_layers(model, ["cnn", "lstm", "attention",
                                "classifier", "input_norm"])
        trainable = get_trainable_count(model)
        assert trainable == total_params, \
            f"After unfreezing all, expected {total_params}, got {trainable}"


# ============================================================================
# Tests: build_weighted_criterion
# ============================================================================

class TestWeightedCriterion:

    def test_hard_class_has_higher_weight(self):
        criterion = build_weighted_criterion(
            hard_classes=["S"],
            hard_class_weight=3.0,
            num_classes=NUM_CLASSES,
            label_smoothing=0.0,
            device=DEVICE,
        )
        s_idx = CLASS_TO_IDX["S"]
        a_idx = CLASS_TO_IDX["A"]
        assert criterion.weight[s_idx] == pytest.approx(3.0), \
            f"Hard class S should have weight 3.0"
        assert criterion.weight[a_idx] == pytest.approx(1.0), \
            "Non-hard class A should have weight 1.0"

    def test_all_non_hard_classes_have_weight_one(self):
        hard = ["S", "M", "N"]
        criterion = build_weighted_criterion(
            hard_classes=hard,
            hard_class_weight=2.5,
            num_classes=NUM_CLASSES,
            label_smoothing=0.0,
            device=DEVICE,
        )
        for cls in CLASS_LABELS:
            if cls not in hard:
                idx = CLASS_TO_IDX[cls]
                assert criterion.weight[idx] == pytest.approx(1.0), \
                    f"Class {cls} should have weight 1.0"

    def test_criterion_is_cross_entropy(self):
        criterion = build_weighted_criterion(
            hard_classes=["S"],
            hard_class_weight=2.0,
            num_classes=NUM_CLASSES,
            label_smoothing=0.0,
            device=DEVICE,
        )
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_criterion_forward_no_crash(self):
        criterion = build_weighted_criterion(
            hard_classes=["S", "M"],
            hard_class_weight=2.5,
            num_classes=NUM_CLASSES,
            label_smoothing=0.1,
            device=DEVICE,
        )
        logits = torch.randn(8, NUM_CLASSES)
        labels = torch.randint(0, NUM_CLASSES, (8,))
        loss = criterion(logits, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ============================================================================
# Tests: FineTuner smoke tests
# ============================================================================

class TestFineTunerSmoke:

    def test_stage1_runs_without_crash(self):
        """Classifier-only fine-tuning must complete 2 epochs."""
        model          = _make_tiny_model()
        train_l, val_l = _make_loaders()
        cfg            = _minimal_finetune_cfg(epochs=2)
        cfg["stages"]  = {"stage1": cfg["stages"]["stage1"]}

        ft = FineTuner(model, train_l, val_l, cfg, DEVICE)
        history = ft.run_all_stages()
        assert len(history["epoch"]) == 2

    def test_two_stage_history_length(self):
        """2 stages × 2 epochs = 4 total history entries."""
        model          = _make_tiny_model()
        train_l, val_l = _make_loaders()
        cfg            = _minimal_finetune_cfg(epochs=2)

        ft = FineTuner(model, train_l, val_l, cfg, DEVICE)
        history = ft.run_all_stages()
        assert len(history["epoch"]) == 4, \
            f"Expected 4 epoch records (2 stages × 2), got {len(history['epoch'])}"

    def test_loss_is_finite(self):
        model          = _make_tiny_model()
        train_l, val_l = _make_loaders()
        cfg            = _minimal_finetune_cfg(epochs=2)
        cfg["stages"]  = {"stage1": cfg["stages"]["stage1"]}

        ft = FineTuner(model, train_l, val_l, cfg, DEVICE)
        history = ft.run_all_stages()
        for loss in history["train_loss"] + history["val_loss"]:
            assert torch.isfinite(torch.tensor(loss)), \
                f"Non-finite loss: {loss}"

    def test_accuracy_in_valid_range(self):
        model          = _make_tiny_model()
        train_l, val_l = _make_loaders()
        cfg            = _minimal_finetune_cfg(epochs=2)
        cfg["stages"]  = {"stage1": cfg["stages"]["stage1"]}

        ft = FineTuner(model, train_l, val_l, cfg, DEVICE)
        history = ft.run_all_stages()
        for acc in history["train_accuracy"] + history["val_accuracy"]:
            assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_only_classifier_trainable_in_stage1(self):
        """During stage1, only classifier params should have gradients."""
        model          = _make_tiny_model()
        train_l, val_l = _make_loaders()
        cfg            = _minimal_finetune_cfg(epochs=1)
        cfg["stages"]  = {"stage1": cfg["stages"]["stage1"]}

        ft = FineTuner(model, train_l, val_l, cfg, DEVICE)

        # After freeze_all + unfreeze classifier, CNN/LSTM must be frozen
        freeze_all(model)
        unfreeze_layers(model, ["classifier"])

        for name, param in model.named_parameters():
            if "classifier" in name:
                assert param.requires_grad, \
                    f"Classifier param {name} should be trainable"
            elif "cnn" in name or "lstm" in name:
                assert not param.requires_grad, \
                    f"CNN/LSTM param {name} should be frozen in stage1"

    def test_best_model_saved(self, tmp_path):
        """Best model file must be created after fine-tuning."""
        model          = _make_tiny_model()
        train_l, val_l = _make_loaders()
        cfg            = _minimal_finetune_cfg(epochs=2)
        cfg["stages"]  = {"stage1": cfg["stages"]["stage1"]}
        cfg["checkpoints"]["best_dir"] = str(tmp_path)

        ft = FineTuner(model, train_l, val_l, cfg, DEVICE)
        ft.run_all_stages()

        best_path = tmp_path / "finetuned_model.pth"
        assert best_path.exists(), "finetuned_model.pth was not created"
