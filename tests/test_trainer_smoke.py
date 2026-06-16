"""
test_trainer_smoke.py
---------------------
Smoke tests for the training pipeline.

Uses a tiny synthetic dataset (4 classes, 20 samples each) to verify:
  - Training loop runs without crash
  - Loss decreases over 3 epochs (model is learning)
  - Accuracy is computed correctly
  - Callbacks execute without error
  - Scheduler steps correctly
  - Early stopping fires when patience is exceeded

These tests run entirely on CPU in under 60 seconds.

Run:
    pytest tests/test_trainer_smoke.py -v
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
from src.training.scheduler import (
    WarmupCosineScheduler,
    EarlyStopping,
)
from src.training.callbacks import (
    CheckpointCallback,
    CSVLoggerCallback,
    OverfittingMonitorCallback,
)
from src.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Tiny synthetic dataset helpers
# ---------------------------------------------------------------------------

_TINY_CLASSES   = 4
_TINY_SEQ_LEN   = 30
_TINY_FEAT_DIM  = 126
_TINY_SAMPLES   = 20    # per class
_BATCH          = 8
_DEVICE         = torch.device("cpu")


def _make_tiny_loaders(
    n_classes: int = _TINY_CLASSES,
    n_samples: int = _TINY_SAMPLES,
) -> tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders with random data for smoke testing.
    Each class has distinguishable signal so the model can learn.
    """
    torch.manual_seed(0)
    N = n_classes * n_samples

    # Give each class a distinct mean to make the task solvable
    seqs   = []
    labels = []
    for cls in range(n_classes):
        signal = torch.randn(n_samples, _TINY_SEQ_LEN, _TINY_FEAT_DIM)
        signal[:, :, cls * 5:(cls + 1) * 5] += 3.0   # distinctive feature
        seqs.append(signal)
        labels.extend([cls] * n_samples)

    X = torch.cat(seqs, dim=0).float()
    y = torch.tensor(labels, dtype=torch.long)

    # Shuffle
    perm = torch.randperm(N)
    X, y = X[perm], y[perm]

    # Split 80/20
    split = int(0.8 * N)
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds   = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=_BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=_BATCH, shuffle=False)
    return train_loader, val_loader


def _make_tiny_model(n_classes: int = _TINY_CLASSES) -> SignBridgeCNNLSTM:
    """Create a very small model for fast smoke testing."""
    return SignBridgeCNNLSTM(
        num_classes=     n_classes,
        sequence_length= _TINY_SEQ_LEN,
        feature_dim=     _TINY_FEAT_DIM,
        cnn_channels=    [16, 32, 64],   # tiny channels
        lstm_hidden=     32,              # tiny hidden
        lstm_layers=     1,
        lstm_dropout=    0.0,
        attn_heads=      2,
        attn_dim=        16,
        attn_dropout=    0.0,
        clf_dims=        [64, 32],
        clf_dropout=     [0.0, 0.0],
    )


def _make_trainer(
    model:        nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    max_epochs:   int = 3,
    callbacks     = None,
    early_stop_patience: int = 100,
) -> Trainer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(
        optimizer=       optimizer,
        warmup_epochs=   1,
        warmup_start_lr= 1e-5,
        base_lr=         1e-3,
        T_0=             5,
        T_mult=          1,
        eta_min=         1e-6,
    )
    criterion      = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=     early_stop_patience,
        min_delta=    0.0,
        mode=         "max",
        restore_best= False,
    )
    return Trainer(
        model=          model,
        train_loader=   train_loader,
        val_loader=     val_loader,
        optimizer=      optimizer,
        scheduler=      scheduler,
        criterion=      criterion,
        early_stopping= early_stopping,
        callbacks=      callbacks or [],
        device=         _DEVICE,
        max_epochs=     max_epochs,
        grad_clip=      1.0,
        use_amp=        False,    # CPU smoke test
        log_every_n=    5,
    )


# ---------------------------------------------------------------------------
# Tests: WarmupCosineScheduler
# ---------------------------------------------------------------------------

class TestWarmupCosineScheduler:

    def test_lr_increases_during_warmup(self):
        model     = _make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched     = WarmupCosineScheduler(
            optimizer=       optimizer,
            warmup_epochs=   5,
            warmup_start_lr= 1e-5,
            base_lr=         1e-3,
        )
        lrs = []
        for _ in range(5):
            sched.step()
            lrs.append(sched.get_lr())

        # LR should increase monotonically during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], \
                f"LR decreased during warmup at step {i}: {lrs}"

    def test_lr_at_end_of_warmup_is_base_lr(self):
        model     = _make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched     = WarmupCosineScheduler(
            optimizer=       optimizer,
            warmup_epochs=   5,
            warmup_start_lr= 1e-5,
            base_lr=         1e-3,
        )
        for _ in range(5):
            lr = sched.step()
        assert lr == pytest.approx(1e-3, rel=0.01)

    def test_lr_decreases_after_warmup(self):
        model     = _make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched     = WarmupCosineScheduler(
            optimizer=       optimizer,
            warmup_epochs=   2,
            warmup_start_lr= 1e-5,
            base_lr=         1e-3,
            T_0=             5,
            eta_min=         1e-6,
        )
        # Warmup phase
        for _ in range(2):
            sched.step()
        peak_lr = sched.get_lr()

        # Post-warmup cosine phase
        for _ in range(5):
            sched.step()
        final_lr = sched.get_lr()

        assert final_lr < peak_lr, \
            f"Expected LR to decrease after warmup: peak={peak_lr}, final={final_lr}"

    def test_state_dict_roundtrip(self):
        model     = _make_tiny_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched     = WarmupCosineScheduler(optimizer, warmup_epochs=3)
        for _ in range(3):
            sched.step()
        state = sched.state_dict()
        assert state["_epoch"] == 3


# ---------------------------------------------------------------------------
# Tests: EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:

    def test_does_not_stop_on_improvement(self):
        model = _make_tiny_model()
        es    = EarlyStopping(patience=3, min_delta=0.001, mode="max",
                              restore_best=False)
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, s in enumerate(scores):
            stopped = es(s, model, epoch=i + 1)
            assert not stopped, f"Stopped early at step {i+1} despite improvement"

    def test_stops_after_patience_exceeded(self):
        model = _make_tiny_model()
        es    = EarlyStopping(patience=3, min_delta=0.001, mode="max",
                              restore_best=False)
        # One good epoch then plateau
        es(0.8, model, epoch=1)
        stopped = False
        for i in range(4):
            stopped = es(0.8, model, epoch=i + 2)
        assert stopped, "EarlyStopping should have triggered after patience=3"

    def test_best_score_tracked(self):
        model = _make_tiny_model()
        es    = EarlyStopping(patience=10, mode="max", restore_best=False)
        for s in [0.5, 0.7, 0.6, 0.8, 0.75]:
            es(s, model, epoch=1)
        assert es.best_score == pytest.approx(0.8)

    def test_restore_best_weights(self):
        """Weights from the best epoch should be restored on stop."""
        model = _make_tiny_model()
        es    = EarlyStopping(patience=2, mode="max", restore_best=True)

        # Record weights at best epoch (score=0.9)
        es(0.9, model, epoch=1)
        best_param = next(model.parameters()).data.clone()

        # Modify weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 100)

        # Plateau → early stop
        es(0.85, model, epoch=2)
        es(0.84, model, epoch=3)
        stopped = es(0.83, model, epoch=4)

        assert stopped
        current_param = next(model.parameters()).data
        assert torch.allclose(current_param, best_param, atol=1e-5), \
            "Best weights were not correctly restored."


# ---------------------------------------------------------------------------
# Tests: Trainer smoke tests
# ---------------------------------------------------------------------------

class TestTrainerSmoke:

    def test_train_runs_without_crash(self):
        """Full 3-epoch training loop must not raise any exception."""
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=3)
        history      = trainer.train()
        assert history is not None

    def test_history_has_correct_keys(self):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=2)
        history      = trainer.train()

        expected_keys = {
            "epoch", "train_loss", "val_loss",
            "train_accuracy", "val_accuracy",
            "top3_accuracy", "lr", "epoch_time_s",
        }
        assert expected_keys.issubset(set(history.keys()))

    def test_history_length_matches_epochs(self):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=3)
        history      = trainer.train()
        assert len(history["epoch"]) == 3

    def test_loss_is_finite(self):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=2)
        history      = trainer.train()
        for loss in history["train_loss"] + history["val_loss"]:
            assert torch.isfinite(torch.tensor(loss)), \
                f"Non-finite loss encountered: {loss}"

    def test_accuracy_in_valid_range(self):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=2)
        history      = trainer.train()
        for acc in history["train_accuracy"] + history["val_accuracy"]:
            assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"

    def test_loss_decreases_over_epochs(self):
        """With a learnable synthetic dataset, training loss should decrease."""
        torch.manual_seed(42)
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders(n_samples=40)
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=5)
        history      = trainer.train()
        first_loss = history["train_loss"][0]
        last_loss  = history["train_loss"][-1]
        assert last_loss < first_loss, \
            f"Training loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"

    def test_lr_changes_over_epochs(self):
        """Learning rate should not stay constant (scheduler must be active)."""
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(model, train_loader, val_loader, max_epochs=4)
        history      = trainer.train()
        lrs = history["lr"]
        assert len(set(lrs)) > 1, \
            f"Learning rate never changed: {lrs}"

    def test_early_stopping_fires(self):
        """With patience=1, early stopping should fire after 2 epochs."""
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        trainer      = _make_trainer(
            model, train_loader, val_loader,
            max_epochs=50,
            early_stop_patience=1,
        )
        history = trainer.train()
        # Should stop well before 50 epochs
        assert len(history["epoch"]) < 50, \
            "Early stopping did not fire within 50 epochs with patience=1."


# ---------------------------------------------------------------------------
# Tests: Callback smoke tests
# ---------------------------------------------------------------------------

class TestCallbackSmoke:

    def test_checkpoint_callback_saves_file(self, tmp_path):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        optimizer    = torch.optim.AdamW(model.parameters(), lr=1e-3)
        cb           = CheckpointCallback(
            checkpoint_dir= tmp_path / "checkpoints",
            best_dir=       tmp_path / "best",
            save_every=     1,
            keep_last_n=    3,
            optimizer=      optimizer,
        )
        trainer = _make_trainer(
            model, train_loader, val_loader,
            max_epochs=3, callbacks=[cb],
        )
        trainer.train()
        best_path = tmp_path / "best" / "best_model.pth"
        assert best_path.exists(), "best_model.pth was not saved."

    def test_csv_logger_creates_file(self, tmp_path):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        cb           = CSVLoggerCallback(log_dir=tmp_path, filename="log.csv")
        trainer      = _make_trainer(
            model, train_loader, val_loader,
            max_epochs=2, callbacks=[cb],
        )
        trainer.train()
        log_path = tmp_path / "log.csv"
        assert log_path.exists(), "training_log.csv was not created."
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3, \
            f"Expected header + 2 data rows, got {len(lines)} lines."

    def test_overfitting_monitor_no_crash(self):
        model        = _make_tiny_model()
        train_loader, val_loader = _make_tiny_loaders()
        cb           = OverfittingMonitorCallback(
            overfit_threshold=0.01,     # low threshold to trigger warning
            underfit_threshold=0.99,    # high threshold to trigger warning
        )
        trainer = _make_trainer(
            model, train_loader, val_loader,
            max_epochs=2, callbacks=[cb],
        )
        # Should not crash even when warnings are triggered
        trainer.train()
