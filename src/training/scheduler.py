"""
scheduler.py
------------
Learning rate scheduling and early stopping for SignBridge training.

Components
----------
WarmupCosineScheduler
    Linear warmup for the first N epochs, then CosineAnnealingWarmRestarts.
    Warmup prevents large gradient updates before the model has settled.
    Cosine annealing with warm restarts escapes local minima periodically.

EarlyStopping
    Monitors a metric (val_accuracy) and stops training when no improvement
    is seen for `patience` consecutive epochs.  Optionally restores the
    best weights when stopping.

Why cosine annealing with warm restarts?
-----------------------------------------
* Standard step decay drops LR abruptly — unstable for LSTM.
* Cosine gives smooth gradual decrease — better convergence.
* Warm restarts (T_0=30, T_mult=2) allow periodic exploration:
    Restart 1 at epoch 30  (cycle length 30)
    Restart 2 at epoch 90  (cycle length 60)
    Restart 3 at epoch 210 (cycle length 120)
  This often finds better minima than a single long decay.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Warmup + Cosine Scheduler
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """
    Linear warmup followed by CosineAnnealingWarmRestarts.

    Parameters
    ----------
    optimizer       : torch optimizer
    warmup_epochs   : int   — epochs of linear warmup (5)
    warmup_start_lr : float — LR at start of warmup (1e-5)
    base_lr         : float — target LR after warmup = optimizer initial LR
    T_0             : int   — epochs for first cosine cycle (30)
    T_mult          : int   — cycle length multiplier (2)
    eta_min         : float — minimum LR (1e-6)
    """

    def __init__(
        self,
        optimizer:       optim.Optimizer,
        warmup_epochs:   int   = 5,
        warmup_start_lr: float = 1e-5,
        base_lr:         float = 1e-3,
        T_0:             int   = 30,
        T_mult:          int   = 2,
        eta_min:         float = 1e-6,
    ) -> None:
        self.optimizer       = optimizer
        self.warmup_epochs   = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr         = base_lr
        self.eta_min         = eta_min
        self._epoch          = 0

        # Initialise cosine scheduler (used after warmup ends)
        # Set initial LR to base_lr so it starts cosine from correct point
        self._cosine = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
        )

        # Set starting LR to warmup_start_lr
        self._set_lr(warmup_start_lr)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> float:
        """
        Advance the scheduler by one epoch.
        Call once per epoch AFTER the optimizer step.

        Returns
        -------
        float — current learning rate after this step.
        """
        epoch = self._epoch

        if epoch < self.warmup_epochs:
            # Linear warmup: LR increases from warmup_start_lr → base_lr
            progress = (epoch + 1) / self.warmup_epochs
            lr = self.warmup_start_lr + progress * (self.base_lr - self.warmup_start_lr)
            self._set_lr(lr)
        else:
            # Cosine annealing (epoch offset by warmup length)
            cosine_epoch = epoch - self.warmup_epochs
            self._cosine.step(cosine_epoch)

        self._epoch += 1
        return self.get_lr()

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {
            "_epoch":          self._epoch,
            "warmup_epochs":   self.warmup_epochs,
            "warmup_start_lr": self.warmup_start_lr,
            "base_lr":         self.base_lr,
            "cosine":          self._cosine.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._epoch          = state["_epoch"]
        self.warmup_epochs   = state["warmup_epochs"]
        self.warmup_start_lr = state["warmup_start_lr"]
        self.base_lr         = state["base_lr"]
        self._cosine.load_state_dict(state["cosine"])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a validation metric and signals when training should stop.

    Parameters
    ----------
    patience      : int   — epochs with no improvement before stopping (20)
    min_delta     : float — minimum improvement to reset patience counter (0.001)
    mode          : str   — 'max' for accuracy, 'min' for loss
    restore_best  : bool  — restore best weights when stopping
    verbose       : bool  — log improvement/stagnation messages
    """

    def __init__(
        self,
        patience:     int   = 20,
        min_delta:    float = 0.001,
        mode:         str   = "max",
        restore_best: bool  = True,
        verbose:      bool  = True,
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'.")

        self.patience      = patience
        self.min_delta     = min_delta
        self.mode          = mode
        self.restore_best  = restore_best
        self.verbose       = verbose

        self._counter      = 0
        self._best_score: Optional[float]  = None
        self._best_weights: Optional[dict] = None
        self._stopped      = False
        self._best_epoch   = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def __call__(
        self,
        score: float,
        model: torch.nn.Module,
        epoch: int,
    ) -> bool:
        """
        Call after each validation epoch.

        Parameters
        ----------
        score : float — current metric value
        model : model (weights copied if improved)
        epoch : int   — current epoch (for logging)

        Returns
        -------
        bool — True if training should STOP.
        """
        if self._best_score is None:
            # First epoch — always record as best
            self._record_best(score, model, epoch)
            return False

        improved = self._is_improved(score)

        if improved:
            self._record_best(score, model, epoch)
            self._counter = 0
            if self.verbose:
                logger.info(
                    f"EarlyStopping: improvement "
                    f"{self._best_score:.4f} → {score:.4f} "
                    f"(epoch {epoch})"
                )
        else:
            self._counter += 1
            if self.verbose:
                logger.debug(
                    f"EarlyStopping: no improvement "
                    f"({self._counter}/{self.patience}) "
                    f"best={self._best_score:.4f}"
                )

        if self._counter >= self.patience:
            self._stopped = True
            logger.info(
                f"EarlyStopping triggered at epoch {epoch}. "
                f"Best score: {self._best_score:.4f} at epoch {self._best_epoch}."
            )
            if self.restore_best and self._best_weights is not None:
                model.load_state_dict(self._best_weights)
                logger.info("Best weights restored.")
            return True

        return False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def best_score(self) -> Optional[float]:
        return self._best_score

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    @property
    def counter(self) -> int:
        return self._counter

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _is_improved(self, score: float) -> bool:
        if self.mode == "max":
            return score > self._best_score + self.min_delta
        return score < self._best_score - self.min_delta

    def _record_best(
        self,
        score: float,
        model: torch.nn.Module,
        epoch: int,
    ) -> None:
        self._best_score = score
        self._best_epoch = epoch
        if self.restore_best:
            self._best_weights = deepcopy(model.state_dict())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer:    optim.Optimizer,
    training_cfg: dict,
) -> WarmupCosineScheduler:
    """Build WarmupCosineScheduler from training.yaml config."""
    sched_cfg   = training_cfg.get("scheduler", {})
    opt_cfg     = training_cfg.get("optimizer",  {})

    return WarmupCosineScheduler(
        optimizer=       optimizer,
        warmup_epochs=   int(  sched_cfg.get("warmup_epochs",      5)),
        warmup_start_lr= float(sched_cfg.get("warmup_start_lr",    1e-5)),
        base_lr=         float(opt_cfg.get("lr",                   1e-3)),
        T_0=             int(  sched_cfg.get("T_0",                30)),
        T_mult=          int(  sched_cfg.get("T_mult",              2)),
        eta_min=         float(sched_cfg.get("eta_min",            1e-6)),
    )


def build_early_stopping(training_cfg: dict) -> EarlyStopping:
    """Build EarlyStopping from training.yaml config."""
    es_cfg = training_cfg.get("early_stopping", {})
    return EarlyStopping(
        patience=     int(  es_cfg.get("patience",    20)),
        min_delta=    float(es_cfg.get("min_delta",    0.001)),
        mode=               es_cfg.get("mode",        "max"),
        restore_best= bool( es_cfg.get("restore_best", True)),
        verbose=True,
    )
