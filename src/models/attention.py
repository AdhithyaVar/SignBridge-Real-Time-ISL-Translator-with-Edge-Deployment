"""
attention.py
------------
Temporal attention modules for the SignBridge CNN-LSTM model.

Architecture design rationale
------------------------------
Sign language gestures have two critical temporal properties:

1.  Most discriminative information is concentrated in SPECIFIC frames
    (peak of gesture, finger configuration moments) rather than evenly
    distributed across all 30 frames.  A simple mean-pool over time
    discards this structure.

2.  Temporal dependencies between frames (trajectory) matter as much
    as individual frame features.  Self-attention captures long-range
    frame relationships that LSTM alone cannot model as explicitly.

Solution: two-stage temporal attention
  Stage 1 — MultiheadSelfAttention
      Each frame attends to all other frames.
      Captures: which frames are contextually related.
      Output shape: unchanged  (batch, T, D)

  Stage 2 — AdditiveAttentionPool
      Learns a scalar importance score per frame.
      Produces a single context vector via weighted sum.
      Output shape: (batch, D)  — reduces T dimension.
      Also returns attention weights for inference-time visualisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stage 1: Multi-head Self-Attention over temporal dimension
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Wrapper around nn.MultiheadAttention with residual connection
    and LayerNorm (Pre-LN style, more stable than Post-LN).

    Parameters
    ----------
    hidden_dim : int   — BiLSTM output dimension (512)
    num_heads  : int   — number of attention heads (4)
    dropout    : float — attention dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads:  int,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )

        self.norm       = nn.LayerNorm(hidden_dim)
        self.attn       = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,       # (batch, seq, feature) convention
        )
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, hidden_dim)

        Returns
        -------
        out          : torch.Tensor, shape (batch, T, hidden_dim)
        attn_weights : torch.Tensor, shape (batch, T, T)
            Row i shows how much frame i attended to every other frame.
        """
        # Pre-LayerNorm for training stability
        x_norm = self.norm(x)

        # Self-attention: Q = K = V = x_norm
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=True,   # average over heads
        )

        # Residual connection
        out = x + self.dropout(attn_out)   # (batch, T, hidden_dim)

        return out, attn_weights


# ---------------------------------------------------------------------------
# Stage 2: Additive Attention Pooling (Bahdanau-style)
# ---------------------------------------------------------------------------

class AdditiveAttentionPool(nn.Module):
    """
    Computes a scalar importance score for each temporal position,
    then returns a weighted sum over the time dimension.

    This collapses (batch, T, D) → (batch, D) and produces interpretable
    per-frame attention weights useful for debugging and visualisation.

    Parameters
    ----------
    hidden_dim  : int — input feature dimension (512)
    attn_dim    : int — intermediate projection dimension (128)
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_dim:   int,
    ) -> None:
        super().__init__()

        # Project hidden states to attention space
        self.W = nn.Linear(hidden_dim, attn_dim, bias=True)
        # Scalar score per projected vector
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, hidden_dim)

        Returns
        -------
        context : torch.Tensor, shape (batch, hidden_dim)
            Attention-weighted temporal summary.
        weights : torch.Tensor, shape (batch, T)
            Normalised importance scores (sum to 1 over T).
        """
        # (batch, T, attn_dim)
        energy = torch.tanh(self.W(x))

        # (batch, T, 1) → (batch, T)
        scores = self.v(energy).squeeze(-1)

        # Softmax over time axis
        weights = F.softmax(scores, dim=-1)    # (batch, T)

        # Weighted sum over time
        context = (x * weights.unsqueeze(-1)).sum(dim=1)   # (batch, hidden_dim)

        return context, weights


# ---------------------------------------------------------------------------
# Combined temporal attention block
# ---------------------------------------------------------------------------

class TemporalAttentionBlock(nn.Module):
    """
    Full temporal attention: MultiheadSelfAttention → AdditiveAttentionPool.

    Combines contextual frame-to-frame reasoning (self-attention) with
    discriminative frame selection (additive pooling).

    Parameters
    ----------
    hidden_dim : int   — BiLSTM output dim (512)
    num_heads  : int   — self-attention heads (4)
    attn_dim   : int   — pooling projection dim (128)
    dropout    : float — dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads:  int,
        attn_dim:   int,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pool = AdditiveAttentionPool(
            hidden_dim=hidden_dim,
            attn_dim=attn_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, hidden_dim)

        Returns
        -------
        context       : (batch, hidden_dim) — final context vector
        pool_weights  : (batch, T)          — additive pooling weights
        self_attn_map : (batch, T, T)       — self-attention weight matrix
        """
        # Stage 1: self-attention over frames
        attended, self_attn_map = self.self_attention(x)   # (batch, T, D)

        # Stage 2: compress to single context vector
        context, pool_weights = self.pool(attended)        # (batch, D)

        return context, pool_weights, self_attn_map
