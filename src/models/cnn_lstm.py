"""
cnn_lstm.py
-----------
SignBridge CNN-LSTM model architecture.

Full architecture
-----------------

Input: (batch, T=30, F=126)
   ↓
┌─────────────────────────────────────────────────────────┐
│  INPUT NORMALISATION                                    │
│  LayerNorm(126) — stabilises varying coordinate scales  │
└─────────────────────────────────────────────────────────┘
   ↓ permute (batch, 126, 30)
┌─────────────────────────────────────────────────────────┐
│  CNN FEATURE EXTRACTOR  (3 Conv1D blocks)               │
│  Block 1: Conv1d(126→64,  k=3, p=1) + BN + ReLU        │
│  Block 2: Conv1d(64→128,  k=3, p=1) + BN + ReLU + D0.1 │
│  Block 3: Conv1d(128→256, k=3, p=1) + BN + ReLU + D0.1 │
│  Output: (batch, 256, 30)                               │
└─────────────────────────────────────────────────────────┘
   ↓ permute (batch, 30, 256)
┌─────────────────────────────────────────────────────────┐
│  BIDIRECTIONAL LSTM  (2 layers)                         │
│  hidden=256, bidir=True → output_dim=512                │
│  Inter-layer dropout=0.3                                │
│  Output: (batch, 30, 512)  — all timestep hidden states │
└─────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────────────┐
│  TEMPORAL ATTENTION BLOCK                               │
│  MultiheadSelfAttention(512, heads=4) + residual + LN   │
│  AdditiveAttentionPool(512 → 128 → 1)                   │
│  Output: (batch, 512) context vector                    │
└─────────────────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────────────────┐
│  CLASSIFIER HEAD                                        │
│  Linear(512→256) + BN + ReLU + Dropout(0.4)            │
│  Linear(256→128) + ReLU + Dropout(0.3)                 │
│  Linear(128→26)  — raw logits                           │
└─────────────────────────────────────────────────────────┘
   ↓
Output: (batch, 26)  — raw logits for CrossEntropyLoss
        (batch, 26)  — softmax probabilities at inference

Regularisation strategy (anti-overfitting)
-------------------------------------------
* BatchNorm in every CNN block          → stable gradient flow
* LSTM inter-layer dropout (0.3)        → prevents co-adaptation
* Dropout in classifier (0.4, 0.3)      → prevents memorisation
* LayerNorm on input + attention        → training stability
* Label smoothing (in loss, training.py)→ prevents overconfident predictions
* Weight decay AdamW (training.yaml)    → L2 regularisation
* Early stopping (training.yaml)        → halts before overfitting peak
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import TemporalAttentionBlock
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CNN block builder helper
# ---------------------------------------------------------------------------

def _build_cnn_block(
    in_channels:  int,
    out_channels: int,
    kernel_size:  int,
    padding:      int,
    use_bn:       bool,
    dropout:      float,
) -> nn.Sequential:
    """Build one Conv1d block: Conv → BN → ReLU → Dropout."""
    layers: list[nn.Module] = [
        nn.Conv1d(in_channels, out_channels, kernel_size,
                  padding=padding, bias=not use_bn),
    ]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class SignBridgeCNNLSTM(nn.Module):
    """
    CNN-LSTM with temporal attention for ISL alphabet recognition.

    Parameters
    ----------
    num_classes    : int   — number of output classes (26)
    sequence_length: int   — temporal window size (30)
    feature_dim    : int   — input feature dimension (126)
    cnn_channels   : list  — output channels for each CNN block
    lstm_hidden    : int   — LSTM hidden size per direction (256)
    lstm_layers    : int   — number of LSTM layers (2)
    lstm_dropout   : float — dropout between LSTM layers (0.3)
    attn_heads     : int   — attention heads (4)
    attn_dim       : int   — additive attention projection dim (128)
    attn_dropout   : float — attention dropout (0.1)
    clf_dims       : list  — classifier hidden layer dimensions ([256, 128])
    clf_dropout    : list  — classifier dropout rates ([0.4, 0.3])
    """

    def __init__(
        self,
        num_classes:     int         = 26,
        sequence_length: int         = 30,
        feature_dim:     int         = 126,
        cnn_channels:    list[int]   = None,
        lstm_hidden:     int         = 256,
        lstm_layers:     int         = 2,
        lstm_dropout:    float       = 0.3,
        attn_heads:      int         = 4,
        attn_dim:        int         = 128,
        attn_dropout:    float       = 0.1,
        clf_dims:        list[int]   = None,
        clf_dropout:     list[float] = None,
    ) -> None:
        super().__init__()

        # Defaults
        cnn_channels = cnn_channels or [64, 128, 256]
        clf_dims      = clf_dims     or [256, 128]
        clf_dropout   = clf_dropout  or [0.4, 0.3]

        self.num_classes     = num_classes
        self.sequence_length = sequence_length
        self.feature_dim     = feature_dim

        # ── 1. Input normalisation ────────────────────────────────────────
        self.input_norm = nn.LayerNorm(feature_dim)

        # ── 2. CNN feature extractor ──────────────────────────────────────
        # Dropout schedule: no dropout on first block, light on blocks 2+
        cnn_dropouts = [0.0] + [0.1] * (len(cnn_channels) - 1)

        cnn_blocks: list[nn.Module] = []
        in_ch = feature_dim
        for out_ch, drop in zip(cnn_channels, cnn_dropouts):
            cnn_blocks.append(
                _build_cnn_block(in_ch, out_ch,
                                 kernel_size=3, padding=1,
                                 use_bn=True, dropout=drop)
            )
            in_ch = out_ch
        self.cnn = nn.ModuleList(cnn_blocks)
        self._cnn_out_dim = cnn_channels[-1]   # 256

        # ── 3. Bidirectional LSTM ─────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self._cnn_out_dim,        # 256
            hidden_size=lstm_hidden,              # 256
            num_layers=lstm_layers,               # 2
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self._lstm_out_dim = lstm_hidden * 2     # 512 (bidirectional)

        # ── 4. Temporal attention ─────────────────────────────────────────
        self.attention = TemporalAttentionBlock(
            hidden_dim=self._lstm_out_dim,        # 512
            num_heads=attn_heads,                 # 4
            attn_dim=attn_dim,                    # 128
            dropout=attn_dropout,                 # 0.1
        )

        # ── 5. Classifier head ────────────────────────────────────────────
        clf_layers: list[nn.Module] = []
        in_dim = self._lstm_out_dim               # 512

        for out_dim, drop in zip(clf_dims, clf_dropout):
            clf_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
            ])
            in_dim = out_dim

        clf_layers.append(nn.Linear(in_dim, num_classes))   # logit layer
        self.classifier = nn.Sequential(*clf_layers)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"SignBridgeCNNLSTM | "
            f"classes={num_classes} | "
            f"params={total_params:,} total, {train_params:,} trainable"
        )

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (returns logits only).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, T, F)
            Normalised landmark sequences.

        Returns
        -------
        logits : torch.Tensor, shape (batch, num_classes)
            Raw logits — apply softmax for probabilities.
        """
        logits, _, _ = self._forward_full(x)
        return logits

    def forward_with_attention(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns attention weights.
        Used at inference time for visualisation and debugging.

        Returns
        -------
        logits         : (batch, num_classes)
        pool_weights   : (batch, T)    — which frames were most important
        self_attn_map  : (batch, T, T) — frame-to-frame attention
        """
        return self._forward_full(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax of logits)."""
        return F.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices (argmax of logits)."""
        return self.forward(x).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Internal forward implementation
    # ------------------------------------------------------------------

    def _forward_full(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        x: (batch, T, F)
        Returns: logits, pool_weights, self_attn_map
        """
        # ── Step 1: Input normalisation ──────────────────────────────
        x = self.input_norm(x)                  # (batch, T, 126)

        # ── Step 2: CNN (operates on feature/channel dimension) ──────
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)                  # (batch, 126, T)
        for cnn_block in self.cnn:
            x = cnn_block(x)                    # (batch, C, T)
        x = x.permute(0, 2, 1)                  # (batch, T, 256)

        # ── Step 3: BiLSTM ───────────────────────────────────────────
        x, _ = self.lstm(x)                     # (batch, T, 512)

        # ── Step 4: Temporal attention ───────────────────────────────
        context, pool_weights, self_attn_map = self.attention(x)
        # context: (batch, 512)

        # ── Step 5: Classifier ───────────────────────────────────────
        logits = self.classifier(context)       # (batch, 26)

        return logits, pool_weights, self_attn_map

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Apply standard weight initialisation:
        - Conv1d / Linear: Kaiming normal (He init for ReLU)
        - BatchNorm: weight=1, bias=0
        - LSTM: orthogonal for recurrent weights, Xavier for input weights
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        # Input-hidden: Xavier uniform
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        # Hidden-hidden: orthogonal (prevents gradient vanishing)
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1.0 (helps LSTM remember)
                        hidden = param.data.shape[0] // 4
                        param.data[hidden: 2 * hidden].fill_(1.0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per submodule."""
        def count(m): return sum(p.numel() for p in m.parameters())
        return {
            "cnn":        sum(count(b) for b in self.cnn),
            "lstm":       count(self.lstm),
            "attention":  count(self.attention),
            "classifier": count(self.classifier),
            "total":      count(self),
        }

    def get_feature_dim(self) -> int:
        """Return the context vector dimension (512) before classifier."""
        return self._lstm_out_dim

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"seq_len={self.sequence_length}, "
            f"feat_dim={self.feature_dim}"
        )
