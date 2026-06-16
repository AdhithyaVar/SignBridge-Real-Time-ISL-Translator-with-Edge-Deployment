"""
test_model_forward.py
---------------------
Unit and integration tests for the SignBridge CNN-LSTM model.

Tests cover:
  - Attention module shapes and numerical properties
  - Full model forward pass shapes on CPU
  - Forward pass with attention weights
  - Gradient flow (no dead/exploding gradients)
  - Weight initialisation (forget gate bias, orthogonal LSTM)
  - Model save / load roundtrip
  - build_model() from config
  - print_model_summary()

Run:
    pytest tests/test_model_forward.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.attention import (
    MultiHeadSelfAttention,
    AdditiveAttentionPool,
    TemporalAttentionBlock,
)
from src.models.cnn_lstm import SignBridgeCNNLSTM
from src.models.model_factory import (
    build_model,
    save_checkpoint,
    load_checkpoint,
    print_model_summary,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BATCH        = 8
SEQ_LEN      = 30
FEAT_DIM     = 126
NUM_CLASSES  = 26
LSTM_OUT_DIM = 512    # hidden_size(256) × 2 (bidir)
DEVICE       = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model() -> SignBridgeCNNLSTM:
    """Default model with config-matched architecture on CPU."""
    m = SignBridgeCNNLSTM(
        num_classes=NUM_CLASSES,
        sequence_length=SEQ_LEN,
        feature_dim=FEAT_DIM,
        cnn_channels=[64, 128, 256],
        lstm_hidden=256,
        lstm_layers=2,
        lstm_dropout=0.3,
        attn_heads=4,
        attn_dim=128,
        attn_dropout=0.1,
        clf_dims=[256, 128],
        clf_dropout=[0.4, 0.3],
    )
    m.eval()
    return m


@pytest.fixture
def batch_input() -> torch.Tensor:
    """Random batch of normalised landmark sequences."""
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ_LEN, FEAT_DIM)


@pytest.fixture
def single_input() -> torch.Tensor:
    """Single sample for inference tests."""
    torch.manual_seed(1)
    return torch.randn(1, SEQ_LEN, FEAT_DIM)


@pytest.fixture
def minimal_configs() -> tuple[dict, dict]:
    """Minimal model_cfg and dataset_cfg dicts for build_model()."""
    model_cfg = {
        "input": {
            "sequence_length": SEQ_LEN,
            "feature_dim":     FEAT_DIM,
        },
        "cnn": {
            "conv_layers": [
                {"out_channels": 64,  "kernel_size": 3,
                 "padding": 1, "dropout": 0.0, "batch_norm": True},
                {"out_channels": 128, "kernel_size": 3,
                 "padding": 1, "dropout": 0.1, "batch_norm": True},
                {"out_channels": 256, "kernel_size": 3,
                 "padding": 1, "dropout": 0.1, "batch_norm": True},
            ],
            "cnn_output_dim": 256,
        },
        "lstm": {
            "hidden_size":   256,
            "num_layers":    2,
            "bidirectional": True,
            "dropout":       0.3,
        },
        "attention": {
            "enabled":       True,
            "num_heads":     4,
            "attention_dim": 128,
            "dropout":       0.1,
        },
        "classifier": {
            "hidden_layers": [
                {"dim": 256, "activation": "relu",
                 "dropout": 0.4, "batch_norm": True},
                {"dim": 128, "activation": "relu",
                 "dropout": 0.3, "batch_norm": False},
            ],
            "output_dim": NUM_CLASSES,
        },
    }
    dataset_cfg = {
        "num_classes": NUM_CLASSES,
        "recording":   {"sequence_length": SEQ_LEN},
        "features":    {"two_hand_features": FEAT_DIM},
    }
    return model_cfg, dataset_cfg


# ============================================================================
# Tests: Attention modules
# ============================================================================

class TestMultiHeadSelfAttention:

    def test_output_shape(self):
        attn = MultiHeadSelfAttention(hidden_dim=512, num_heads=4)
        x    = torch.randn(BATCH, SEQ_LEN, 512)
        out, weights = attn(x)
        assert out.shape     == (BATCH, SEQ_LEN, 512), \
            f"Expected ({BATCH},{SEQ_LEN},512), got {out.shape}"
        assert weights.shape == (BATCH, SEQ_LEN, SEQ_LEN), \
            f"Expected ({BATCH},{SEQ_LEN},{SEQ_LEN}), got {weights.shape}"

    def test_residual_preserves_shape(self):
        attn = MultiHeadSelfAttention(hidden_dim=512, num_heads=4)
        x    = torch.randn(BATCH, SEQ_LEN, 512)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_attention_weights_sum_to_one(self):
        attn = MultiHeadSelfAttention(hidden_dim=512, num_heads=4)
        x    = torch.randn(4, SEQ_LEN, 512)
        _, weights = attn(x)
        row_sums = weights.sum(dim=-1)   # sum over key dimension
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Attention weights per row should sum to 1."

    def test_invalid_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadSelfAttention(hidden_dim=512, num_heads=3)


class TestAdditiveAttentionPool:

    def test_output_shapes(self):
        pool    = AdditiveAttentionPool(hidden_dim=512, attn_dim=128)
        x       = torch.randn(BATCH, SEQ_LEN, 512)
        context, weights = pool(x)
        assert context.shape == (BATCH, 512)
        assert weights.shape == (BATCH, SEQ_LEN)

    def test_pool_weights_sum_to_one(self):
        pool    = AdditiveAttentionPool(hidden_dim=512, attn_dim=128)
        x       = torch.randn(BATCH, SEQ_LEN, 512)
        _, weights = pool(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_context_is_weighted_combination(self):
        """Verify context = sum_t(weight_t * x_t) within numerical tolerance."""
        pool = AdditiveAttentionPool(hidden_dim=8, attn_dim=4)
        pool.eval()
        x = torch.randn(2, SEQ_LEN, 8)
        with torch.no_grad():
            context, weights = pool(x)
        manual = (x * weights.unsqueeze(-1)).sum(dim=1)
        assert torch.allclose(context, manual, atol=1e-6)


class TestTemporalAttentionBlock:

    def test_output_shapes(self):
        block = TemporalAttentionBlock(
            hidden_dim=512, num_heads=4, attn_dim=128, dropout=0.0
        )
        x = torch.randn(BATCH, SEQ_LEN, 512)
        context, pool_w, self_attn = block(x)
        assert context.shape   == (BATCH, 512)
        assert pool_w.shape    == (BATCH, SEQ_LEN)
        assert self_attn.shape == (BATCH, SEQ_LEN, SEQ_LEN)

    def test_deterministic_in_eval_mode(self):
        block = TemporalAttentionBlock(
            hidden_dim=512, num_heads=4, attn_dim=128, dropout=0.1
        )
        block.eval()
        x = torch.randn(2, SEQ_LEN, 512)
        with torch.no_grad():
            c1, _, _ = block(x)
            c2, _, _ = block(x)
        assert torch.allclose(c1, c2), "Eval mode should be deterministic."


# ============================================================================
# Tests: SignBridgeCNNLSTM forward pass
# ============================================================================

class TestModelForward:

    def test_output_shape(self, model, batch_input):
        with torch.no_grad():
            logits = model(batch_input)
        assert logits.shape == (BATCH, NUM_CLASSES), \
            f"Expected ({BATCH},{NUM_CLASSES}), got {logits.shape}"

    def test_output_dtype(self, model, batch_input):
        with torch.no_grad():
            logits = model(batch_input)
        assert logits.dtype == torch.float32

    def test_single_sample_shape(self, model, single_input):
        with torch.no_grad():
            logits = model(single_input)
        assert logits.shape == (1, NUM_CLASSES)

    def test_forward_with_attention_shapes(self, model, batch_input):
        with torch.no_grad():
            logits, pool_w, self_attn = model.forward_with_attention(batch_input)
        assert logits.shape   == (BATCH, NUM_CLASSES)
        assert pool_w.shape   == (BATCH, SEQ_LEN)
        assert self_attn.shape == (BATCH, SEQ_LEN, SEQ_LEN)

    def test_predict_proba_sums_to_one(self, model, batch_input):
        with torch.no_grad():
            proba = model.predict_proba(batch_input)
        row_sums = proba.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Softmax probabilities should sum to 1."

    def test_predict_returns_valid_class_indices(self, model, batch_input):
        with torch.no_grad():
            preds = model.predict(batch_input)
        assert preds.shape == (BATCH,)
        assert (preds >= 0).all() and (preds < NUM_CLASSES).all()

    def test_eval_mode_is_deterministic(self, model, batch_input):
        model.eval()
        with torch.no_grad():
            out1 = model(batch_input)
            out2 = model(batch_input)
        assert torch.allclose(out1, out2), \
            "Model should be deterministic in eval mode."

    def test_no_nan_in_output(self, model, batch_input):
        with torch.no_grad():
            logits = model(batch_input)
        assert not torch.isnan(logits).any(), "Model output contains NaN values."

    def test_no_inf_in_output(self, model, batch_input):
        with torch.no_grad():
            logits = model(batch_input)
        assert not torch.isinf(logits).any(), "Model output contains Inf values."

    def test_different_batch_sizes(self, model):
        """Model must handle any batch size including 1."""
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, SEQ_LEN, FEAT_DIM)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (bs, NUM_CLASSES), \
                f"Failed for batch_size={bs}"

    def test_zero_input_no_crash(self, model):
        """All-zero input (absent hands) should not crash."""
        x = torch.zeros(BATCH, SEQ_LEN, FEAT_DIM)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)
        assert not torch.isnan(out).any()


# ============================================================================
# Tests: Gradient flow
# ============================================================================

class TestGradientFlow:

    def test_gradients_flow_to_all_params(self):
        """All trainable parameters should receive gradients."""
        m = SignBridgeCNNLSTM(
            num_classes=NUM_CLASSES,
            sequence_length=SEQ_LEN,
            feature_dim=FEAT_DIM,
        )
        m.train()
        x      = torch.randn(4, SEQ_LEN, FEAT_DIM)
        target = torch.randint(0, NUM_CLASSES, (4,))
        logits = m(x)
        loss   = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        no_grad = []
        for name, param in m.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)

        assert len(no_grad) == 0, \
            f"Parameters with no gradient: {no_grad}"

    def test_no_exploding_gradients(self):
        """Gradient norms should be finite after one backward pass."""
        m = SignBridgeCNNLSTM(
            num_classes=NUM_CLASSES,
            sequence_length=SEQ_LEN,
            feature_dim=FEAT_DIM,
        )
        m.train()
        x      = torch.randn(4, SEQ_LEN, FEAT_DIM)
        target = torch.randint(0, NUM_CLASSES, (4,))
        logits = m(x)
        loss   = nn.CrossEntropyLoss()(logits, target)
        loss.backward()

        for name, param in m.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), \
                    f"Inf gradient in {name}"


# ============================================================================
# Tests: Weight initialisation
# ============================================================================

class TestWeightInitialisation:

    def test_lstm_forget_gate_bias_is_one(self):
        """
        Forget gate bias should be initialised to 1.0 to help the LSTM
        remember long sequences from the start of training.
        """
        m = SignBridgeCNNLSTM(
            num_classes=NUM_CLASSES,
            sequence_length=SEQ_LEN,
            feature_dim=FEAT_DIM,
        )
        for layer_idx in range(m.lstm.num_layers):
            for direction in range(2):   # 0=fwd, 1=bwd
                suffix = "_reverse" if direction == 1 else ""
                bias_name = f"bias_hh_l{layer_idx}{suffix}"
                bias = getattr(m.lstm, bias_name).data
                hidden = bias.shape[0] // 4
                forget_bias = bias[hidden: 2 * hidden]
                assert torch.allclose(
                    forget_bias, torch.ones_like(forget_bias), atol=1e-5
                ), f"LSTM forget gate bias not 1.0 in {bias_name}"

    def test_bn_weight_ones_bias_zeros(self):
        """BatchNorm layers should start with weight=1, bias=0."""
        m = SignBridgeCNNLSTM(
            num_classes=NUM_CLASSES,
            sequence_length=SEQ_LEN,
            feature_dim=FEAT_DIM,
        )
        for name, module in m.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                assert torch.allclose(
                    module.weight.data,
                    torch.ones_like(module.weight.data),
                    atol=1e-6,
                ), f"BN weight not 1 in {name}"
                assert torch.allclose(
                    module.bias.data,
                    torch.zeros_like(module.bias.data),
                    atol=1e-6,
                ), f"BN bias not 0 in {name}"


# ============================================================================
# Tests: Model factory
# ============================================================================

class TestModelFactory:

    def test_build_model_from_config(self, minimal_configs):
        model_cfg, dataset_cfg = minimal_configs
        m = build_model(model_cfg, dataset_cfg, device=DEVICE)
        assert isinstance(m, SignBridgeCNNLSTM)
        assert m.num_classes == NUM_CLASSES

    def test_build_model_forward_shape(self, minimal_configs):
        model_cfg, dataset_cfg = minimal_configs
        m = build_model(model_cfg, dataset_cfg, device=DEVICE)
        m.eval()
        x   = torch.randn(BATCH, SEQ_LEN, FEAT_DIM)
        out = m(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_save_and_load_checkpoint(self, model, tmp_path):
        optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3)
        metrics    = {"val_accuracy": 0.95, "val_loss": 0.12}
        save_path  = tmp_path / "test_ckpt.pth"

        save_checkpoint(model, optimizer, epoch=5, metrics=metrics,
                        save_path=save_path)

        assert save_path.exists(), "Checkpoint file was not created."

        # Load into a fresh model and verify weights match
        new_model = SignBridgeCNNLSTM(
            num_classes=NUM_CLASSES,
            sequence_length=SEQ_LEN,
            feature_dim=FEAT_DIM,
        )
        new_model, epoch, loaded_metrics = load_checkpoint(
            new_model, save_path, device=DEVICE
        )
        assert epoch == 5
        assert loaded_metrics["val_accuracy"] == pytest.approx(0.95)

        # Forward pass must produce identical output
        x = torch.randn(2, SEQ_LEN, FEAT_DIM)
        model.eval()
        new_model.eval()
        with torch.no_grad():
            out_orig = model(x)
            out_load = new_model(x)
        assert torch.allclose(out_orig, out_load, atol=1e-6), \
            "Loaded model outputs differ from original."

    def test_best_model_saved_separately(self, model, tmp_path):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        best_dir  = tmp_path / "best"
        save_checkpoint(
            model, optimizer, epoch=10,
            metrics={"val_accuracy": 0.92},
            save_path=tmp_path / "epoch_010.pth",
            is_best=True,
            best_dir=best_dir,
        )
        assert (best_dir / "best_model.pth").exists()

    def test_print_model_summary_no_crash(self, model, capsys):
        print_model_summary(model)
        captured = capsys.readouterr()
        assert "SignBridge" in captured.out
        assert "TOTAL" in captured.out

    def test_count_parameters_positive(self, model):
        counts = model.count_parameters()
        for key, val in counts.items():
            assert val > 0, f"Parameter count for '{key}' is zero."
