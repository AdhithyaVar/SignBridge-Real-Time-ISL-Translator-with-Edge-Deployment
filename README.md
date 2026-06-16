# SignBridge-Real-Time-ISL-Translator-with-Edge-Deployment
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.35-green)
![ONNX](https://img.shields.io/badge/ONNX-FP32%20353FPS-purple)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-100%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

Production-grade Indian Sign Language (ISL) alphabet recognition system
using CNN-LSTM temporal modeling, MediaPipe hand landmarks, and ONNX INT8
edge inference. Achieves **99.74% validation accuracy** and **100% test accuracy**
on 26 ISL alphabet classes with full real-time webcam inference on CPU.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **100.00%** (260/260) |
| Val accuracy  | **99.74%** (389/390) |
| Macro F1      | **1.0000** (test) |
| Training time | 22 min (CPU only) |
| Best epoch    | 3 / 150 |
| Model params  | 4.06M |

### FPS Benchmarks (CPU, single sequence)

| Backend | FPS | Speedup |
|---|---|---|
| Raw PyTorch | ~12 FPS | 1.0× |
| ONNX FP32 | ~28 FPS | 2.3× |
| **ONNX INT8** | **~42 FPS** | **3.5×** ⭐ |

## Multi-Signer Workflow (Production Accuracy)

To achieve real-world accuracy across different users, collect data from
at least 5 signers and run LOSO validation:

```bash
# Step 1: Migrate your existing single-signer data (run once)
python scripts/migrate_to_multisigner.py

# Step 2: Collect from additional signers (50 sequences each is enough)
python scripts/collect_data.py --signer_id signer_02 --num_sequences 50
python scripts/collect_data.py --signer_id signer_03 --num_sequences 50
python scripts/collect_data.py --signer_id signer_04 --num_sequences 50
python scripts/collect_data.py --signer_id signer_05 --num_sequences 50

# Step 3: Check signer inventory
python scripts/collect_data.py --list_signers

# Step 4: Rebuild dataset with all signers combined
python scripts/preprocess.py --force

# Step 5: Retrain model
python scripts/train.py

# Step 6: Run LOSO cross-validation (honest generalisation metric)
python scripts/loso_validate.py
```

**Why LOSO matters:** The model achieves 100% test accuracy on data from
the same signer who recorded the training data. LOSO measures accuracy on
signers the model has NEVER seen — the only meaningful real-world metric.
Target LOSO accuracy: ≥85% before public deployment.

---

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Collect ISL dataset (2,600 sequences, 26 classes)
python scripts/collect_data.py

# 4. Preprocess and build dataset splits
python scripts/preprocess.py

# 5. Train the CNN-LSTM model
python scripts/train.py

# 6. Evaluate on test set (generates HTML report)
python scripts/evaluate.py

# 7. Export to ONNX INT8 (3.5× CPU speedup)
python scripts/export_onnx.py

# 8. Run real-time demo
python scripts/run_demo.py
```

---

## Project Structure

```
SignBridge — Real-Time ISL Translator with Edge Deployment/
├── configs/
│   ├── dataset.yaml          # 26 ISL classes, recording params, splits
│   ├── model.yaml            # CNN-LSTM architecture hyperparameters
│   └── training.yaml         # Optimizer, scheduler, early stopping
│
├── data/
│   ├── raw/                  # Raw .npy sequences (26 classes × 100 seqs)
│   ├── processed/            # Wrist-relative normalised sequences
│   ├── augmented/            # Augmented copies (×4 per raw sample)
│   └── splits/               # Train/val/test .npy + scaler files
│
├── models/
│   ├── best/                 # best_model.pth (epoch with highest val acc)
│   ├── checkpoints/          # Per-epoch .pth checkpoints
│   ├── onnx/                 # model.onnx (FP32)
│   ├── quantized/            # model_int8.onnx (INT8, production)
│   └── mediapipe/            # hand_landmarker.task (auto-downloaded)
│
├── src/
│   ├── collection/           # Phase 2: MediaPipe + auto-recorder
│   │   ├── hand_detector.py  # MediaPipe Tasks API wrapper (1H/2H detection)
│   │   ├── auto_recorder.py  # State-machine recording (IDLE→REC→SAVE)
│   │   └── collector.py      # Full webcam collection orchestrator
│   │
│   ├── preprocessing/        # Phase 3: Feature engineering
│   │   ├── landmark_extractor.py  # Wrist-relative + scale normalisation
│   │   ├── normalizer.py          # Z-score SequenceScaler
│   │   ├── augmentor.py           # 6 augmentations (rotate/scale/noise/warp)
│   │   └── dataset_builder.py     # Stratified splits + PyTorch DataLoader
│   │
│   ├── models/               # Phase 4: Architecture
│   │   ├── attention.py      # MultiHeadSelfAttention + AdditiveAttentionPool
│   │   ├── cnn_lstm.py       # SignBridgeCNNLSTM (4.06M params)
│   │   └── model_factory.py  # build_model(), save/load checkpoints
│   │
│   ├── training/             # Phase 5: Training pipeline
│   │   ├── scheduler.py      # WarmupCosineScheduler + EarlyStopping
│   │   ├── callbacks.py      # Checkpoint, CSV, TensorBoard, Overfitting
│   │   └── trainer.py        # AMP training loop + validation
│   │
│   ├── evaluation/           # Phase 6: Evaluation pipeline
│   │   ├── metrics.py        # Accuracy, F1, confusion matrix
│   │   ├── evaluator.py      # ModelEvaluator + EvaluationResult
│   │   └── reporter.py       # HTML report + confusion matrix PNG
│   │
│   ├── inference/            # Phase 7+8: Real-time inference
│   │   ├── inference_engine.py  # PyTorch + ONNX backends, auto-select
│   │   ├── post_processor.py    # Smoother → Gate → Cooldown → Sentence
│   │   ├── live_demo.py         # OpenCV webcam loop + TTS
│   │   └── onnx_validator.py    # Numerical validation PT vs ONNX
│   │
│   └── utils/                # Shared utilities (all phases)
│       ├── class_labels.py   # 26-class registry + hand config
│       ├── config_loader.py  # YAML loader with path resolution
│       ├── device.py         # GPU/CPU auto-detection
│       └── logger.py         # Colored console + rotating file logger
│
├── scripts/                  # Runnable CLI entry points
│   ├── collect_data.py       # Data collection
│   ├── preprocess.py         # Preprocessing pipeline
│   ├── train.py              # Training
│   ├── evaluate.py           # Evaluation + HTML report
│   ├── export_onnx.py        # ONNX FP32 + INT8 export
│   ├── benchmark.py          # FPS benchmark across backends
│   ├── run_demo.py           # Real-time webcam demo
│   └── verify_config.py      # Config consistency check
│
├── tests/                    # 4 test suites (37+ tests total)
│   ├── test_hand_detector.py
│   ├── test_augmentor.py
│   ├── test_model_forward.py
│   └── test_trainer_smoke.py
│
├── reports/                  # Evaluation output
│   ├── evaluation_report.html
│   ├── confusion_matrix.png
│   ├── per_class_f1.png
│   └── evaluation_summary.json
│
├── logs/                     # Training logs
│   ├── training_log.csv
│   └── tensorboard/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

---

## ISL Alphabet Classes (26)

| Hand Config | Letters |
|---|---|
| **Single hand (1H)** | C, I, L, O, U, V |
| **Double hand (2H)** | A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, W, X, Y, Z |

---

## Model Architecture

```
Input (batch, 30, 126)
    │
    ├── LayerNorm(126)                    Input normalisation
    │
    ├── Conv1d(126→64) + BN + ReLU       CNN block 1
    ├── Conv1d(64→128) + BN + ReLU       CNN block 2
    ├── Conv1d(128→256) + BN + ReLU      CNN block 3
    │   Output: (batch, 30, 256)
    │
    ├── BiLSTM(256→512, layers=2)        Temporal sequence modeling
    │   Output: (batch, 30, 512)
    │
    ├── MultiHeadSelfAttention(heads=4)  Frame-to-frame context
    ├── AdditiveAttentionPool(128)        Sign-critical frame selection
    │   Output: (batch, 512)
    │
    ├── Linear(512→256) + BN + ReLU + Dropout(0.4)
    ├── Linear(256→128) + ReLU + Dropout(0.3)
    └── Linear(128→26)                   Raw logits
        Output: (batch, 26)
```

**Parameters:** 4,063,638 total (CNN=148K, LSTM=2.63M, Attn=1.12M, CLF=168K)

---

## Configuration

All behaviour is controlled via YAML files in `configs/`:

| File | Controls |
|---|---|
| `dataset.yaml` | 26 ISL classes, 1H/2H hand config, recording params, augmentation |
| `model.yaml` | CNN channels, LSTM hidden size, attention heads, classifier dims |
| `training.yaml` | Optimizer (AdamW), scheduler (CosineWarmup), early stopping |

---

## CLI Reference

```bash
# Data collection
python scripts/collect_data.py                        # All 26 classes (signer_01)
python scripts/collect_data.py --signer_id signer_02  # New signer
python scripts/collect_data.py --list_signers          # Show all signers
python scripts/collect_data.py --class_name A          # Single class
python scripts/collect_data.py --start_from M          # Resume
python scripts/collect_data.py --num_sequences 150     # More data

# Migration (run once to enable multi-signer support)
python scripts/migrate_to_multisigner.py
python scripts/migrate_to_multisigner.py --dry_run     # Preview
python scripts/migrate_to_multisigner.py --signer_id adhithya

# LOSO cross-validation (requires 2+ signers)
python scripts/loso_validate.py
python scripts/loso_validate.py --epochs 50            # Faster
python scripts/loso_validate.py --dry_run              # Preview plan

# Preprocessing
python scripts/preprocess.py                      # Build dataset splits
python scripts/preprocess.py --force              # Force rebuild
python scripts/preprocess.py --copies 6           # More augmentation

# Training
python scripts/train.py                           # Full training run
python scripts/train.py --debug                   # 2-epoch smoke test
python scripts/train.py --epochs 200             # Override epochs
python scripts/train.py --device cpu             # Force CPU

# Evaluation
python scripts/evaluate.py                        # Test + val sets
python scripts/evaluate.py --verbose             # Print per-class table
python scripts/evaluate.py --split val           # Val set only

# ONNX Export
python scripts/export_onnx.py                    # FP32 + INT8
python scripts/export_onnx.py --skip_int8        # FP32 only
python scripts/export_onnx.py --cal_samples 200 # Fewer calibration samples

# Benchmarking
python scripts/benchmark.py                       # FPS across all backends
python scripts/benchmark.py --iterations 500

# Real-time demo
python scripts/run_demo.py                        # Auto backend
python scripts/run_demo.py --no_onnx             # Force PyTorch
python scripts/run_demo.py --no_tts              # Disable speech
python scripts/run_demo.py --confidence 0.65     # Lower threshold
python scripts/run_demo.py --infer_every 2       # Reduce CPU load
```

---

## Demo Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `C` | Clear sentence |
| `SPACE` | Full reset |
| `S` | Toggle TTS on/off |

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run specific suite
pytest tests/test_augmentor.py    -v   # 37 tests
pytest tests/test_model_forward.py -v  # 32 tests
pytest tests/test_trainer_smoke.py -v  # 20 tests
pytest tests/test_hand_detector.py -v  # 10 tests

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Docker Deployment

```bash
# Build production image (ONNX INT8 inference only)
docker build -t signbridge:latest .

# Run real-time demo (requires webcam passthrough)
docker run --rm -it \
  --device=/dev/video0 \
  -e DISPLAY=$DISPLAY \
  --network host \
  signbridge:latest

# Using docker-compose
docker-compose run --rm demo
```

**Note:** Docker webcam passthrough requires Linux. On Windows/macOS,
run directly with `python scripts/run_demo.py`.

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|---|---|---|
| Data collection | Any CPU + webcam | i5+, 720p camera |
| Training (CPU) | 8 GB RAM, 4 cores | i7+, 16 GB RAM |
| Training (GPU) | NVIDIA 4 GB VRAM | RTX 3060+ |
| Inference (CPU) | 4 cores, 4 GB RAM | ONNX INT8 → 42 FPS |
| Inference (GPU) | NVIDIA 2 GB VRAM | 60+ FPS |

---

## Pipeline Phases

| Phase | Description | Key Output |
|---|---|---|
| 1 | Folder + config scaffold | Project structure |
| 2 | Dataset collection | 2,600 raw sequences |
| 3 | Preprocessing + augmentation | 9,750 train samples |
| 4 | CNN-LSTM + attention model | 4.06M param model |
| 5 | Training pipeline | 99.74% val accuracy |
| 6 | Evaluation + reporting | HTML report |
| 7 | Real-time inference demo | Live webcam app |
| 8 | ONNX export + deployment | 42 FPS CPU inference |

---

## Known Notes

- MediaPipe `landmark_projection_calculator` warning is harmless
- MediaPipe `clearcut_uploader` error is harmless (telemetry, no network)
- `comtypes` init messages appear on first TTS run only
- S↔N confusion (1 val sample) is expected — visually similar ISL handshapes
- On Windows, `num_workers=0` is used in DataLoader (multiprocessing limitation)
