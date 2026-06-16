FROM python:3.11-slim

LABEL maintainer="SignBridge Team"
LABEL description="SignBridge ISL Translator — production image with ONNX INT8 inference"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    espeak \
    espeak-data \
    libespeak1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────
# Copy requirements first for layer caching
COPY requirements.txt .

# Install ONNX runtime only (no PyTorch for production inference image)
# Remove torch/torchvision/torchaudio lines and install onnxruntime instead
RUN pip install --no-cache-dir \
    opencv-python>=4.8.0 \
    mediapipe>=0.10.7 \
    onnx>=1.15.0 \
    onnxruntime>=1.16.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    PyYAML>=6.0.1 \
    pyttsx3>=2.90 \
    tqdm>=4.65.0 \
    psutil>=5.9.0

# ── Application code ──────────────────────────────────────────────────────
COPY src/       ./src/
COPY configs/   ./configs/
COPY scripts/   ./scripts/
COPY setup.py   .

RUN pip install --no-cache-dir -e .

# ── Model artefacts ───────────────────────────────────────────────────────
# INT8 model (primary — fastest CPU inference)
COPY models/quantized/  ./models/quantized/

# FP32 fallback
COPY models/onnx/        ./models/onnx/

# Scaler (required for preprocessing at inference time)
COPY data/splits/scaler_mean.npy ./data/splits/scaler_mean.npy
COPY data/splits/scaler_std.npy  ./data/splits/scaler_std.npy

# MediaPipe hand landmarker model
COPY models/mediapipe/   ./models/mediapipe/

# ── Runtime config ────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=4

# Default: run the real-time demo
CMD ["python", "scripts/run_demo.py", "--no_tts"]

# ── Health check ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import onnxruntime; import mediapipe; print('OK')" || exit 1
