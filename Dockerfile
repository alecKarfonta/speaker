# syntax=docker/dockerfile:1.4
# ^^^ Enable BuildKit features (cache mounts)

# Use NVIDIA CUDA base image with Python for GPU support
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/GLM-TTS

# Install system dependencies (rarely change - good cache layer)
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    git \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# ===== STABLE DEPENDENCY LAYER =====
# These rarely change, keep them cached separately
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging ninja

# Upgrade transformers before other requirements (NGC container has older version)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade --force-reinstall transformers==4.57.3

# ===== STABLE PYTHON DEPS (rarely change) =====
COPY requirements-stable.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy<2.0" && pip install -r requirements-stable.txt

# ===== ML PYTHON DEPS (change more often) =====
COPY requirements-ml.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-ml.txt

# Clone GLM-TTS code (before flash-attn, rarely changes)
RUN git clone --depth 1 https://github.com/zai-org/GLM-TTS.git GLM-TTS && \
    rm -rf GLM-TTS/.git GLM-TTS/ckpt && \
    mkdir -p /app/data/voices /app/logs

# ===== FLASH-ATTN (BUILD FROM SOURCE) =====
# Build AFTER all other deps to ensure ABI compatibility with final torch version
# IMPORTANT: Use --no-cache-dir to prevent using a wheel built against wrong PyTorch!
RUN MAX_JOBS=10 TORCH_CUDA_ARCH_LIST="8.6" pip install "flash-attn>=2.1.0" --no-build-isolation --no-cache-dir && \
    python -c "import flash_attn; print('Flash Attention OK:', flash_attn.__version__)"

# ===== APPLICATION CODE (LAST - changes frequently) =====
# Copy application code AFTER flash-attn so code changes don't trigger rebuild
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY start_api.sh .

# Copy voice files into the container
COPY data/voices/ ./data/voices/

# Make scripts executable
RUN chmod +x scripts/*.sh start_api.sh

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["./start_api.sh"]
