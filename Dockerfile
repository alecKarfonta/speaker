# syntax=docker/dockerfile:1.4
# ^^^ Enable BuildKit features (cache mounts)

# Use vLLM official image - already has vLLM + numpy 2.x + PyTorch
# This provides ~2x faster LLM inference via PagedAttention
FROM vllm/vllm-openai:latest

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/GLM-TTS

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
    git \
    ffmpeg \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# ===== PYTHON DEPENDENCIES =====
# Install packaging tools first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging ninja

# Install GLM-TTS required transformers version
# Note: vLLM may have its own transformers version, we override for GLM-TTS compatibility
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade transformers==4.57.3

# Copy and install stable dependencies
COPY requirements-stable.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-stable.txt

# Copy and install ML dependencies (except vLLM which is already in base)
COPY requirements-ml.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-ml.txt

# Clone GLM-TTS code
RUN git clone --depth 1 https://github.com/zai-org/GLM-TTS.git GLM-TTS && \
    rm -rf GLM-TTS/.git GLM-TTS/ckpt && \
    mkdir -p /app/data/voices /app/logs

# Patch GLM-TTS for dtype consistency (fixes FP16/FP32 matmul errors)
COPY scripts/patch_glm_tts.py /tmp/patch_glm_tts.py
RUN python3 /tmp/patch_glm_tts.py /app/GLM-TTS

# ===== FLASH-ATTN (BUILD FROM SOURCE) =====
# Build for multiple GPU architectures:
# - 8.6: RTX 3090/3090 Ti (Ampere)
# - 12.0: RTX 5090 (Blackwell sm_120)
# vLLM already has flash-attn, but we ensure latest for GLM-TTS
RUN --mount=type=cache,target=/root/.cache/pip \
    MAX_JOBS=10 TORCH_CUDA_ARCH_LIST="8.6;12.0" pip install "flash-attn>=2.7.0" --no-build-isolation --no-cache-dir && \
    python3 -c "import flash_attn; print('Flash Attention OK:', flash_attn.__version__)"

# Verify vLLM is available
RUN python3 -c "import vllm; print('vLLM OK:', vllm.__version__)"

# ===== APPLICATION CODE (LAST - changes frequently) =====
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY start_api.sh .

# Copy voice files
COPY data/voices/ ./data/voices/

# Make scripts executable
RUN chmod +x scripts/*.sh start_api.sh

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Override vLLM's default entrypoint (vllm serve) with our application
ENTRYPOINT []
CMD ["python3", "-m", "app.main"]
