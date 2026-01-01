# Use NVIDIA CUDA base image with Python for GPU support
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/GLM-TTS

# Install system dependencies
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

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install torch first (required for flash-attn build)
#RUN pip install torch>=2.0.0 torchaudio>=2.0.0

# Install flash-attn separately (requires torch to be installed first)
#RUN pip install packaging
#RUN pip install ninja
#RUN MAX_JOBS=16 pip install  flash-attn --no-build-isolation

# Install remaining dependencies
RUN pip install  -r requirements.txt

# Copy the application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY start_api.sh .

# Copy voice files into the container
COPY data/voices/ ./data/voices/

# Clone GLM-TTS code (without model checkpoints - mount those at runtime)
RUN git clone --depth 1 https://github.com/zai-org/GLM-TTS.git GLM-TTS && \
    rm -rf GLM-TTS/.git GLM-TTS/ckpt

# Make scripts executable
RUN chmod +x scripts/*.sh
RUN chmod +x start_api.sh

# Create necessary directories
RUN mkdir -p /app/data/voices
RUN mkdir -p /app/logs

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["./start_api.sh"]
