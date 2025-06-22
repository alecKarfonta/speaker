# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Fish Audio and OpenAudio S1
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libsamplerate0-dev \
    portaudio19-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for caching 
ENV HF_HOME=/app/model_cache/huggingface
ENV TRANSFORMERS_CACHE=/app/model_cache/transformers
ENV HF_DATASETS_CACHE=/app/model_cache/datasets
ENV TORCH_HOME=/app/model_cache/torch
ENV PYTHONPATH=/app:/app/
ENV COQUI_TOS_AGREED=1

# Create cache and data directories
RUN mkdir -p /app/model_cache/huggingface \
             /app/model_cache/transformers \
             /app/model_cache/datasets \
             /app/model_cache/torch \
             /app/data/voices \
             /app/logs

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./
COPY data/ ./data/
COPY scripts/ ./scripts/

# Create .project-root file (required by fish-speech components)
RUN touch .project-root

# Make scripts executable
RUN chmod +x scripts/*.sh

# Expose port (matching docker-compose)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Add app to the path
ENV PYTHONPATH=/app

# Run setup and start the application
CMD ["bash", "-c", "./scripts/start.sh"]
