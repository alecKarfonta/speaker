# Use the official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/GLM-TTS

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gcc \
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
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config.yaml .
COPY start_api.sh .

# Copy voice files into the container
COPY data/voices/ ./data/voices/

# Copy GLM-TTS code (without model checkpoints - mount those at runtime)
COPY GLM-TTS/cosyvoice/ ./GLM-TTS/cosyvoice/
COPY GLM-TTS/flow/ ./GLM-TTS/flow/
COPY GLM-TTS/llm/ ./GLM-TTS/llm/
COPY GLM-TTS/utils/ ./GLM-TTS/utils/
COPY GLM-TTS/frontend/ ./GLM-TTS/frontend/
COPY GLM-TTS/configs/ ./GLM-TTS/configs/

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
