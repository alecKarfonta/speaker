#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Move to the parent directory of the scripts folder
cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -t tts-api .

echo "Docker image built successfully!"