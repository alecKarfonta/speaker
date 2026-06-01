#!/bin/bash
set -euo pipefail

WEIGHTS_DIR="${WEIGHTS_DIR:-/app/weights}"
MODEL="${OPENMOSS_MODEL:-$WEIGHTS_DIR/moss-tts-v1.5-q8_0.gguf}"
EXTRAS="${MODEL%.gguf}.extras.gguf"
HF_REPO="${OPENMOSS_HF_REPO:-smcleod/MOSS-TTS-v1.5-GGUF}"
MAIN_GPU="${OPENMOSS_MAIN_GPU:-0}"
AUX_CPU="${OPENMOSS_AUX_CPU:-false}"
N_GPU_LAYERS="${OPENMOSS_N_GPU_LAYERS:--1}"
N_CTX="${OPENMOSS_N_CTX:-4096}"
NO_FLASH_ATTN="${OPENMOSS_NO_FLASH_ATTN:-false}"

# Blackwell RTX 50-series: FA quants must match GGML_CUDA_FA_ALL_QUANTS build
export GGML_CUDA_FA_ALL_QUANTS="${GGML_CUDA_FA_ALL_QUANTS:-1}"

mkdir -p "$WEIGHTS_DIR"

download_weights() {
    echo "⬇️  Downloading MOSS-TTS-v1.5 GGUF weights from $HF_REPO (~13 GB, one-time)..."
    python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo = os.environ["OPENMOSS_HF_REPO"]
dest = os.environ["WEIGHTS_DIR"]
files = ["moss-tts-v1.5-q8_0.gguf", "moss-tts-v1.5-q8_0.extras.gguf"]
for name in files:
    path = hf_hub_download(repo_id=repo, filename=name, local_dir=dest)
    print(f"  ✓ {path}")
PY
}

if [[ ! -f "$MODEL" || ! -f "$EXTRAS" ]]; then
  if [[ "${OPENMOSS_HF_REPO:-}" == "local" ]]; then
    echo "ERROR: OPENMOSS_HF_REPO=local but weights missing:"
    echo "  MODEL=$MODEL"
    echo "  EXTRAS=$EXTRAS"
    exit 1
  fi
  pip install --break-system-packages -q huggingface_hub
  export OPENMOSS_HF_REPO="$HF_REPO"
  export WEIGHTS_DIR="$WEIGHTS_DIR"
  download_weights
fi

echo "🚀 Starting moss-tts-server model=$(basename "$MODEL") GPU=$MAIN_GPU n_gpu_layers=$N_GPU_LAYERS n_ctx=$N_CTX aux_cpu=$AUX_CPU no_flash_attn=$NO_FLASH_ATTN ..."
AUX_CPU_FLAG=()
if [[ "$AUX_CPU" == "true" || "$AUX_CPU" == "1" ]]; then
    AUX_CPU_FLAG=(--aux-cpu)
fi
FLASH_ATTN_FLAG=()
if [[ "$NO_FLASH_ATTN" == "true" || "$NO_FLASH_ATTN" == "1" ]]; then
    FLASH_ATTN_FLAG=(--no-flash-attn)
fi
moss-tts-server \
    --model "$MODEL" \
    --host 127.0.0.1 \
    --port 8081 \
    --main-gpu "$MAIN_GPU" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --n-ctx "$N_CTX" \
    --no-webui \
    "${AUX_CPU_FLAG[@]}" \
    "${FLASH_ATTN_FLAG[@]}" &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "⏳ Waiting for openmoss server..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8081/health >/dev/null 2>&1; then
        echo "✅ openmoss server ready"
        break
    fi
    sleep 2
done

echo "🌐 Starting Speaker API shim on :8000 ..."
pip install --break-system-packages -q python-multipart 2>/dev/null || true
exec python3 -m app.openmoss_shim
