#!/usr/bin/env bash
# Launch pwilkin/openmoss GGML server (fast C++ MOSS-TTS backend)
#
# Usage:
#   ./scripts/start-openmoss.sh                    # Q8_0 if present, else f16
#   ./scripts/start-openmoss.sh --main-gpu 0
#   ./scripts/start-openmoss.sh --quantize          # run Q8_0 quantize first
#
# WebUI:  http://127.0.0.1:8014/
# API:    http://127.0.0.1:8014/tts

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENMOSS="$ROOT/openmoss"
LLAMA_CPP="${LLAMA_CPP_DIR:-/home/alec/git/llama-nexus/llama.cpp}"
PORT="${OPENMOSS_PORT:-8014}"
HOST="${OPENMOSS_HOST:-0.0.0.0}"
MAIN_GPU="${OPENMOSS_MAIN_GPU:-}"

WEIGHTS_DIR="$OPENMOSS/weights"
# OPENMOSS_MODEL_VERSION: v15 (default) | v10
OPENMOSS_VER="${OPENMOSS_MODEL_VERSION:-v15}"
case "$OPENMOSS_VER" in
  v15|1.5)
    MODEL_F16="$WEIGHTS_DIR/moss-tts.gguf"
    MODEL_Q8="$WEIGHTS_DIR/moss-tts-v15-q8_0.gguf"
    EXTRAS="$WEIGHTS_DIR/moss-tts.extras.gguf"
    ;;
  v10|1.0)
    MODEL_F16="$WEIGHTS_DIR/moss-tts-v10.gguf"
    MODEL_Q8="$WEIGHTS_DIR/moss-tts-v10-q8_0.gguf"
    EXTRAS="$WEIGHTS_DIR/moss-tts-v10.extras.gguf"
    ;;
  *)
    echo "Unknown OPENMOSS_MODEL_VERSION=$OPENMOSS_VER (use v15 or v10)" >&2
    exit 1
    ;;
esac
# Sidecar is not quantized; symlink lets Q8 backbone find *.extras.gguf
if [[ -f "$EXTRAS" && ! -f "${MODEL_Q8%.gguf}.extras.gguf" ]]; then
  ln -sf "$(basename "$EXTRAS")" "${MODEL_Q8%.gguf}.extras.gguf"
fi
if [[ -n "${OPENMOSS_MODEL_PATH:-}" ]]; then
  MODEL="${OPENMOSS_MODEL_PATH}"
elif [[ -f "$MODEL_Q8" ]]; then
  MODEL="$MODEL_Q8"
elif [[ -f "$MODEL_F16" ]]; then
  MODEL="$MODEL_F16"
else
  MODEL=""
fi
SERVER="$OPENMOSS/build/moss-tts-server"
QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"
if [[ ! -x "$QUANTIZE" && -x /tmp/llama-quant-cpu/bin/llama-quantize ]]; then
  QUANTIZE="/tmp/llama-quant-cpu/bin/llama-quantize"
fi

export LD_LIBRARY_PATH="$LLAMA_CPP/build/bin:${LD_LIBRARY_PATH:-}"

if [[ "${1:-}" == "--quantize" ]]; then
  shift
  if [[ ! -f "$MODEL_F16" ]]; then
    echo "Missing $MODEL_F16 — run conversion first (see openmoss/README.speaker.md)"
    exit 1
  fi
  echo "Quantizing $OPENMOSS_VER backbone to Q8_0 (embeddings stay f16)..."
  "$QUANTIZE" --token-embedding-type f16 "$MODEL_F16" "$MODEL_Q8" Q8_0
  if [[ -f "$EXTRAS" && ! -f "${MODEL_Q8%.gguf}.extras.gguf" ]]; then
    ln -sf "$(basename "$EXTRAS")" "${MODEL_Q8%.gguf}.extras.gguf"
  fi
  echo "Done: $MODEL_Q8"
  exit 0
fi

if [[ ! -x "$SERVER" ]]; then
  echo "Building openmoss..."
  cmake -B "$OPENMOSS/build" -DLLAMA_CPP_DIR="$LLAMA_CPP" -DGGML_CUDA=ON
  cmake --build "$OPENMOSS/build" -j"$(nproc)"
fi

if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
  echo "No GGUF weights found in $WEIGHTS_DIR"
  echo "Run conversion first — see openmoss/README.speaker.md"
  exit 1
fi
if [[ ! -f "${MODEL%.gguf}.extras.gguf" && -f "$EXTRAS" ]]; then
  ln -sf "$(basename "$EXTRAS")" "${MODEL%.gguf}.extras.gguf"
fi

GPU_ARGS=()
if [[ -n "$MAIN_GPU" ]]; then
  # Hide other GPUs so llama.cpp does not probe/allocate ~250 MiB on every device.
  export CUDA_VISIBLE_DEVICES="$MAIN_GPU"
  GPU_ARGS+=(--main-gpu 0)
fi
if [[ "${OPENMOSS_AUX_CPU:-0}" == "1" ]]; then
  GPU_ARGS+=(--aux-cpu)
fi

echo "Starting openmoss on http://${HOST}:${PORT}/"
echo "Model ($OPENMOSS_VER): $MODEL"
if [[ -n "$MAIN_GPU" ]]; then
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (physical GPU $MAIN_GPU)"
fi
if [[ "${OPENMOSS_AUX_CPU:-0}" == "1" ]]; then
  echo "Aux backend: CPU (OPENMOSS_AUX_CPU=1)"
fi
exec "$SERVER" \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  "${GPU_ARGS[@]}" \
  "$@"
