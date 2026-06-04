#!/usr/bin/env bash
# Launch MOSS-TTS-Realtime standalone (PyTorch, token-level streaming)
#
# Usage:
#   ./scripts/start-moss-realtime.sh
#   MOSS_RT_GPU=3 ./scripts/start-moss-realtime.sh
#
# API:     http://127.0.0.1:8016/
# Stream:  POST http://127.0.0.1:8016/tts/stream
# Test:    python3 scripts/test_moss_stream.py --api-url http://127.0.0.1:8016

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${MOSS_RT_PORT:-8016}"
HOST="${MOSS_RT_HOST:-0.0.0.0}"
GPU="${MOSS_RT_GPU:-3}"

export MOSS_ENABLE_MAIN_MODEL=false
export MOSS_ENABLE_VOICE_GEN=false
export MOSS_ENABLE_REALTIME=true
export MOSS_ENABLE_STREAMING=true
export MOSS_RT_MODEL_ID="${MOSS_RT_MODEL_ID:-OpenMOSS-Team/MOSS-TTS-Realtime}"
export MOSS_RT_NATIVE_VOICE="${MOSS_RT_NATIVE_VOICE:-false}"
# Production decode (major_03 warm_092_072); override per deploy if needed.
export MOSS_RT_AUDIO_TEMPERATURE="${MOSS_RT_AUDIO_TEMPERATURE:-0.92}"
export MOSS_RT_AUDIO_TOP_P="${MOSS_RT_AUDIO_TOP_P:-0.72}"
export MOSS_RT_AUDIO_TOP_K="${MOSS_RT_AUDIO_TOP_K:-40}"
export MOSS_RT_AUDIO_REPETITION_PENALTY="${MOSS_RT_AUDIO_REPETITION_PENALTY:-1.05}"
export MOSS_RT_BASE_MODEL_ID="${MOSS_RT_BASE_MODEL_ID:-OpenMOSS-Team/MOSS-TTS-Realtime}"
export MOSS_RT_CODEC_ID="${MOSS_RT_CODEC_ID:-OpenMOSS-Team/MOSS-Audio-Tokenizer}"
export MOSS_RT_CODEC_BACKEND="${MOSS_RT_CODEC_BACKEND:-auto}"
export MOSS_RT_ONNX_CODEC_DIR="${MOSS_RT_ONNX_CODEC_DIR:-$ROOT/training/weights/MOSS-Audio-Tokenizer-ONNX}"
export MOSS_RT_ONNX_GPU="${MOSS_RT_ONNX_GPU:-true}"
export MOSS_TTS_DIR="${MOSS_TTS_DIR:-$ROOT/third_party/MOSS-TTS}"
# TTFA-first streaming profile (see training/loli_15s/eval/bench/ttfa_baseline_epoch7.json)
export MOSS_RT_INITIAL_TEXT_CHUNK="${MOSS_RT_INITIAL_TEXT_CHUNK:-1}"
export MOSS_RT_STEADY_TEXT_CHUNK="${MOSS_RT_STEADY_TEXT_CHUNK:-4}"
export MOSS_RT_MIN_SAMPLES_FIRST_MS="${MOSS_RT_MIN_SAMPLES_FIRST_MS:-40}"
export MOSS_RT_MIN_SAMPLES_STEADY_MS="${MOSS_RT_MIN_SAMPLES_STEADY_MS:-120}"
export MOSS_RT_DECODER_CHUNK_FRAMES="${MOSS_RT_DECODER_CHUNK_FRAMES:-6}"
export MOSS_RT_DECODER_INITIAL_FRAMES="${MOSS_RT_DECODER_INITIAL_FRAMES:-1}"
export MOSS_RT_DECODER_OVERLAP_FRAMES="${MOSS_RT_DECODER_OVERLAP_FRAMES:-4}"
export MOSS_RT_STREAM_CODEC_BACKEND="${MOSS_RT_STREAM_CODEC_BACKEND:-torch}"
export MOSS_RT_STREAM_DECODER_INITIAL_FRAMES="${MOSS_RT_STREAM_DECODER_INITIAL_FRAMES:-none}"
export MOSS_RT_STREAM_DECODER_OVERLAP_FRAMES="${MOSS_RT_STREAM_DECODER_OVERLAP_FRAMES:-0}"
export MOSS_RT_DRAIN_BATCH_STEPS="${MOSS_RT_DRAIN_BATCH_STEPS:-1}"
export MOSS_RT_PRIME_DELAY="${MOSS_RT_PRIME_DELAY:-true}"
export MOSS_RT_DEVICES=0
export MOSS_QUANTIZE=none
export VOICES_DIR="${VOICES_DIR:-$ROOT/data/voices}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export CUDA_CACHE_MAXSIZE=4294967296
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES="$GPU"

cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT"
export PORT="$PORT"
export HOST="$HOST"

maybe_download_onnx_codec() {
  local backend="${MOSS_RT_CODEC_BACKEND:-auto}"
  [[ "$backend" == "torch" ]] && return 0
  local dest="${MOSS_RT_ONNX_CODEC_DIR:-$ROOT/training/weights/MOSS-Audio-Tokenizer-ONNX}"
  if compgen -G "$dest"/*.onnx >/dev/null 2>&1 || [[ -d "$dest/encoder" ]]; then
    return 0
  fi
  if [[ "${MOSS_RT_DOWNLOAD_ONNX:-}" != "1" ]]; then
    echo "ONNX codec not found at $dest (set MOSS_RT_DOWNLOAD_ONNX=1 to fetch, or MOSS_RT_CODEC_BACKEND=torch)"
    return 0
  fi
  local hf="${MOSS_HF_CLI:-hf}"
  command -v "$hf" >/dev/null 2>&1 || hf="huggingface-cli"
  echo "Downloading MOSS-Audio-Tokenizer ONNX to $dest ..."
  "$hf" download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir "$dest"
}

maybe_download_onnx_codec

PYTHON="${MOSS_RT_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$ROOT/.venv-finetune/bin/python3" ]]; then
    PYTHON="$ROOT/.venv-finetune/bin/python3"
  elif [[ -x "$ROOT/.venv/bin/python3" ]]; then
    PYTHON="$ROOT/.venv/bin/python3"
  else
    PYTHON="python3"
  fi
fi
if ! "$PYTHON" -c "import torch" 2>/dev/null; then
  echo "PyTorch not found. Use Docker instead:"
  echo "  docker compose --env-file .env.moss-realtime up -d moss-realtime"
  exit 1
fi

echo "Starting MOSS-TTS-Realtime on http://${HOST}:${PORT}/ (GPU ${GPU})"
echo "Model: ${MOSS_RT_MODEL_ID}"
echo ""
echo "Smoke test after startup (~3 min first load):"
echo "  curl -s http://127.0.0.1:${PORT}/health | jq ."
echo "  python3 scripts/test_moss_stream.py --api-url http://127.0.0.1:${PORT}"

exec "$PYTHON" -m app.moss_api
