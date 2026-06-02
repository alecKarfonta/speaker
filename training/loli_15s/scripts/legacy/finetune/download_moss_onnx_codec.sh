#!/usr/bin/env bash
# Download MOSS-Audio-Tokenizer ONNX weights for fast realtime decode.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/training/weights/MOSS-Audio-Tokenizer-ONNX}"
HF="${MOSS_HF_CLI:-hf}"
if ! command -v "$HF" >/dev/null 2>&1; then
  HF="huggingface-cli"
fi
echo "Downloading to $DEST ..."
"$HF" download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir "$DEST"
echo "Done. Enable with: MOSS_RT_CODEC_BACKEND=onnx ./scripts/start-moss-realtime.sh"
