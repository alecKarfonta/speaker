#!/usr/bin/env bash
# Download MOSS-Audio-Tokenizer ONNX weights (+ external .data) for realtime decode.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/models/moss-audio-tokenizer-onnx}"
mkdir -p "$OUT"
BASE="https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX/resolve/main"
for f in encoder.onnx encoder.data decoder.onnx decoder.data; do
  echo "Fetching $f ..."
  curl -L -o "$OUT/$f" "$BASE/$f"
done
echo "ONNX codec ready in $OUT ($(du -sh "$OUT" | cut -f1))"
