#!/usr/bin/env bash
# Build TensorRT engines for MOSS-Audio-Tokenizer ONNX (optional speed tier).
#
# Requires: tensorrt, ONNX weights from ./scripts/download_moss_onnx_codec.sh
# MOSS upstream: moss_audio_tokenizer/trt/build_engine.sh (from MOSS-Audio-Tokenizer repo)
#
# Usage:
#   ./scripts/download_moss_onnx_codec.sh
#   ./scripts/build_moss_onnx_trt_codec.sh [onnx_dir] [out_dir]

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ONNX_DIR="${1:-$ROOT/training/weights/MOSS-Audio-Tokenizer-ONNX}"
OUT_DIR="${2:-$ROOT/training/weights/MOSS-Audio-Tokenizer-TRT}"

if [[ ! -f "$ONNX_DIR/encoder.onnx" || ! -f "$ONNX_DIR/decoder.onnx" ]]; then
  echo "Missing ONNX models in $ONNX_DIR — run ./scripts/download_moss_onnx_codec.sh"
  exit 1
fi

mkdir -p "$OUT_DIR"

if command -v trtexec >/dev/null 2>&1; then
  echo "Building TRT engines with trtexec ..."
  trtexec --onnx="$ONNX_DIR/encoder.onnx" --saveEngine="$OUT_DIR/encoder.trt" --fp16
  trtexec --onnx="$ONNX_DIR/decoder.onnx" --saveEngine="$OUT_DIR/decoder.trt" --fp16
  echo "Engines written to $OUT_DIR"
  echo "Wire TRT in app/rt_codec_onnx.py when moss_audio_tokenizer.trt is available."
else
  echo "trtexec not found. Install TensorRT or use MOSS build_engine.sh from:"
  echo "  https://github.com/OpenMOSS/MOSS-Audio-Tokenizer"
  echo ""
  echo "ONNX GPU path (MOSS_RT_CODEC_BACKEND=auto) is already enabled without TRT."
  exit 1
fi
