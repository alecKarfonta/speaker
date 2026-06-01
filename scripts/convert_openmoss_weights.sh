#!/bin/bash
# Convert MOSS-TTS-v1.5 HF weights to openmoss GGUF format (backbone + sidecar).
# Run once on a machine with ~30 GB disk and network access to HuggingFace.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/build/llama.cpp}"
OPENMOSS_DIR="${OPENMOSS_DIR:-/build/openmoss}"
OUT_DIR="${OUT_DIR:-/app/weights}"
MOSS_MODEL="${MOSS_MODEL:-OpenMOSS-Team/MOSS-TTS-v1.5}"
CODEC_MODEL="${CODEC_MODEL:-OpenMOSS-Team/MOSS-Audio-Tokenizer}"

# Derive output basename from model id (moss-tts vs moss-tts-v1.5)
if [[ "$MOSS_MODEL" == *"v1.5"* ]]; then
  BASE_NAME="moss-tts-v1.5"
else
  BASE_NAME="moss-tts"
fi

mkdir -p "$OUT_DIR"

pip install --break-system-packages -q safetensors numpy huggingface_hub gguf torch transformers sentencepiece

python3 "$OPENMOSS_DIR/scripts/convert_hf_to_gguf.py" \
    --moss-tts "$MOSS_MODEL" \
    --codec "$CODEC_MODEL" \
    --output "$OUT_DIR/${BASE_NAME}.gguf" \
    --llama-cpp-dir "$LLAMA_DIR" \
    --backbone-dtype f16

QUANT="${LLAMA_QUANTIZE:-}"
if [[ -z "$QUANT" || ! -x "$QUANT" ]]; then
    QUANT="$LLAMA_DIR/build/bin/llama-quantize"
fi
if [[ ! -x "$QUANT" && -x /usr/local/bin/llama-quantize ]]; then
    QUANT="/usr/local/bin/llama-quantize"
fi
if [[ ! -x "$QUANT" ]]; then
    echo "Building llama-quantize..."
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_DIR/build" -j"$(nproc)" --target llama-quantize
fi

"$QUANT" --token-embedding-type f16 \
    "$OUT_DIR/${BASE_NAME}.gguf" \
    "$OUT_DIR/${BASE_NAME}-q8_0.gguf" \
    Q8_0

mv -f "$OUT_DIR/${BASE_NAME}.extras.gguf" "$OUT_DIR/${BASE_NAME}-q8_0.extras.gguf"
echo "✅ Converted weights in $OUT_DIR (${BASE_NAME}-q8_0.gguf)"
