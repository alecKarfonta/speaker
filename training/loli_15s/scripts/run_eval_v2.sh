#!/usr/bin/env bash
# Loli v2 eval: emotion listen set, ECAPA bench, streaming cutoff vs epoch-7.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LOLI="$ROOT/training/loli_15s"
API="${MOSS_RT_API:-http://127.0.0.1:8016}"
BENCH="$LOLI/eval/bench"
LISTEN="$LOLI/eval/listen"

mkdir -p "$BENCH" "$LISTEN"

echo "=== Emotion + variety eval WAVs (warm_092 server defaults) ==="
MOSS_RT_API="$API" \
  python3 "$LOLI/scripts/generate_eval_samples.py" \
    --preset emotion \
    --out "$LISTEN/v2_emotion"
MOSS_RT_API="$API" \
  python3 "$LOLI/scripts/generate_eval_samples.py" \
    --out "$LISTEN/v2_variety"

echo "=== ECAPA voice similarity bench ==="
MOSS_RT_TRAIN_DIR="$LOLI" \
  GEN_DIR="$LISTEN/v2_emotion" \
  OUT_DIR="$BENCH/v2_emotion_ecapa" \
  REF_WAV="$ROOT/data/voices/loli/loli_15s.wav" \
  "$SCRIPT_DIR/run_voice_similarity_bench.sh" || echo "Bench skipped (deps/server)"

echo "=== Streaming STT completeness (primary API) ==="
python3 "$LOLI/scripts/verify_streaming_stack.py" \
  --api-url "$API" \
  --out "$BENCH"

if [[ -n "${COMPARE_API:-}" ]]; then
  python3 "$LOLI/scripts/verify_streaming_stack.py" \
    --api-url "$API" \
    --compare-api "$COMPARE_API" \
    --out "$BENCH"
fi

echo "Eval artifacts under $LISTEN and $BENCH"
