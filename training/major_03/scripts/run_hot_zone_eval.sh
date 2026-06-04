#!/usr/bin/env bash
# Story-focused fine sweep around T=1.0 p=0.75 k=50 (hot_100_075 winner).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SPEAKER_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MAJOR="$ROOT/training/major_03"
PYTHON="${BENCH_PYTHON:-$ROOT/.venv-finetune/bin/python3}"
PORT="${MOSS_RT_PORT:-8016}"
API="${MOSS_RT_API:-http://127.0.0.1:${PORT}}"

echo "=== Hot-zone sweep TTS (${API}) ==="
echo "  13 presets × 6 clips (4 story + calm + question)"
MOSS_RT_TRAIN_DIR="$MAJOR" MOSS_RT_API="$API" \
  "$PYTHON" "$SCRIPT_DIR/generate_eval_samples.py" \
  --sweep-hot --wait-health 900

echo "=== ECAPA bench + scored index.html ==="
HOT_ZONE=1 NO_STT=1 "$SCRIPT_DIR/run_voice_similarity_bench.sh"

echo "=== Done ==="
echo "  Listen: file://${MAJOR}/eval/listen/epoch11_hot_zone_sweep/index.html"
echo "  Bench:  file://${MAJOR}/eval/bench/epoch11_hot_zone_sweep/report.html"
